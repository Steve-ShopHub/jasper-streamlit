# app.py
# --- COMPLETE FILE (v11.1 - Fixed Timestamp Error) ---

import streamlit as st
import pandas as pd
import os
import io
import base64
import fitz   # PyMuPDF
import google.generativeai as genai
from google.generativeai import types
import sys
import json
import re
import time
# Updated datetime import
from datetime import datetime, timezone
import traceback
from collections import defaultdict # To group questions by category
import plotly.express as px
import google.cloud.firestore
import google.cloud.storage # <-- IMPORT GCS
import google.oauth2.service_account
from google.api_core.exceptions import NotFound # For GCS blob check, conflict
import copy # Needed for deep copying results
from PIL import Image # For Logo
import uuid # For unique IDs

# --- 1. SET PAGE CONFIG (MUST BE FIRST st COMMAND) ---
st.set_page_config(
    layout="wide",
    page_title="JASPER - Extraction and Review",
    page_icon="📄" # Optional: Set an emoji icon
)

# --- Inject custom CSS ---
st.markdown("""
<style>
    /* Reduce vertical padding within expanders */
    .st-emotion-cache-1hboirw, .st-emotion-cache-p729wp {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    /* Reduce spacing between elements */
     div[data-testid="stVerticalBlock"] > div[style*="gap: 1rem;"] {
        gap: 0.5rem !important;
     }
    /* Define a class for the sticky container */
    .sticky-viewer-content {
        position: sticky;
        top: 55px; /* Adjust vertical offset from top */
        z-index: 101; /* Ensure it's above other elements */
        padding-bottom: 1rem; /* Add some space at the bottom */
        background-color: var(--streamlit-background-color); /* Match background */
        border-radius: 0.5rem; /* Optional: slight rounding */
        padding: 1rem; /* Add some padding inside */
        border: 1px solid #ddd; /* Optional: subtle border */
    }
    /* Improve header alignment */
    div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    /* Smaller logo */
    img[alt="Logo"] {
      max-height: 60px; /* Adjust size as needed */
      margin-top: 5px; /* Adjust vertical alignment */
    }
</style>
""", unsafe_allow_html=True)

# --- Load credentials and Initialize Clients ---
try:
    # Firestore
    firestore_key_dict = st.secrets["firestore"]
    firestore_creds = google.oauth2.service_account.Credentials.from_service_account_info(firestore_key_dict)
    db = google.cloud.firestore.Client(credentials=firestore_creds)

    # Google Cloud Storage
    gcs_key_dict = st.secrets.get("gcs", firestore_key_dict) # Reuse Firestore creds if GCS specific aren't provided
    gcs_creds = google.oauth2.service_account.Credentials.from_service_account_info(gcs_key_dict)
    storage_client = google.cloud.storage.Client(credentials=gcs_creds)
    GCS_BUCKET_NAME = st.secrets["gcs_config"]["bucket_name"]
    GCS_PDF_FOLDER = "analysis_pdfs" # Define a folder within the bucket

except KeyError as e:
    st.error(f"❌ Configuration Error: Missing key '{e}' in Streamlit secrets (`secrets.toml`). Please check your configuration.")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to initialize cloud clients: {e}")
    print(traceback.format_exc())
    st.stop()

# --- 2. Configuration & Setup ---
MODEL_NAME = "gemini-2.5-pro-preview-03-25" # Explicitly using 1.5 Pro
MAX_VALIDATION_RETRIES = 1
RETRY_DELAY_SECONDS = 3
LOGO_FILE = "jasper-logo-1.png" # Ensure this file exists
SEARCH_FALLBACK_MIN_LENGTH = 20
SEARCH_PREFIX_MIN_WORDS = 4
CHECKLIST_COLLECTION = "checklists" # Firestore collection for checklists
PROMPT_PLACEHOLDER = "{{QUESTIONS_HERE}}" # Placeholder for injecting questions

# --- Get the absolute path to the directory containing app.py ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(APP_DIR, LOGO_FILE)

# --- Schema Definition ---
ai_response_schema_dict = {
  "type": "array", "description": "List of question analysis results FOR THE QUESTIONS IN THIS SPECIFIC PROMPT.", # Updated description
  "items": {
    "type": "object", "properties": {
      "Question Number": {"type": "integer"}, "Question Category": {"type": "string"}, "Question": {"type": "string"}, "Answer": {"type": "string"}, "Answer Justification": {"type": "string"},
      "Evidence": { "type": "array", "items": { "type": "object", "properties": { "Clause Reference": {"type": "string"}, "Clause Wording": {"type": "string"}, "Searchable Clause Text": {"type": "string"} }, "required": ["Clause Reference", "Clause Wording", "Searchable Clause Text"] } }
    }, "required": ["Question Number", "Question Category", "Question", "Answer", "Answer Justification", "Evidence"]
  }
}
AI_REQUIRED_KEYS = set(ai_response_schema_dict['items']['required'])
AI_EVIDENCE_REQUIRED_KEYS = set(ai_response_schema_dict['items']['properties']['Evidence']['items']['required'])

# --- Excel Column Order ---
EXCEL_COLUMN_ORDER = [ "File Name", "Generation Time", "Checklist Name", "Question Number", "Question Category", "Question", "Answer", "Answer Justification", "Clause References (Concatenated)", "First Searchable Clause Text"]
TEMPLATE_COLUMN_ORDER = ["Question Category", "Question", "Answer Options"] # For download template

# --- System Instruction ---
system_instruction_text = """You are an AI assistant specialized in analyzing legal facility agreements. Carefully read the provided document and answer ONLY the specific questions listed in the user prompt below. Adhere strictly to the requested JSON output schema. Prioritize accuracy and extract evidence directly from the text."""

# --- 3. Helper Function Definitions ---

@st.cache_data(ttl=300, show_spinner="Loading checklists...") # Cache for 5 mins
def load_checklists():
    """Loads checklist names and IDs from Firestore."""
    checklists = {"--- Select Checklist ---": None} # Default option
    try:
        docs = db.collection(CHECKLIST_COLLECTION).stream()
        checklists_found = {}
        for doc in docs:
            data = doc.to_dict()
            name = data.get("name", doc.id) # Use name field, fallback to ID
            # Handle potential duplicates or empty names gracefully
            if name and name != "--- Select Checklist ---":
                checklists_found[name] = doc.id
            elif not name:
                print(f"Warning: Checklist document {doc.id} has no name.") # Log warning

        # Sort checklists by name, keeping the default option first
        sorted_items = sorted(checklists_found.items())
        sorted_checklists = {"--- Select Checklist ---": None}
        sorted_checklists.update(dict(sorted_items))
        return sorted_checklists
    except Exception as e:
        st.error(f"Error loading checklists from Firestore: {e}")
        print(traceback.format_exc())
        return {"--- Select Checklist ---": None, "ERROR": "Could not load"}

@st.cache_data(ttl=60, show_spinner="Loading checklist details...") # Shorter cache for details
def load_checklist_details(checklist_id):
    """Loads the full details (prompt, questions) for a specific checklist ID."""
    if not checklist_id:
        return None
    try:
        doc_ref = db.collection(CHECKLIST_COLLECTION).document(checklist_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            st.error(f"Checklist with ID '{checklist_id}' not found.")
            return None
    except Exception as e:
        st.error(f"Error loading checklist details for ID '{checklist_id}': {e}")
        print(traceback.format_exc())
        return None

def parse_checklist_excel(uploaded_file):
    """Parses the uploaded Excel file for checklist questions."""
    if not uploaded_file:
        return None, "No file provided."
    try:
        # Ensure reading from the start of the file
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file)
        # Validate columns
        required_cols = TEMPLATE_COLUMN_ORDER # Use the template definition
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return None, f"Excel file is missing required columns: {', '.join(missing_cols)}. Expected columns: {', '.join(required_cols)}."

        # Convert to list of dicts, handling potential NaN values and stripping whitespace
        questions = []
        for record in df[required_cols].to_dict('records'):
            cleaned_record = {k: str(v).strip() if pd.notna(v) else "" for k, v in record.items()}
            questions.append(cleaned_record)

        # Basic validation on content
        valid_questions = []
        warnings = []
        for i, q in enumerate(questions):
            row_num = i + 2 # Excel row number (1-based + header)
            if not q.get("Question Category"):
                warnings.append(f"Row {row_num}: Missing 'Question Category'. Using 'Uncategorized'.")
                q["Question Category"] = "Uncategorized"
            if not q.get("Question"):
                warnings.append(f"Row {row_num}: Skipping question due to missing 'Question' text.")
                continue # Skip if question text itself is missing
            valid_questions.append(q)

        if warnings:
            st.warning("Issues found during Excel parsing:\n" + "\n".join(f"- {w}" for w in warnings))

        if not valid_questions:
            return None, "No valid questions found in the Excel file (check for missing 'Question' text)."

        return valid_questions, None # Return questions and no critical error message
    except Exception as e:
        error_msg = f"Error parsing Excel file: {e}"
        print(traceback.format_exc())
        return None, error_msg

def generate_excel_template():
    """Generates an Excel template file in memory."""
    df_template = pd.DataFrame(columns=TEMPLATE_COLUMN_ORDER)
    # Add an example row
    example_row = {"Question Category": "Example Category", "Question": "Is this an example question?", "Answer Options": "Yes; No; Maybe"}
    df_template = pd.concat([df_template, pd.DataFrame([example_row])], ignore_index=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_template.to_excel(writer, index=False, sheet_name='Checklist Template')
    output.seek(0)
    return output.getvalue()

def save_checklist_to_firestore(checklist_name, prompt_template, questions_list):
    """Saves a new or updated checklist to Firestore."""
    if not checklist_name:
        st.error("Checklist Name cannot be empty.")
        return None, False
    if not prompt_template:
        st.error("Prompt Template cannot be empty.")
        return None, False
    if PROMPT_PLACEHOLDER not in prompt_template:
        st.error(f"Prompt Template MUST include the placeholder '{PROMPT_PLACEHOLDER}' where questions should be inserted.")
        return None, False
    if not questions_list:
        st.error("Checklist must contain at least one valid question.")
        return None, False

    # Check for existing checklist with the same name (case-insensitive)
    try:
        query = db.collection(CHECKLIST_COLLECTION).where("name", "==", checklist_name.strip()).limit(1)
        existing = list(query.stream())
        if existing:
            st.error(f"A checklist named '{checklist_name.strip()}' already exists. Please choose a different name.")
            return None, False
    except Exception as e:
        st.warning(f"Could not check for existing checklist name: {e}")
        # Proceed with saving but warn the user

    # Sanitize name for potential use in ID (though we use UUID now)
    sanitized_name = re.sub(r'[^\w\-]+', '_', checklist_name.strip())
    # Generate a unique ID for the checklist
    checklist_id = f"checklist_{sanitized_name}_{uuid.uuid4()}"

    data_to_save = {
        "name": checklist_name.strip(),
        "prompt_template": prompt_template,
        "questions": questions_list,
        "created_at": google.cloud.firestore.SERVER_TIMESTAMP,
        "last_updated_at": google.cloud.firestore.SERVER_TIMESTAMP,
    }

    try:
        doc_ref = db.collection(CHECKLIST_COLLECTION).document(checklist_id)
        doc_ref.create(data_to_save) # Use create to avoid accidentally overwriting
        st.success(f"Checklist '{checklist_name}' saved successfully!")
        # Clear the cache for load_checklists and load details
        load_checklists.clear()
        load_checklist_details.clear()
        return checklist_id, True
    except google.api_core.exceptions.AlreadyExists:
         # This shouldn't happen with UUIDs, but handle just in case
         st.error(f"Error: A checklist with the generated ID '{checklist_id}' already exists. Please try renaming slightly.")
         return None, False
    except Exception as e:
        st.error(f"Failed to save checklist to Firestore: {e}")
        print(traceback.format_exc())
        return None, False

def update_checklist_in_firestore(checklist_id, checklist_name, prompt_template, questions_list):
    """Updates an existing checklist in Firestore."""
    if not checklist_id:
        st.error("Cannot update checklist: Missing ID.")
        return False
    if not checklist_name:
        st.error("Checklist Name cannot be empty.")
        return False
    if not prompt_template:
        st.error("Prompt Template cannot be empty.")
        return False
    if PROMPT_PLACEHOLDER not in prompt_template:
        st.error(f"Prompt Template MUST include the placeholder '{PROMPT_PLACEHOLDER}'.")
        return False
    # Note: We allow updating with an empty question list if the user explicitly does so (though UI prevents this now)

    # Check if name is being changed and if the new name already exists (excluding the current doc)
    try:
        doc_ref = db.collection(CHECKLIST_COLLECTION).document(checklist_id)
        current_doc = doc_ref.get()
        if current_doc.exists:
            current_name = current_doc.to_dict().get("name")
            if current_name != checklist_name.strip():
                query = db.collection(CHECKLIST_COLLECTION).where("name", "==", checklist_name.strip()).limit(1)
                existing = list(query.stream())
                # Make sure the existing one found is not the document we are currently editing
                if existing and existing[0].id != checklist_id:
                     st.error(f"Cannot rename: A different checklist named '{checklist_name.strip()}' already exists.")
                     return False
        else:
            st.error("Cannot update: Checklist to be updated does not exist.")
            return False

    except Exception as e:
        st.warning(f"Could not verify checklist name uniqueness before update: {e}")
        # Proceed with update but warn user

    data_to_update = {
        "name": checklist_name.strip(),
        "prompt_template": prompt_template,
        "questions": questions_list, # Assume questions might have been edited/cleared
        "last_updated_at": google.cloud.firestore.SERVER_TIMESTAMP,
    }

    try:
        doc_ref.update(data_to_update)
        st.success(f"Checklist '{checklist_name}' updated successfully!")
        # Clear the cache for load_checklists and load details
        load_checklists.clear()
        load_checklist_details.clear()
        return True
    except Exception as e:
        st.error(f"Failed to update checklist (ID: {checklist_id}) in Firestore: {e}")
        print(traceback.format_exc())
        return False

def assign_question_numbers(questions_list):
    """Assigns sequential 'assigned_number' to each question dict."""
    numbered_questions = []
    for i, q_dict in enumerate(questions_list):
        # Ensure we don't modify the original dict if it's from state
        q_copy = q_dict.copy()
        q_copy['assigned_number'] = i + 1 # 1-based indexing
        numbered_questions.append(q_copy)
    return numbered_questions

def format_category_questions_for_prompt(questions_in_category):
    """Formats a list of questions (already numbered) for a specific category prompt."""
    prompt_lines = []
    total_question_count = len(questions_in_category)

    if not questions_in_category:
        return "**No questions provided for this category.**" # Handle empty category case

    # Assume questions_in_category is a list of dicts, each already having 'assigned_number'
    category_name = questions_in_category[0].get("Question Category", "Uncategorized") # Get category name from first question
    prompt_lines.append(f"**Please answer the following {total_question_count} questions from the category '{category_name}' based on the provided facility agreement document:**\n")

    for q in questions_in_category:
        q_num = q.get('assigned_number', '???') # Use pre-assigned number
        q_text = q.get('Question', 'MISSING QUESTION TEXT').strip()
        q_opts = q.get('Answer Options', '').strip()

        prompt_lines.append(f"{q_num}. **Question:** {q_text}")
        if q_opts:
            # Format options clearly
            opts_cleaned = str(q_opts).strip()
            # Simple heuristic: if it looks like a list, format nicely
            if any(c in opts_cleaned for c in [',', ';', '\n']) and len(opts_cleaned) > 5:
                 options = [opt.strip() for opt in re.split(r'[;,|\n]+', opts_cleaned) if opt.strip()]
                 if options:
                     prompt_lines.append(f"   **Answer Options:**")
                     for opt in options:
                         prompt_lines.append(f"     - {opt}")
                 elif opts_cleaned:
                      prompt_lines.append(f"   **Answer Guidance:** {opts_cleaned}")
            elif opts_cleaned:
                 prompt_lines.append(f"   **Answer Guidance:** {opts_cleaned}")
        prompt_lines.append("") # Add empty line for spacing

    formatted_string = "\n".join(prompt_lines)
    return formatted_string

def validate_ai_data(ai_response_data, expected_questions_subset):
    """Validates AI response against the schema and a SUBSET of expected questions.
       Returns (validated_data, issues_list).
    """
    if not isinstance(ai_response_data, list):
        # If the schema expects an array, but we don't get one, that's a critical schema violation.
        return None, [f"CRITICAL SCHEMA VIOLATION: AI Response is not a list (received type: {type(ai_response_data).__name__}). The schema requires a JSON array."]

    validated_data = []
    issues_list = []
    expected_q_nums = {q['assigned_number'] for q in expected_questions_subset}
    expected_q_details = {q['assigned_number']: q for q in expected_questions_subset}
    found_q_nums = set()
    total_expected_count = len(expected_questions_subset)

    if not expected_questions_subset:
        # If we didn't expect any questions (e.g., empty category), but got data, flag it.
        if ai_response_data:
             issues_list.append("Validation Warning: Received analysis data when no questions were expected for this prompt/category.")
             # Decide whether to keep or discard this unexpected data. Let's discard for now.
             return [], issues_list
        else:
            # Expected empty, got empty. This is fine.
            return [], issues_list

    for index, item in enumerate(ai_response_data):
        q_num = item.get('Question Number')
        q_num_str = f"Q#{q_num}" if q_num is not None else f"Item Index {index}"
        is_outer_valid = True
        item_issues = [] # Collect issues for this specific item

        if not isinstance(item, dict):
            item_issues.append(f"Item Index {index}: Item is not a dictionary.")
            is_outer_valid = False
            # Do not continue validation for this malformed item, record issues and skip
            issues_list.append(f"Item {q_num_str} Validation Issues:")
            issues_list.extend([f"  - {issue}" for issue in item_issues])
            continue

        # Check Question Number validity
        if not isinstance(q_num, int):
            item_issues.append(f"Item Index {index}: 'Question Number' is missing or not an integer.")
            is_outer_valid = False
        elif q_num not in expected_q_nums:
             # AI returned a question number that wasn't asked in *this specific prompt*.
             item_issues.append(f"{q_num_str}: Unexpected Question Number found (was not asked in this prompt).")
             is_outer_valid = False
        else:
            # Check for duplicate question numbers within this response batch
            if q_num in found_q_nums:
                 item_issues.append(f"{q_num_str}: Duplicate Question Number found in this response batch.")
                 is_outer_valid = False
            else:
                found_q_nums.add(q_num)
                # Check if category and question text match expectation (optional but good)
                expected_q_data = expected_q_details.get(q_num)
                if expected_q_data:
                    # Note: AI might hallucinate a slightly different category name sometimes, treat as warning.
                    if item.get("Question Category") != expected_q_data.get("Question Category"):
                         item_issues.append(f"Warning: Question Category mismatch (Expected: '{expected_q_data.get('Question Category')}', Got: '{item.get('Question Category')}')")
                    # Also treat question text mismatch as a warning unless it's wildly different
                    if item.get("Question") != expected_q_data.get("Question"):
                         item_issues.append(f"Warning: Question text mismatch (Expected: '{expected_q_data.get('Question')[:50]}...', Got: '{item.get('Question', '')[:50]}...')")

        # Check for required keys at the top level
        missing_outer_keys = AI_REQUIRED_KEYS - set(item.keys())
        if missing_outer_keys:
            item_issues.append(f"Missing required top-level keys: {missing_outer_keys}")
            is_outer_valid = False

        # Validate Evidence structure (if present or required)
        evidence_list = item.get("Evidence")
        # Check if 'Evidence' is required by schema AND (missing OR not a list)
        if "Evidence" in AI_REQUIRED_KEYS and ("Evidence" not in item or not isinstance(evidence_list, list)):
            item_issues.append(f"Required 'Evidence' field is missing or not a list (found type: {type(evidence_list).__name__}).")
            is_outer_valid = False
        elif "Evidence" in item and isinstance(evidence_list, list): # Evidence is present and is a list (could be optional or required)
            for ev_index, ev_item in enumerate(evidence_list):
                ev_id_str = f"Ev[{ev_index}]"
                if not isinstance(ev_item, dict):
                    item_issues.append(f"{ev_id_str}: Evidence item is not a dictionary.")
                    is_outer_valid = False; continue # Skip further checks for this evidence item
                missing_ev_keys = AI_EVIDENCE_REQUIRED_KEYS - set(ev_item.keys())
                if missing_ev_keys:
                    item_issues.append(f"{ev_id_str}: Missing required evidence keys: {missing_ev_keys}")
                    is_outer_valid = False
                # Check types of required evidence keys (only if key is required AND present)
                for key, expected_type in [("Clause Reference", str), ("Clause Wording", str), ("Searchable Clause Text", str)]:
                     if key in AI_EVIDENCE_REQUIRED_KEYS and key in ev_item and not isinstance(ev_item.get(key), expected_type):
                         item_issues.append(f"{ev_id_str}: Key '{key}' has incorrect type (expected {expected_type.__name__}, got {type(ev_item.get(key)).__name__}).")
                         is_outer_valid = False
                # Check if searchable text is reasonably populated if present
                search_text = ev_item.get("Searchable Clause Text")
                if search_text is not None and not search_text.strip():
                    # Flag as warning, not necessarily invalidating the whole item
                    item_issues.append(f"{ev_id_str}: Warning: 'Searchable Clause Text' is present but empty or only whitespace.")


        if is_outer_valid:
            # Add original expected question data for reference if validation passes
            item['_expected_question_data'] = expected_q_details.get(q_num) # Add reference to the input question data
            validated_data.append(item)
        else:
            # If the item failed validation, add its specific issues to the main issues list
             issues_list.append(f"Item {q_num_str} Validation Issues:")
             issues_list.extend([f"  - {issue}" for issue in item_issues])


    # Check: Were all questions expected IN THIS SUBSET answered?
    missing_q_nums_in_subset = expected_q_nums - found_q_nums
    if missing_q_nums_in_subset:
        issues_list.append(f"Validation: Missing answers for expected Question Numbers in this prompt: {sorted(list(missing_q_nums_in_subset))}")

    # Add a summary header if issues were found
    if issues_list:
        issues_list.insert(0, f"Validation Issues Found ({len(validated_data)} of {total_expected_count} items passed validation for this prompt):")

    # Return validated data (even if empty) or None only if critical error occurred at the start
    # If the input was a list but validation failed for all items, return empty list + issues.
    if validated_data is None: return None, issues_list # Should only happen if input wasn't a list initially
    else: return validated_data, issues_list


def generate_checklist_analysis_per_category(checklist_prompt_template, all_checklist_questions, uploaded_file_ref, status_placeholder, api_key_to_use, gen_config_params, progress_bar):
    """
    Generates analysis by sending one prompt per question category.
    Returns (all_results, overall_status, all_warnings).
    """
    try:
        genai.configure(api_key=api_key_to_use)
    except Exception as config_err:
        status_placeholder.error(f"❌ Invalid API Key provided or configuration failed: {config_err}")
        return None, "Failed", [f"Invalid API Key or config error: {config_err}"]

    # --- Preparation ---
    status_placeholder.info(f"🔄 Preparing prompts for analysis...")
    all_analysis_warnings = []
    all_validated_data = []
    category_statuses = {} # Track status per category

    # 1. Assign unique numbers to all questions first
    try:
        numbered_questions = assign_question_numbers(all_checklist_questions)
        if not numbered_questions:
            raise ValueError("No valid questions found in the checklist.")
    except Exception as e:
        status_placeholder.error(f"❌ Error preparing questions: {e}")
        return None, "Failed", [f"Error preparing questions: {e}"]

    # 2. Group numbered questions by category
    grouped_questions = defaultdict(list)
    for q in numbered_questions:
        grouped_questions[q.get("Question Category", "Uncategorized")].append(q)

    total_categories = len(grouped_questions)
    processed_categories = 0
    if total_categories == 0:
         status_placeholder.warning("⚠️ No question categories found. Analysis cannot proceed.")
         return [], "Skipped", ["No question categories found."]

    status_placeholder.info(f"✅ Found {len(numbered_questions)} questions across {total_categories} categories.")

    # --- Setup GenAI Model and Config (once) ---
    try:
        model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction_text)
        generation_config = types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=ai_response_schema_dict,
            temperature=gen_config_params['temperature'],
            top_p=gen_config_params['top_p'],
            top_k=gen_config_params['top_k']
        )
        safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
    except Exception as model_setup_err:
        status_placeholder.error(f"❌ Failed to set up Generative Model: {model_setup_err}")
        return None, "Failed", [f"Model setup error: {model_setup_err}"]


    # --- Process Each Category ---
    sorted_category_names = sorted(grouped_questions.keys()) # Process alphabetically

    for category_name in sorted_category_names:
        questions_in_category = grouped_questions[category_name]
        processed_categories += 1
        category_progress = 15 + int(65 * (processed_categories / total_categories)) # Progress from 15% to 80%
        progress_bar.progress(category_progress, text=f"Processing Category {processed_categories}/{total_categories}: '{category_name}'...")
        status_placeholder.info(f"⏳ Processing Category: '{category_name}' ({len(questions_in_category)} questions)...")

        category_validated_data = None
        category_status = "Failed"
        category_warnings = []

        try:
            # 3. Format questions for this category only
            formatted_questions_str = format_category_questions_for_prompt(questions_in_category)

            # 4. Integrate into the prompt template
            if PROMPT_PLACEHOLDER not in checklist_prompt_template:
                 raise ValueError(f"Prompt template is missing the required placeholder: {PROMPT_PLACEHOLDER}")
            category_prompt_for_api = checklist_prompt_template.replace(PROMPT_PLACEHOLDER, formatted_questions_str)

            # Add final instruction about the JSON schema for this specific category
            final_instruction = f"\n\n**Final Instruction:** Ensure the final output is a valid JSON array containing an object for **all** questions listed above for category '{category_name}'. Each question object must follow the specified schema precisely, including all required keys (`Question Number`, `Question Category`, `Question`, `Answer`, `Answer Justification`, `Evidence`). The `Question Number` must match the number assigned in the list above. Ensure the `Evidence` array contains objects with *all* required keys (`Clause Reference`, `Clause Wording`, `Searchable Clause Text`) or is an empty array (`[]`) if no direct evidence applies. Double-check this structure carefully."
            category_prompt_for_api += final_instruction

            # 5. Call API and Validate (with retries per category)
            for attempt in range(1, MAX_VALIDATION_RETRIES + 2):
                if attempt > 1:
                    status_placeholder.info(f"⏳ Retrying '{category_name}' (Attempt {attempt}/{MAX_VALIDATION_RETRIES+1})..."); time.sleep(RETRY_DELAY_SECONDS)
                try:
                    if not uploaded_file_ref or not hasattr(uploaded_file_ref, 'name'):
                        raise ValueError("Invalid or missing uploaded file reference for GenAI call.")

                    contents = [uploaded_file_ref, category_prompt_for_api]
                    status_placeholder.info(f"🧠 Calling AI for '{category_name}' (Attempt {attempt})...")
                    response = model.generate_content(contents=contents, generation_config=generation_config, safety_settings=safety_settings, request_options={'timeout': 900}) # 15 min timeout

                    parsed_ai_data = None; validated_ai_data_subset = None; validation_issues = []
                    status_placeholder.info(f"🔍 Processing response for '{category_name}' (Attempt {attempt})...")

                    if response.parts:
                        full_response_text = response.text
                        try:
                            # Handle potential markdown ```json ... ``` wrapping
                            match = re.search(r"```json\s*([\s\S]*?)\s*```", full_response_text, re.IGNORECASE | re.DOTALL)
                            json_text = match.group(1).strip() if match else full_response_text.strip()

                            if not json_text: raise json.JSONDecodeError("Extracted JSON content is empty.", json_text, 0)
                            parsed_ai_data = json.loads(json_text)
                            status_placeholder.info(f"✔️ Validating structure for '{category_name}'...")
                            # IMPORTANT: Validate against the subset of questions for this category
                            validated_ai_data_subset, validation_issues = validate_ai_data(parsed_ai_data, questions_in_category)

                            # Add validation issues to this category's warnings
                            if validation_issues:
                                category_warnings.extend(validation_issues)

                            # --- Check validation result for this category ---
                            if validated_ai_data_subset is not None and len(validated_ai_data_subset) > 0:
                                # Check if all questions *for this category* were answered
                                missing_answers_issue = next((issue for issue in validation_issues if "Missing answers" in issue), None)
                                if missing_answers_issue:
                                     status_placeholder.warning(f"⚠️ Partial Success for '{category_name}': Some questions missed. (Attempt {attempt}).")
                                     category_validated_data = validated_ai_data_subset
                                     category_status = "Partial Success"
                                else:
                                     status_placeholder.info(f"✅ Validation successful for '{category_name}'.")
                                     category_validated_data = validated_ai_data_subset
                                     category_status = "Success"
                                break # Exit retry loop for this category on success/partial success

                            elif validated_ai_data_subset is not None and len(validated_ai_data_subset) == 0 and not validation_issues:
                                # Valid schema, but empty list returned.
                                status_placeholder.warning(f"ℹ️ AI returned an empty list for '{category_name}' (Attempt {attempt}).")
                                category_warnings.append(f"AI returned an empty list for category '{category_name}'.")
                                category_validated_data = [] # Store empty list
                                category_status = "Success (Empty)"
                                break # Exit retry loop, treat as success but empty

                            else: # Validation failed critically or produced no valid data
                                error_msg = f"Validation failed for '{category_name}'. Issues: {validation_issues}"
                                status_placeholder.warning(f"⚠️ {error_msg} (Attempt {attempt}).")
                                # Continue retry loop if possible

                            # This case handles where validate_ai_data returns None (critical schema error)
                            if validated_ai_data_subset is None:
                                category_warnings.append(f"CRITICAL validation error for '{category_name}': Response did not match schema base structure (e.g., not a list).")
                                category_status = "Failed (Validation)"
                                break # Do not retry critical validation error

                        except json.JSONDecodeError as json_err:
                            error_msg = f"JSON Decode Error for '{category_name}' (Attempt {attempt}): {json_err}. Raw text: '{full_response_text[:500]}...'"
                            st.error(f"{error_msg}"); st.code(full_response_text, language='text')
                            category_warnings.append(error_msg); category_status = "Failed (JSON Error)"
                            # Don't retry JSON errors unless they seem transient
                        except Exception as parse_validate_err:
                            error_msg = f"Parsing/Validation Error for '{category_name}' (Attempt {attempt}): {type(parse_validate_err).__name__}: {parse_validate_err}"
                            st.error(error_msg); category_warnings.append(error_msg); print(traceback.format_exc()); category_status = "Failed (Validation)"
                    else: # No response parts (blocked, etc.)
                        block_reason = "Unknown"; finish_reason = "Unknown"; safety_ratings = None
                        try: # Use getattr for safer access
                            if response.prompt_feedback:
                                block_reason_obj = getattr(response.prompt_feedback, 'block_reason', 'Unknown')
                                block_reason = block_reason_obj.name if hasattr(block_reason_obj, 'name') else str(block_reason_obj)
                                safety_ratings = getattr(response.prompt_feedback, 'safety_ratings', None)
                            # Access finish_reason from the first candidate if it exists
                            if response.candidates:
                                finish_reason_obj = getattr(response.candidates[0], 'finish_reason', 'Unknown')
                                finish_reason = finish_reason_obj.name if hasattr(finish_reason_obj, 'name') else str(finish_reason_obj)
                                # Also check candidate safety ratings if prompt feedback didn't have them
                                if not safety_ratings:
                                     safety_ratings = getattr(response.candidates[0], 'safety_ratings', None)
                        except Exception as resp_parse_err:
                            print(f"Warning: Could not parse response feedback/candidate details: {resp_parse_err}")

                        safety_info = f" Ratings: {safety_ratings}" if safety_ratings else ""
                        warn_msg = f"API Issue for '{category_name}' (Attempt {attempt}): Finish: {finish_reason}, Block: {block_reason}.{safety_info}";
                        if finish_reason == "SAFETY": st.error(warn_msg); category_status = "Failed (Safety Block)"
                        else: st.warning(warn_msg); category_status = "Failed (API Issue)"
                        category_warnings.append(warn_msg)
                        if finish_reason in ["SAFETY", "MAX_TOKENS"]: break # Don't retry safety/token issues

                except types.StopCandidateException as sce: error_msg = f"Generation Stopped Error for '{category_name}' (Attempt {attempt}): {sce}."; st.error(error_msg); category_warnings.append(error_msg); print(traceback.format_exc()); category_status = "Failed (API Error)"
                except google.api_core.exceptions.GoogleAPIError as api_err:
                     err_str = str(api_err).lower()
                     if "api key not valid" in err_str or "permission denied" in err_str or "quota exceeded" in err_str:
                          st.error(f"Google API Error for '{category_name}' (Attempt {attempt}): {api_err}. Stopping analysis.")
                          category_warnings.append(f"API Error (Stopping): {api_err}"); category_status = "Failed (API Error)"
                          raise # Re-raise critical API errors to stop the whole process
                     else: error_msg = f"Google API Error for '{category_name}' (Attempt {attempt}): {api_err}."; st.error(error_msg); category_warnings.append(error_msg); print(traceback.format_exc()); category_status = "Failed (API Error)"
                except Exception as e: error_msg = f"Processing Error for '{category_name}' (Attempt {attempt}): {type(e).__name__}: {e}"; st.error(error_msg); category_warnings.append(error_msg); print(traceback.format_exc()); category_status = "Failed (Processing Error)"

                # If category processing succeeded (even partially/empty), break retry loop for this category
                if category_validated_data is not None:
                    break

            # --- After retry loop for category ---
            if category_validated_data is not None:
                 all_validated_data.extend(category_validated_data) # Add results to the main list
                 if category_status == "Success": status_placeholder.success(f"✅ Category '{category_name}' processed successfully.")
                 elif category_status == "Partial Success": status_placeholder.warning(f"⚠️ Category '{category_name}' processed with missing answers.")
                 elif category_status == "Success (Empty)": status_placeholder.info(f"ℹ️ Category '{category_name}' processed, returned empty list.")
            else:
                 status_placeholder.error(f"❌ Failed to get valid response for category '{category_name}' after {attempt} attempts.")
                 category_warnings.append(f"Failed to get valid response for '{category_name}' after {MAX_VALIDATION_RETRIES + 1} attempts.")
                 category_status = "Failed" # Ensure status reflects failure

            category_statuses[category_name] = {"status": category_status, "warnings": category_warnings}
            # Prefix warnings with category name for clarity in the overall summary
            all_analysis_warnings.extend([f"Category '{category_name}': {w}" for w in category_warnings])

        except ValueError as ve: # Catch prompt generation errors etc.
            status_placeholder.error(f"❌ Error preparing prompt for category '{category_name}': {ve}")
            category_statuses[category_name] = {"status": "Failed (Setup Error)", "warnings": [str(ve)]}
            all_analysis_warnings.append(f"Category '{category_name}' Setup Error: {ve}")
            # Continue with other categories
        except Exception as cat_err:
            status_placeholder.error(f"❌ Unexpected Error processing category '{category_name}': {cat_err}")
            category_statuses[category_name] = {"status": "Failed (Critical Error)", "warnings": [str(cat_err)]}
            all_analysis_warnings.append(f"Category '{category_name}' Critical Error: {cat_err}")
            print(f"Critical error in category loop for {category_name}: {traceback.format_exc()}")
            # Continue to next category unless it was a critical API error handled above which would have raised

    # --- Final Aggregation and Status Determination ---
    progress_bar.progress(85, text="Aggregating results...")
    status_placeholder.info("📊 Aggregating results from all categories...")

    # Check if all *original* questions were answered across all categories
    final_found_q_nums = {item.get('Question Number') for item in all_validated_data if isinstance(item, dict)}
    original_q_nums = {q.get('assigned_number') for q in numbered_questions}
    missing_overall_q_nums = original_q_nums - final_found_q_nums
    if missing_overall_q_nums:
        warn_msg = f"Overall Analysis: Missing answers for expected Question Numbers: {sorted(list(missing_overall_q_nums))}"
        all_analysis_warnings.append(warn_msg)
        status_placeholder.warning(warn_msg)

    # Determine overall status based on category statuses
    overall_status = "Success" # Assume success initially
    any_partial = False
    any_failed = False
    all_empty_or_skipped = True # Assume all are empty/skipped until proven otherwise

    if not category_statuses: # Handle case where no categories were processed (e.g., initial error)
        overall_status = "Failed"
        if not all_analysis_warnings: all_analysis_warnings.append("No categories processed.")
    else:
        for cat_name, cat_info in category_statuses.items():
            status = cat_info.get('status', 'Unknown')
            if status.startswith("Failed"): any_failed = True
            if status == "Partial Success": any_partial = True
            # If any category succeeded or was partial, not all are empty/skipped
            if status in ["Success", "Partial Success"]: all_empty_or_skipped = False

        if any_failed: overall_status = "Failed"
        elif any_partial: overall_status = "Partial Success"
        # If all categories were either Success(Empty) or Skipped (e.g., no questions), and no data was actually produced
        elif all_empty_or_skipped and not all_validated_data: overall_status = "Success (Empty)"
        # If we have *some* data, but no failures or partials, it's a success.
        elif all_validated_data and not any_failed and not any_partial: overall_status = "Success"
        # If no data, and not all empty/skipped (meaning some should have run but didn't), it's a failure.
        elif not all_validated_data and not all_empty_or_skipped: overall_status = "Failed"


    # Display final status message
    if overall_status == "Success":
        status_placeholder.success("✅ Analysis completed successfully across all categories.")
    elif overall_status == "Partial Success":
        status_placeholder.warning("⚠️ Analysis completed, but some categories had partial success or missing answers.")
    elif overall_status == "Success (Empty)":
         status_placeholder.info("ℹ️ Analysis completed, but the AI returned no results across all categories.")
    elif overall_status == "Skipped":
         status_placeholder.info("ℹ️ Analysis skipped (e.g., no categories or questions).")
    else: # Failed
        status_placeholder.error("❌ Analysis failed for one or more categories.")

    progress_bar.progress(90, text="Analysis aggregated.")
    # Return all collected data, the determined overall status, and aggregated warnings
    return all_validated_data, overall_status, all_analysis_warnings


# --- PDF Search/Render Functions ---
@st.cache_data(show_spinner=False)
def find_text_in_pdf(_pdf_bytes, search_text):
    """Searches PDF. Returns (first_page_found, instances_on_first_page, term_used, status_msg, all_findings)"""
    if not _pdf_bytes or not search_text: return None, None, None, "Invalid input (PDF bytes or search text missing).", None

    doc = None # Initialize doc to None
    search_text_cleaned = search_text.strip()
    words = search_text_cleaned.split()
    num_words = len(words)
    search_attempts = []

    # --- Build Search Terms List (unchanged) ---
    term_full = search_text_cleaned
    if term_full: search_attempts.append({'term': term_full, 'desc': "full text"})
    sentences = re.split(r'(?<=[.?!])\s+', term_full); term_sentence = sentences[0].strip() if sentences else ""
    if term_sentence and len(term_sentence) >= SEARCH_FALLBACK_MIN_LENGTH and term_sentence != term_full:
        search_attempts.append({'term': term_sentence, 'desc': "first sentence"})
    if num_words >= 10:
        term_10 = ' '.join(words[:10])
        if term_10 != term_full and term_10 != term_sentence:
            search_attempts.append({'term': term_10, 'desc': "first 10 words"})
    if num_words >= SEARCH_PREFIX_MIN_WORDS:
        term_5 = ' '.join(words[:5])
        if term_5 != term_full and term_5 != term_sentence and term_5 != (search_attempts[-1]['term'] if search_attempts else None):
             search_attempts.append({'term': term_5, 'desc': "first 5 words"})
    if num_words < SEARCH_PREFIX_MIN_WORDS and len(term_full) >= SEARCH_FALLBACK_MIN_LENGTH and not any(term_full == a['term'] for a in search_attempts):
        search_attempts.append({'term': term_full, 'desc': "short text fallback"})

    # --- Execute Search Attempts ---
    try:
        doc = fitz.open(stream=_pdf_bytes, filetype="pdf")
        search_flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES

        for attempt in search_attempts:
            term = attempt['term']; desc = attempt['desc']; findings_for_term = []
            if not term: continue # Skip empty terms

            if not doc or doc.is_closed:
                 raise ValueError("Document closed unexpectedly during search attempts.")

            for page_index in range(doc.page_count):
                page = doc.load_page(page_index);
                try:
                    # *** CORRECTED LINE: Removed hit_max=0 ***
                    instances = page.search_for(term, flags=search_flags, quads=False)
                    if instances: findings_for_term.append((page_index + 1, instances))
                except Exception as search_page_err:
                    # Print warning but continue searching other pages/terms
                    print(f"WARN: Error searching page {page_index+1} for '{term}': {search_page_err}")
                    continue

            if findings_for_term:
                first_page_found = findings_for_term[0][0]
                instances_on_first_page = findings_for_term[0][1]
                status = ""
                if len(findings_for_term) == 1:
                    status = f"✅ Found using '{desc}' on page {first_page_found} ({len(instances_on_first_page)} instance(s))."
                else:
                    pages_found = sorted([f[0] for f in findings_for_term])
                    total_matches = sum(len(f[1]) for f in findings_for_term)
                    status = f"⚠️ Found {total_matches} matches using '{desc}' on multiple pages: {pages_found}. Showing first match on page {first_page_found}."
                return first_page_found, instances_on_first_page, term, status, findings_for_term

        # If loop finishes without finding anything
        tried_descs = [a['desc'] for a in search_attempts if a['term']];
        return None, None, None, f"❌ Text not found (tried methods: {', '.join(tried_descs)}).", None

    except Exception as e:
        print(f"ERROR searching PDF: {e}\n{traceback.format_exc()}")
        return None, None, None, f"❌ Error during PDF search: {e}", None

    finally:
        # Ensure doc is closed *only* here if it was successfully opened and is not already closed
        if doc and not doc.is_closed:
             try:
                 doc.close()
             except Exception as close_err:
                 print(f"WARN: Error closing PDF in finally block: {close_err}")


@st.cache_data(show_spinner=False, max_entries=20) # Cache rendered pages
def render_pdf_page_to_image(_pdf_bytes_hash, page_number, highlight_instances_tuple=None, dpi=150):
    """Renders PDF page to PNG image bytes, applying highlights. Uses hash and tuple for caching.
       Returns (image_bytes, status_msg)."""
    # Note: _pdf_bytes_hash is not used directly but forces cache invalidation if PDF changes.
    # highlight_instances_tuple is used because lists/fitz.Rect aren't hashable.

    if not st.session_state.pdf_bytes or page_number < 1: return None, "Invalid input for rendering (PDF bytes missing or invalid page number)."
    doc = None; image_bytes = None; render_status_message = f"Rendered page {page_number}."
    try:
        # Use the actual bytes from session state
        doc = fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf"); page_index = page_number - 1
        if page_index < 0: # Allow rendering page 0 if page_number is 1
             page_index = 0
             if page_number != 1: # Check if page_number was explicitly invalid (e.g., 0 or negative)
                  doc.close(); return None, f"Page number {page_number} is invalid. Must be 1 or greater."

        if page_index >= doc.page_count:
             doc.close(); return None, f"Page number {page_number} is out of range (Total pages: {doc.page_count})."

        page = doc.load_page(page_index); highlight_applied_count = 0
        # Convert tuple back to list of Rects if needed
        highlight_instances = [fitz.Rect(r) for r in highlight_instances_tuple] if highlight_instances_tuple else None

        if highlight_instances:
            try:
                for inst in highlight_instances:
                    # Ensure inst is a valid Rect object before highlighting
                    if isinstance(inst, fitz.Rect) and not inst.is_empty and inst.is_valid:
                        highlight = page.add_highlight_annot(inst)
                        if highlight:
                            highlight.set_colors(stroke=fitz.utils.getColor("yellow"))
                            highlight.set_opacity(0.4); highlight.update(); highlight_applied_count += 1
                        else: print(f"WARN: Failed to add highlight annotation for instance: {inst} on page {page_number}")
                    elif isinstance(inst, (tuple, list)) and len(inst) == 4: # Attempt to create Rect from tuple/list
                         try:
                             rect_from_tuple = fitz.Rect(inst)
                             if not rect_from_tuple.is_empty and rect_from_tuple.is_valid:
                                 highlight = page.add_highlight_annot(rect_from_tuple)
                                 if highlight:
                                     highlight.set_colors(stroke=fitz.utils.getColor("yellow"))
                                     highlight.set_opacity(0.4); highlight.update(); highlight_applied_count += 1
                                 else: print(f"WARN: Failed to add highlight annotation for tuple instance: {inst} on page {page_number}")
                             else: print(f"WARN: Invalid Rect from tuple instance: {inst} on page {page_number}")
                         except Exception as rect_conv_err:
                              print(f"WARN: Could not convert instance {inst} to Rect on page {page_number}: {rect_conv_err}")
                    else:
                        print(f"WARN: Skipping invalid or non-Rect highlight instance: {type(inst)} on page {page_number}")

                if highlight_applied_count > 0: render_status_message = f"Rendered page {page_number} with {highlight_applied_count} highlight(s)."
                elif highlight_instances: render_status_message = f"Rendered page {page_number}, but no valid highlights applied from provided instances."
            except Exception as highlight_err: print(f"ERROR applying highlights on page {page_number}: {highlight_err}\n{traceback.format_exc()}"); render_status_message = f"⚠️ Error applying highlights: {highlight_err}"

        pix = page.get_pixmap(dpi=dpi, alpha=False); image_bytes = pix.tobytes("png")
    except Exception as e: print(f"ERROR rendering page {page_number}: {e}\n{traceback.format_exc()}"); render_status_message = f"❌ Error rendering page {page_number}: {e}"; image_bytes = None
    finally:
        if doc and not doc.is_closed:
            try: doc.close()
            except Exception: pass # Ignore errors during final close
    return image_bytes, render_status_message

# --- GCS Upload/Download Functions ---
def upload_to_gcs(bucket_name, source_bytes, destination_blob_name, status_placeholder=None):
    """Uploads bytes to GCS bucket."""
    if status_placeholder: status_placeholder.info(f"☁️ Uploading PDF to Google Cloud Storage...")
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(source_bytes, content_type='application/pdf', timeout=120) # Add timeout
        gcs_path = f"gs://{bucket_name}/{destination_blob_name}"
        if status_placeholder: status_placeholder.info(f"☁️ PDF successfully uploaded to {gcs_path}")
        print(f"Uploaded to {gcs_path}")
        return gcs_path
    except Exception as e:
        if status_placeholder: status_placeholder.error(f"❌ GCS Upload Failed: {e}")
        print(f"GCS Upload Error: {e}\n{traceback.format_exc()}")
        raise # Re-raise the exception to be caught by the main analysis loop

def download_from_gcs(bucket_name, source_blob_name):
    """Downloads a blob from the bucket."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        pdf_bytes = blob.download_as_bytes(timeout=120) # Add timeout
        return pdf_bytes
    except NotFound:
        st.error(f"❌ Error: PDF file not found in Google Cloud Storage at gs://{bucket_name}/{source_blob_name}")
        return None
    except Exception as e:
        st.error(f"❌ Failed to download PDF from GCS: {e}")
        print(f"GCS Download Error: {e}\n{traceback.format_exc()}")
        return None

def reset_app_state():
    """Resets the session state for a new analysis or clearing history."""
    # Preserve API key and AI params across resets
    current_api_key = st.session_state.get('api_key', None)
    current_temp = st.session_state.get('ai_temperature', 0.0)
    current_top_p = st.session_state.get('ai_top_p', 0.05)
    current_top_k = st.session_state.get('ai_top_k', 1)
    current_available_checklists = st.session_state.get('available_checklists', {})

    # Clear most state variables
    keys_to_reset = [
        'pdf_bytes', 'pdf_display_ready', 'analysis_results', 'analysis_complete',
        'processing_in_progress', 'current_page', 'run_status_summary',
        'excel_data', 'search_trigger', 'last_search_result', 'show_wording_states',
        'viewing_history', 'history_filename', 'history_timestamp', 'current_filename',
        'load_history_id', 'run_key', 'pdf_bytes_hash',
        'selected_checklist_id', 'selected_checklist_name',
        'current_checklist_prompt', 'current_checklist_questions',
        'new_checklist_name', 'uploaded_checklist_file_obj', 'checklist_action',
        'history_checklist_name' # Clear history specific data
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key] # Remove to allow reinitialization with defaults

    # Re-initialize with defaults (or preserved values)
    initialize_session_state()

    # Restore preserved values
    st.session_state.api_key = current_api_key
    st.session_state.ai_temperature = current_temp
    st.session_state.ai_top_p = current_top_p
    st.session_state.ai_top_k = current_top_k
    st.session_state.available_checklists = current_available_checklists # Restore loaded checklists


# --- 4. Initialize Session State ---
def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    state_defaults = {
        'show_wording_states': defaultdict(bool),
        'current_page': 1,
        'analysis_results': None,
        'pdf_bytes': None,
        'pdf_bytes_hash': None, # To help with caching PDF-dependent functions
        'pdf_display_ready': False,
        'processing_in_progress': False,
        'analysis_complete': False,
        'run_key': 0,
        'run_status_summary': {}, # Default to empty dict for new summary format
        'excel_data': None,
        'search_trigger': None,
        'last_search_result': None,
        'api_key': None,
        'load_history_id': None,
        'viewing_history': False,
        'history_filename': None,
        'history_timestamp': None,
        'history_checklist_name': None, # Name of checklist used in history
        'current_filename': None,
        'ai_temperature': 0.2, # AI Temperature
        'ai_top_p': 0.95,      # AI Top-P
        'ai_top_k': 40,        # AI Top-K
        # --- Checklist State ---
        'selected_checklist_id': None,
        'selected_checklist_name': None,
        'current_checklist_prompt': "", # Prompt for the selected/new checklist
        'current_checklist_questions': [], # Questions list for selected/new checklist
        'new_checklist_name': "", # For the create new form
        'uploaded_checklist_file_obj': None, # Holds the uploaded file object
        'checklist_action': None, # Tracks if user wants to create/edit ('create', 'selected')
        'available_checklists': {}, # Stores loaded checklist names/IDs
    }
    for key, default_value in state_defaults.items():
        if key not in st.session_state:
            # Use deepcopy for mutable defaults like defaultdict
            st.session_state[key] = copy.deepcopy(default_value) if isinstance(default_value, (dict, list, set, defaultdict)) else default_value
    # Ensure available_checklists is loaded if empty or errored
    if not st.session_state.available_checklists or "ERROR" in st.session_state.available_checklists:
        st.session_state.available_checklists = load_checklists()

# --- Call Initialization ---
initialize_session_state()


# --- 5. Load History Data (if triggered) ---
if st.session_state.load_history_id:
    history_id = st.session_state.pop('load_history_id') # Get ID and clear trigger
    st.info(f"📜 Loading historical analysis: {history_id}...")
    try:
        doc_ref = db.collection("analysis_runs").document(history_id)
        doc_snapshot = doc_ref.get()

        if doc_snapshot.exists:
            run_data = doc_snapshot.to_dict()
            gcs_pdf_path = run_data.get("gcs_pdf_path")
            results = run_data.get("results")
            filename = run_data.get("filename", "N/A")
            timestamp_obj = run_data.get("analysis_timestamp") # Firestore timestamp object or None
            # Adapt reading summary: it might be a list of dicts (old format) or a single dict (new format)
            run_summary_data = run_data.get("run_status", {}) # Default to empty dict
            checklist_name_hist = run_data.get("checklist_name", "Unknown") # Load checklist name

            if not gcs_pdf_path:
                 st.error("❌ History record is missing the PDF file path (gcs_pdf_path). Cannot load.")
            elif not results:
                 st.error("❌ History record is missing analysis results. Cannot load.")
            else:
                # Determine bucket and blob name from GCS path
                hist_bucket_name = GCS_BUCKET_NAME # Default assumption
                hist_blob_name = None
                if gcs_pdf_path.startswith("gs://"):
                    try:
                        path_parts = gcs_pdf_path[5:].split("/", 1)
                        hist_bucket_name = path_parts[0]
                        hist_blob_name = path_parts[1]
                    except IndexError:
                         st.error(f"❌ Invalid GCS path format in history record: {gcs_pdf_path}")
                         results = None # Prevent further processing
                elif "/" in gcs_pdf_path: # Assume path like FOLDER/FILENAME.PDF relative to default bucket
                     hist_blob_name = gcs_pdf_path
                else:
                    st.error(f"❌ Could not determine GCS bucket/blob from path: {gcs_pdf_path}")
                    results = None # Prevent further processing

                if results and hist_blob_name:
                    with st.spinner(f"Downloading PDF from gs://{hist_bucket_name}/{hist_blob_name}..."):
                        pdf_bytes_from_hist = download_from_gcs(hist_bucket_name, hist_blob_name)

                    if pdf_bytes_from_hist:
                        # Reset state before loading historical data BUT preserve some settings
                        reset_app_state()

                        # Load historical data into session state
                        st.session_state.pdf_bytes = pdf_bytes_from_hist
                        st.session_state.pdf_bytes_hash = base64.b64encode(pdf_bytes_from_hist).decode() # Hash for caching
                        st.session_state.analysis_results = results
                        # Store the loaded summary data (might be old list format or new overall dict)
                        st.session_state.run_status_summary = run_summary_data
                        st.session_state.analysis_complete = True # Mark as complete (for display purposes)
                        st.session_state.pdf_display_ready = True
                        st.session_state.viewing_history = True # Set history mode flag
                        st.session_state.history_filename = filename
                        # Format timestamp safely
                        try:
                            # Firestore Timestamps might need conversion to datetime
                            if isinstance(timestamp_obj, google.cloud.firestore.SERVER_TIMESTAMP.__class__):
                                # Server timestamps aren't readable directly client-side after fetch,
                                # rely on read_time which might be close enough or use a saved string.
                                hist_ts_str = "Timestamp Unavailable"
                            elif hasattr(timestamp_obj, 'strftime'): # Check if it behaves like datetime
                                 hist_ts_str = timestamp_obj.strftime('%Y-%m-%d %H:%M:%S UTC')
                            elif timestamp_obj:
                                hist_ts_str = str(timestamp_obj) # Fallback to string representation
                            else:
                                hist_ts_str = "N/A"
                        except Exception as ts_format_err:
                             print(f"Warning: Could not format history timestamp: {ts_format_err}")
                             hist_ts_str = "Invalid Timestamp"
                        st.session_state.history_timestamp = hist_ts_str

                        st.session_state.current_page = 1 # Reset to first page
                        st.session_state.current_filename = filename # Store filename for potential export
                        st.session_state.history_checklist_name = checklist_name_hist # Store the checklist name

                        st.success(f"✅ Successfully loaded history for '{filename}' (Checklist: {checklist_name_hist}, Generated: {st.session_state.history_timestamp}).")
                        time.sleep(1) # Give user time to see message
                        st.rerun() # Rerun to update UI immediately
                    else:
                        # Download failed, error shown by download_from_gcs
                        st.session_state.viewing_history = False # Ensure we are not stuck in history mode
        else:
            st.error(f"❌ History record with ID '{history_id}' not found in database.")
            st.session_state.viewing_history = False

    except Exception as e:
        st.error(f"❌ Error loading historical data: {e}")
        print(f"History Load Error: {e}\n{traceback.format_exc()}")
        st.session_state.viewing_history = False # Ensure we are not stuck in history mode


# --- 6. Streamlit UI Logic ---

# --- Header ---
header_cols = st.columns([1, 5]) # Column for logo, Column for title
with header_cols[0]:
    if os.path.exists(LOGO_PATH):
        try:
            logo = Image.open(LOGO_PATH)
            st.image(logo, width=80, caption=None, output_format='PNG', alt="Logo") # Adjust width as needed
        except Exception as img_err:
            st.warning(f"Could not load logo: {img_err}")
    else:
        st.warning(f"Logo file '{LOGO_FILE}' not found.")

with header_cols[1]:
    if st.session_state.viewing_history:
        st.title("📜 JASPER - Historical Analysis Review")
        hist_ts_str = st.session_state.history_timestamp or "N/A"
        hist_checklist = st.session_state.get('history_checklist_name', 'Unknown')
        st.caption(f"File: **{st.session_state.history_filename}** | Checklist: **{hist_checklist}** | Generated: {hist_ts_str}")
    else:
        st.title("JASPER - Extraction & Review")
        st.caption("Just A Smart Platform for Extraction and Review")


# --- Sidebar Setup ---
st.sidebar.title("Controls")
st.sidebar.markdown("---")

# --- History Mode Sidebar ---
if st.session_state.viewing_history:
    st.sidebar.info(f"📜 **Viewing Historical Analysis**")
    st.sidebar.markdown(f"**File:** {st.session_state.history_filename}")
    st.sidebar.markdown(f"**Checklist:** {st.session_state.history_checklist_name}")
    st.sidebar.markdown(f"**Generated:** {st.session_state.history_timestamp}")
    if st.sidebar.button("⬅️ Exit History / Start New", key="clear_history_view", use_container_width=True):
        reset_app_state()
        st.rerun()
    st.sidebar.markdown("---")
    # Add Excel download button here too for convenience
    st.sidebar.markdown("## Export Results")
    if st.session_state.analysis_results: # Check if results are loaded
        if st.sidebar.button("Prepare Excel Download", key="prep_excel_hist", use_container_width=True):
            st.session_state.excel_data = None; excel_prep_status = st.sidebar.empty(); excel_prep_status.info("Preparing Excel data...")
            try:
                excel_rows = [];
                # Ensure results are sorted by question number for export
                results_list = sorted(st.session_state.analysis_results, key=lambda x: x.get('Question Number', float('inf'))) if isinstance(st.session_state.analysis_results, list) else []
                for item in results_list:
                    # Ensure item is a dict before processing
                    if not isinstance(item, dict): continue

                    references = []; first_search_text = "N/A"; evidence = item.get("Evidence")
                    file_name_for_excel = st.session_state.history_filename # Always use history filename
                    gen_time_for_excel = st.session_state.history_timestamp # Always use history timestamp
                    checklist_name_for_excel = st.session_state.history_checklist_name # Always use history checklist name

                    if evidence and isinstance(evidence, list):
                        for i, ev in enumerate(evidence):
                            if isinstance(ev, dict): references.append(str(ev.get("Clause Reference", "N/A")));
                            if i == 0 and isinstance(ev, dict): first_search_text = ev.get("Searchable Clause Text", "N/A")

                    excel_row = {
                        "File Name": file_name_for_excel, "Generation Time": gen_time_for_excel,
                        "Checklist Name": checklist_name_for_excel,
                        "Question Number": item.get("Question Number"), "Question Category": item.get("Question Category", "Uncategorized"),
                        "Question": item.get("Question", "N/A"), "Answer": item.get("Answer", "N/A"),
                        "Answer Justification": item.get("Answer Justification", "N/A"),
                        "Clause References (Concatenated)": "; ".join(references) if references else "N/A",
                        "First Searchable Clause Text": first_search_text
                    }
                    excel_rows.append(excel_row)

                if not excel_rows: excel_prep_status.warning("No data available to export."); st.session_state.excel_data = None
                else:
                    df_excel = pd.DataFrame(excel_rows); final_columns = [col for col in EXCEL_COLUMN_ORDER if col in df_excel.columns]; extra_cols = [col for col in df_excel.columns if col not in final_columns]; df_excel = df_excel[final_columns + extra_cols]
                    output = io.BytesIO();
                    with pd.ExcelWriter(output, engine='openpyxl') as writer: df_excel.to_excel(writer, index=False, sheet_name='Analysis Results')
                    st.session_state.excel_data = output.getvalue(); excel_prep_status.success("✅ Excel file ready!"); time.sleep(2); excel_prep_status.empty()
            except Exception as excel_err: excel_prep_status.error(f"Excel Prep Error: {excel_err}"); print(traceback.format_exc())

        if st.session_state.excel_data:
            current_filename_for_download = st.session_state.history_filename or "history_analysis"
            safe_base_name = re.sub(r'[^\w\s-]', '', os.path.splitext(current_filename_for_download)[0]).strip().replace(' ', '_'); download_filename = f"Analysis_{safe_base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            st.sidebar.download_button(label="📥 Download Results as Excel", data=st.session_state.excel_data, file_name=download_filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_excel_final_hist", use_container_width=True)

# --- Standard Mode Sidebar ---
else: # Not viewing history
    # --- Checklist Management ---
    st.sidebar.markdown("### 1. Checklist Setup")
    checklist_status_placeholder = st.sidebar.empty()

    # Load available checklists if needed
    if not st.session_state.available_checklists or "--- Select Checklist ---" not in st.session_state.available_checklists:
         st.session_state.available_checklists = load_checklists()
         st.rerun() # Rerun if lists were loaded/reloaded

    # Determine current selection index for dropdown
    current_options = list(st.session_state.available_checklists.keys()) + ["--- Create New Checklist ---"]
    try:
        # Find the name corresponding to the selected ID
        current_selection_name = next((name for name, id_val in st.session_state.available_checklists.items() if id_val == st.session_state.selected_checklist_id), None)
        if current_selection_name and current_selection_name in current_options:
            current_index = current_options.index(current_selection_name)
        elif st.session_state.checklist_action == 'create':
             current_index = current_options.index("--- Create New Checklist ---")
        else:
             current_index = 0 # Default to "--- Select Checklist ---"
    except ValueError:
         current_index = 0 # Fallback

    # Checklist Selection Dropdown
    selected_checklist_name = st.sidebar.selectbox(
        "Select or Create Checklist",
        options=current_options,
        key='checklist_selector',
        index=current_index,
        help="Choose a saved checklist or start creating a new one."
    )

    # --- Handle selection change ---
    create_new_selected = (selected_checklist_name == "--- Create New Checklist ---")
    valid_existing_selected = (selected_checklist_name != "--- Select Checklist ---" and not create_new_selected)
    placeholder_selected = (selected_checklist_name == "--- Select Checklist ---")

    # User selected "Create New"
    if create_new_selected and st.session_state.checklist_action != 'create':
        st.session_state.checklist_action = 'create'
        st.session_state.selected_checklist_id = None
        st.session_state.selected_checklist_name = None
        st.session_state.new_checklist_name = ""
        st.session_state.current_checklist_prompt = f"Analyse the document based on the following questions:\n\n{PROMPT_PLACEHOLDER}\n\nProvide answers in the required JSON format." # Default prompt
        st.session_state.current_checklist_questions = []
        st.session_state.uploaded_checklist_file_obj = None # Reset uploaded file
        st.rerun() # Rerun to update UI for creation mode

    # User selected an existing checklist
    elif valid_existing_selected:
        newly_selected_id = st.session_state.available_checklists.get(selected_checklist_name)
        if newly_selected_id != st.session_state.selected_checklist_id:
            # Clear potentially cached details for the *newly* selected ID before loading
            load_checklist_details.clear()
            checklist_data = load_checklist_details(newly_selected_id)
            if checklist_data:
                st.session_state.selected_checklist_id = newly_selected_id
                st.session_state.selected_checklist_name = checklist_data.get("name", selected_checklist_name)
                st.session_state.current_checklist_prompt = checklist_data.get("prompt_template", "")
                st.session_state.current_checklist_questions = checklist_data.get("questions", [])
                st.session_state.checklist_action = 'selected' # Indicate an existing one is selected
                st.session_state.new_checklist_name = "" # Clear create form field
                st.session_state.uploaded_checklist_file_obj = None # Clear upload field
                checklist_status_placeholder.success(f"Loaded: '{st.session_state.selected_checklist_name}'")
                time.sleep(1); checklist_status_placeholder.empty()
                st.rerun() # Update UI
            else:
                checklist_status_placeholder.error(f"Failed to load details for '{selected_checklist_name}'.")
                st.session_state.selected_checklist_id = None # Reset selection
                st.session_state.checklist_action = None
                st.session_state.current_checklist_prompt = ""
                st.session_state.current_checklist_questions = []
                st.rerun() # Rerun to reset state

    # User selected the placeholder "--- Select Checklist ---"
    elif placeholder_selected and st.session_state.selected_checklist_id is not None:
         st.session_state.selected_checklist_id = None
         st.session_state.selected_checklist_name = None
         st.session_state.current_checklist_prompt = ""
         st.session_state.current_checklist_questions = []
         st.session_state.checklist_action = None
         st.rerun()

    # --- Checklist Creation/Editing Form (inside expander) ---
    form_expander_label = ""
    if st.session_state.checklist_action == 'create':
        form_expander_label = "➕ Create New Checklist"
    elif st.session_state.checklist_action == 'selected':
        form_expander_label = f"✏️ Edit Checklist: '{st.session_state.selected_checklist_name}'"

    if form_expander_label:
        with st.sidebar.expander(form_expander_label, expanded=True): # Expand by default when active
            if st.session_state.checklist_action == 'create':
                # Checklist Name Input
                st.session_state.new_checklist_name = st.text_input(
                    "New Checklist Name*",
                    value=st.session_state.new_checklist_name,
                    key="new_checklist_name_input",
                    placeholder="Enter a unique name"
                )

                # File Uploader for Questions
                uploaded_excel = st.file_uploader(
                    "Upload Questions (.xlsx)*",
                    type="xlsx",
                    key="checklist_uploader",
                    help=f"Excel file must have columns: {', '.join(TEMPLATE_COLUMN_ORDER)}"
                )

                # Display Template Download Button Here
                excel_template_bytes = generate_excel_template()
                st.download_button(
                    label="📄 Download Template (.xlsx)",
                    data=excel_template_bytes,
                    file_name="checklist_template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_template_inline",
                    use_container_width=True,
                    help="Download an Excel template with the required columns for uploading questions."
                )

                # Parse uploaded file immediately if present and different
                if uploaded_excel is not None and uploaded_excel != st.session_state.uploaded_checklist_file_obj:
                    st.session_state.uploaded_checklist_file_obj = uploaded_excel # Store file object
                    parsed_questions, error = parse_checklist_excel(uploaded_excel)
                    if error:
                        checklist_status_placeholder.error(error)
                        st.session_state.current_checklist_questions = []
                    else:
                        checklist_status_placeholder.success(f"Parsed {len(parsed_questions)} questions from '{uploaded_excel.name}'.")
                        st.session_state.current_checklist_questions = parsed_questions
                        time.sleep(2); checklist_status_placeholder.empty()
                        # Rerun needed to update the question preview in main area
                        st.rerun()

                # Prompt Template Input
                st.session_state.current_checklist_prompt = st.text_area(
                    "Checklist Prompt Template*",
                    value=st.session_state.current_checklist_prompt,
                    key="new_checklist_prompt_input",
                    height=150,
                    help=f"Define prompt text. MUST include '{PROMPT_PLACEHOLDER}' where questions will be inserted."
                )

                # Save Button
                can_save_new = bool(st.session_state.new_checklist_name and
                                    st.session_state.current_checklist_prompt and
                                    PROMPT_PLACEHOLDER in st.session_state.current_checklist_prompt and
                                    st.session_state.current_checklist_questions) # Check if questions were parsed

                if st.button("💾 Save New Checklist", key="save_new_checklist", use_container_width=True, type="primary", disabled=not can_save_new):
                    saved_id, success = save_checklist_to_firestore(
                        st.session_state.new_checklist_name,
                        st.session_state.current_checklist_prompt,
                        st.session_state.current_checklist_questions
                    )
                    if success:
                        checklist_status_placeholder.success(f"Saved '{st.session_state.new_checklist_name}'. Reloading...")
                        st.session_state.available_checklists = {} # Force reload
                        st.session_state.selected_checklist_id = saved_id # Auto-select the new one
                        st.session_state.checklist_action = 'selected' # Switch mode
                        time.sleep(1)
                        st.rerun()

            elif st.session_state.checklist_action == 'selected':
                # Edit Name (Added capability)
                 st.session_state.selected_checklist_name = st.text_input(
                     "Checklist Name*",
                     value=st.session_state.selected_checklist_name,
                     key="edit_checklist_name_input",
                     help="You can rename the checklist here."
                 )

                 # Edit Prompt
                 st.session_state.current_checklist_prompt = st.text_area(
                    "Checklist Prompt Template*",
                    value=st.session_state.current_checklist_prompt,
                    key="edit_checklist_prompt_input",
                    height=150,
                    help=f"Edit the prompt template. MUST include '{PROMPT_PLACEHOLDER}'. Save changes below."
                 )
                 st.caption(f"This checklist currently contains {len(st.session_state.current_checklist_questions)} questions (defined at creation/last update). To change questions, create a new checklist.")

                 # Update Button
                 can_update = bool(st.session_state.selected_checklist_id and
                                   st.session_state.selected_checklist_name and
                                   st.session_state.current_checklist_prompt and
                                   PROMPT_PLACEHOLDER in st.session_state.current_checklist_prompt)

                 if st.button("💾 Save Changes", key="update_checklist", use_container_width=True, type="primary", disabled=not can_update):
                    success = update_checklist_in_firestore(
                        st.session_state.selected_checklist_id,
                        st.session_state.selected_checklist_name, # Use potentially edited name
                        st.session_state.current_checklist_prompt,
                        st.session_state.current_checklist_questions # Keep questions same on prompt/name update
                    )
                    if success:
                         checklist_status_placeholder.success("Checklist updated.")
                         st.session_state.available_checklists = {} # Force reload in case name changed
                         time.sleep(1)
                         st.rerun()

    st.sidebar.markdown("---") # Separator

    # --- Analysis Setup ---
    st.sidebar.markdown("### 2. Analysis Setup")

    # --- API Key Input ---
    api_key_input = st.sidebar.text_input("Google AI Gemini API Key*", type="password", key="api_key_input", help="Your API key is used only for this session and is not stored.", value=st.session_state.get("api_key", ""))
    if api_key_input and api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input

    if not st.session_state.api_key and not st.session_state.processing_in_progress :
        st.sidebar.warning("API Key required to run analysis.", icon="🔑")

    # --- File Upload ---
    uploaded_file_obj = st.sidebar.file_uploader("Upload Facility Agreement PDF*", type="pdf", key=f"pdf_uploader_{st.session_state.get('run_key', 0)}")
    if uploaded_file_obj is not None:
        new_file_bytes = uploaded_file_obj.getvalue()
        new_hash = base64.b64encode(new_file_bytes).decode()
        if new_hash != st.session_state.get('pdf_bytes_hash'):
            # Reset analysis-related state but preserve API key/AI params/Checklist Selection
            current_checklist_id = st.session_state.selected_checklist_id
            current_checklist_name = st.session_state.selected_checklist_name
            current_prompt = st.session_state.current_checklist_prompt
            current_questions = st.session_state.current_checklist_questions
            current_action = st.session_state.checklist_action

            # Partially reset state relevant to analysis run
            keys_to_clear_on_new_pdf = [
                'analysis_results', 'analysis_complete', 'processing_in_progress', 'run_status_summary',
                'excel_data', 'search_trigger', 'last_search_result', 'show_wording_states', 'current_page'
            ]
            for key in keys_to_clear_on_new_pdf:
                if key in st.session_state: del st.session_state[key]
            initialize_session_state() # Re-init defaults for cleared keys

            # Restore checklist state
            st.session_state.selected_checklist_id = current_checklist_id
            st.session_state.selected_checklist_name = current_checklist_name
            st.session_state.current_checklist_prompt = current_prompt
            st.session_state.current_checklist_questions = current_questions
            st.session_state.checklist_action = current_action

            # Load new PDF
            st.session_state.pdf_bytes = new_file_bytes
            st.session_state.pdf_bytes_hash = new_hash
            st.session_state.pdf_display_ready = True
            st.session_state.current_filename = uploaded_file_obj.name
            st.session_state.current_page = 1 # Reset to page 1
            st.session_state.last_search_result = None # Clear search highlight

            st.toast(f"✅ PDF '{uploaded_file_obj.name}' loaded. Ready for analysis.", icon="📄")
            st.rerun()

    elif not st.session_state.pdf_bytes:
         st.sidebar.info("Upload a PDF to enable analysis.")

    st.sidebar.markdown("### 3. Analysis Options")
    # --- AI Parameter Controls ---
    with st.sidebar.expander("Advanced AI Settings"):
        st.session_state.ai_temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, step=0.05, value=st.session_state.ai_temperature, help="Controls randomness (0=deterministic, 1=max creative). Default: 0.2")
        st.session_state.ai_top_p = st.slider("Top-P", min_value=0.0, max_value=1.0, step=0.05, value=st.session_state.ai_top_p, help="Nucleus sampling (consider tokens summing to this probability). Default: 0.95")
        st.session_state.ai_top_k = st.number_input("Top-K", min_value=1, step=1, value=st.session_state.ai_top_k, help="Consider top K most likely tokens. Default: 40")

    st.sidebar.markdown("### 4. Run Analysis")
    # --- Analysis Trigger ---
    can_analyse = (
        st.session_state.pdf_bytes is not None and
        st.session_state.api_key is not None and
        st.session_state.selected_checklist_id is not None and # Must have a checklist selected
        st.session_state.current_checklist_prompt and
        st.session_state.current_checklist_questions and # Questions must exist
        PROMPT_PLACEHOLDER in st.session_state.current_checklist_prompt and # Prompt must be valid
        not st.session_state.processing_in_progress and
        not st.session_state.analysis_complete # Prevent re-running on same file unless reset
    )

    tooltip_parts = []
    if st.session_state.analysis_complete: tooltip_parts.append("Analysis already completed for this PDF. Upload a new PDF or exit history view.")
    if st.session_state.processing_in_progress: tooltip_parts.append("Analysis is currently running.")
    if not st.session_state.pdf_bytes: tooltip_parts.append("Upload a PDF file.")
    if not st.session_state.api_key: tooltip_parts.append("Enter your Gemini API key.")
    if not st.session_state.selected_checklist_id: tooltip_parts.append("Select a checklist.")
    if st.session_state.selected_checklist_id and not st.session_state.current_checklist_questions: tooltip_parts.append("Selected checklist has no questions. Edit or choose another.")
    if st.session_state.selected_checklist_id and not st.session_state.current_checklist_prompt: tooltip_parts.append("Selected checklist has no prompt template. Edit or choose another.")
    if st.session_state.selected_checklist_id and st.session_state.current_checklist_prompt and PROMPT_PLACEHOLDER not in st.session_state.current_checklist_prompt: tooltip_parts.append(f"Prompt template missing '{PROMPT_PLACEHOLDER}'. Edit the checklist.")

    analyse_button_tooltip = " ".join(tooltip_parts) if tooltip_parts else "Start analyzing the document using the selected checklist"

    if st.sidebar.button("✨ Analyse Document", key="analyse_button", disabled=not can_analyse, help=analyse_button_tooltip, use_container_width=True, type="primary"):
        # --- Start Analysis Process ---
        st.session_state.processing_in_progress = True
        st.session_state.analysis_complete = False
        st.session_state.analysis_results = [] # Clear previous results
        st.session_state.run_key += 1 # Increment run key for this analysis run
        st.session_state.run_status_summary = {} # Store overall summary here
        st.session_state.excel_data = None
        st.session_state.search_trigger = None
        st.session_state.last_search_result = None
        st.session_state.show_wording_states = defaultdict(bool) # Reset toggles

        current_api_key = st.session_state.api_key
        current_gen_config_params = {
            'temperature': st.session_state.ai_temperature,
            'top_p': st.session_state.ai_top_p,
            'top_k': st.session_state.ai_top_k
        }
        checklist_prompt = st.session_state.current_checklist_prompt
        # Pass a copy to avoid potential modification by assign_question_numbers inside analysis function
        checklist_questions = copy.deepcopy(st.session_state.current_checklist_questions)
        checklist_id = st.session_state.selected_checklist_id
        checklist_name = st.session_state.selected_checklist_name

        run_start_time = datetime.now()
        # Use timezone aware timestamp for run time string as well
        run_timestamp_str = run_start_time.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


        if not st.session_state.current_filename:
             st.session_state.current_filename = f"analysis_{run_start_time.strftime('%Y%m%d%H%M%S')}.pdf"
             st.warning("Could not determine original filename, using generated name.")
        base_file_name = st.session_state.current_filename

        try:
            genai.configure(api_key=current_api_key)
            st.toast("API Key validated and configured.", icon="🔑")
        except Exception as config_err:
            st.error(f"❌ Failed to configure Gemini API with provided key: {config_err}")
            st.session_state.processing_in_progress = False; st.stop()

        # Use main area for status during analysis
        status_container = st.container()
        progress_bar = status_container.progress(0, text="Initializing analysis...")
        status_text = status_container.empty()
        status_text.info("Preparing analysis...")

        temp_dir = "temp_uploads"
        safe_base_name = re.sub(r'[^\w\-.]', '_', base_file_name)
        temp_file_path = os.path.join(APP_DIR, temp_dir, f"{run_start_time.strftime('%Y%m%d%H%M%S')}_{st.session_state.run_key}_{safe_base_name}")
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        gemini_uploaded_file_ref = None
        all_validated_data = None
        overall_status = "Failed"
        gcs_file_path = None
        run_warnings = []

        try:
            # --- File Upload to Gemini ---
            status_text.info("💾 Saving file temporarily..."); progress_bar.progress(5, text="Saving temp file...");
            with open(temp_file_path, "wb") as f: f.write(st.session_state.pdf_bytes)
            status_text.info("☁️ Uploading file to Google Cloud AI..."); progress_bar.progress(10, text="Uploading to cloud...")
            for upload_attempt in range(3):
                try:
                    display_name = f"JASPER_{safe_base_name}_{run_start_time.strftime('%Y%m%d%H%M%S')}"
                    gemini_uploaded_file_ref = genai.upload_file(path=temp_file_path, display_name=display_name)
                    st.toast(f"File '{gemini_uploaded_file_ref.display_name}' uploaded to cloud.", icon="☁️"); break
                except Exception as upload_err:
                    err_str = str(upload_err).lower()
                    if "api key" in err_str or "authenticat" in err_str or "permission" in err_str:
                        status_text.error(f"❌ File upload failed due to API key/permission issue: {upload_err}")
                        st.error("Please verify the API key has File API permissions enabled.")
                        run_warnings.append(f"Upload Error (Permissions): {upload_err}")
                        raise ValueError(f"Upload Error (Permissions): {upload_err}") # Non-retryable
                    elif upload_attempt < 2:
                        status_text.warning(f"Upload attempt {upload_attempt+1} failed: {upload_err}. Retrying...")
                        run_warnings.append(f"Upload Warning (Attempt {upload_attempt+1}): {upload_err}")
                        time.sleep(2 + upload_attempt)
                    else:
                        status_text.error(f"Upload failed after multiple attempts: {upload_err}")
                        run_warnings.append(f"Upload Error (Final): {upload_err}")
                        raise # Re-raise the final error
            if not gemini_uploaded_file_ref: raise Exception("Failed to upload file to Google Cloud AI after retries.")
            # Progress now handled inside generate_checklist_analysis_per_category

            # --- Run Analysis PER CATEGORY ---
            analysis_data, analysis_status, analysis_warnings = generate_checklist_analysis_per_category(
                checklist_prompt, checklist_questions, gemini_uploaded_file_ref, status_text, current_api_key, current_gen_config_params, progress_bar
            )

            overall_status = analysis_status
            run_warnings.extend(analysis_warnings) # Add warnings from the per-category processing
            # Store overall summary
            st.session_state.run_status_summary = {
                "overall_status": overall_status,
                "checklist": checklist_name,
                "timestamp": run_timestamp_str,
                "warnings": run_warnings # Store aggregated warnings
            }

            # --- Finalize Analysis ---
            if analysis_status in ["Success", "Partial Success", "Success (Empty)"] and analysis_data is not None:
                all_validated_data = analysis_data
                # Add common fields AFTER all results are aggregated
                for item in all_validated_data:
                     # Ensure item is a dict before adding fields
                     if isinstance(item, dict):
                        item["File Name"] = base_file_name
                        item["Generation Time"] = run_timestamp_str
                        item["Checklist Name"] = checklist_name

                st.session_state.analysis_results = all_validated_data
                progress_bar.progress(90, text="Analysis processed. Saving records...")
                # Status message already displayed by generate_checklist_analysis_per_category
                st.session_state.analysis_complete = True # Mark complete even if partial or empty

                # --- Save to GCS and Firestore ---
                if st.session_state.pdf_bytes:
                    try:
                        # Use timezone-aware UTC timestamp for saving
                        timestamp = datetime.now(timezone.utc) # CORRECTED LINE
                        firestore_doc_id = f"{uuid.uuid4()}"
                        # Include date in GCS path for better organization
                        gcs_blob_name = f"{GCS_PDF_FOLDER}/{timestamp.strftime('%Y%m%d')}/{firestore_doc_id}_{safe_base_name}"

                        gcs_file_path = upload_to_gcs(GCS_BUCKET_NAME, st.session_state.pdf_bytes, gcs_blob_name, status_text)

                        status_text.info("💾 Saving results and PDF reference to database...")
                        doc_ref = db.collection("analysis_runs").document(firestore_doc_id)
                        doc_ref.set({
                            "filename": base_file_name,
                            "analysis_timestamp": timestamp, # Store UTC timestamp object
                            "results": st.session_state.analysis_results,
                            "run_status": st.session_state.run_status_summary, # Save the overall summary dict
                            "gcs_pdf_path": gcs_file_path,
                            "checklist_name": checklist_name,
                            "checklist_id": checklist_id
                        })
                        status_text.success("💾 Results and PDF link saved successfully to database.")
                        progress_bar.progress(100, text="Analysis saved!")
                        time.sleep(1)
                    except Exception as db_gcs_err:
                        st.error(f"❌ Failed to save results/PDF to cloud: {db_gcs_err}")
                        print(f"DB/GCS Save Error: {db_gcs_err}\n{traceback.format_exc()}")
                        # Update summary to reflect save error
                        st.session_state.run_status_summary['overall_status'] = "Failed (Save Error)"
                        st.session_state.run_status_summary['warnings'].append(f"Error saving to GCS/Firestore: {db_gcs_err}")
                        overall_status = "Failed (Save Error)"
                        st.session_state.analysis_complete = False
            else:
                 # Status message already displayed by generate_checklist_analysis_per_category
                 st.session_state.analysis_results = [] # Ensure it's an empty list
                 st.session_state.analysis_complete = False

        except ValueError as ve: # Catch non-retryable errors like permission issues during setup
             st.error(f"❌ ANALYSIS HALTED: {ve}")
             overall_status = "Failed (Setup Error)"; st.session_state.analysis_complete = False
             st.session_state.run_status_summary = {"overall_status": overall_status, "checklist": checklist_name, "timestamp": run_timestamp_str, "warnings": [str(ve)]}
             status_text.error(f"Analysis stopped due to setup error: {ve}")
        except Exception as main_err:
            st.error(f"❌ CRITICAL ERROR during analysis workflow: {main_err}"); print(traceback.format_exc())
            overall_status = "Failed (Critical Error)"; st.session_state.analysis_complete = False
            st.session_state.run_status_summary = {"overall_status": overall_status, "checklist": checklist_name, "timestamp": run_timestamp_str, "warnings": [str(main_err), "Analysis halted. See logs."]}
            status_text.error(f"Analysis stopped due to critical error: {main_err}")
        finally:
            # --- Cleanup ---
            st.session_state.processing_in_progress = False
            time.sleep(4); status_text.empty(); progress_bar.empty()

            # Delete Gemini Cloud File
            if gemini_uploaded_file_ref and hasattr(gemini_uploaded_file_ref, 'name'):
                try:
                    status_text.info(f"☁️ Deleting temporary Gemini cloud file: {gemini_uploaded_file_ref.name}...");
                    genai.delete_file(name=gemini_uploaded_file_ref.name)
                    st.toast("Gemini cloud file deleted.", icon="🗑️")
                    time.sleep(1); status_text.empty()
                except Exception as del_err:
                     st.sidebar.warning(f"Gemini cloud cleanup issue: {del_err}", icon="⚠️")
                     status_text.warning(f"Could not delete Gemini cloud file: {del_err}")
                     print(f"WARN: Failed to delete cloud file {gemini_uploaded_file_ref.name}: {del_err}")
                     time.sleep(2); status_text.empty()

            # Delete Local Temp File
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except Exception as local_del_err:
                     st.sidebar.warning(f"Local temp file cleanup issue: {local_del_err}", icon="⚠️")
                     print(f"WARN: Failed to delete local temp file {temp_file_path}: {local_del_err}")

        st.rerun() # Rerun to reflect final state


# --- 7. Display Area (Checklist Preview, Results, PDF Viewer) ---

# --- Checklist Question Preview (Only when not viewing history and questions are loaded) ---
if not st.session_state.viewing_history and st.session_state.current_checklist_questions:
    questions_to_display = st.session_state.current_checklist_questions
    checklist_source_name = st.session_state.selected_checklist_name or "New Checklist (from upload)"
    with st.expander(f"📋 Previewing Questions for Checklist: '{checklist_source_name}' ({len(questions_to_display)} questions)", expanded=False):
        # Group questions by category for display
        grouped_preview = defaultdict(list)
        # Assign temporary numbers for preview purposes if not already done
        temp_numbered_questions = assign_question_numbers(copy.deepcopy(questions_to_display)) # Use deepcopy to avoid modifying state
        for q_dict in temp_numbered_questions:
             # Ensure dict before access
            if isinstance(q_dict, dict):
                grouped_preview[q_dict.get("Question Category", "Uncategorized")].append(q_dict)

        if not grouped_preview:
            st.caption("No questions loaded.")
        else:
            sorted_categories = sorted(grouped_preview.keys())
            for category in sorted_categories:
                st.markdown(f"**{category}**")
                # Sort questions within category by their assigned number for consistent preview
                questions_in_category = sorted(grouped_preview[category], key=lambda q: q.get('assigned_number', float('inf')))
                for q_data in questions_in_category:
                    # Ensure q_data is a dict
                    if isinstance(q_data, dict):
                        q_num = q_data.get('assigned_number', '?')
                        q_text = q_data.get('Question', 'N/A')
                        q_opts = q_data.get('Answer Options', '')
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{q_num}. {q_text}")
                        if q_opts:
                            st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Options/Guidance: {q_opts}*")
                    else:
                         st.warning(f"Skipping invalid question data in preview: {type(q_data)}")
                st.markdown("---") # Separator between categories

# --- Main Content Area (Results & PDF Viewer) ---
if st.session_state.pdf_bytes is not None:
    col1, col2 = st.columns([6, 4]) # Results | PDF Viewer

    # --- Column 1: Analysis Results ---
    with col1:
        # --- Run Status Summary ---
        if st.session_state.run_status_summary:
            st.markdown("#### Analysis Run Summary")
            summary_data = st.session_state.run_status_summary
            # Handle both potential formats (old list vs new dict) for backward compatibility
            overall_status = "Unknown"; checklist_name_sum = "N/A"; warnings_sum = []

            if isinstance(summary_data, dict): # New format (preferred)
                overall_status = summary_data.get("overall_status", "Unknown")
                checklist_name_sum = summary_data.get("checklist", "N/A")
                warnings_sum = summary_data.get("warnings", [])
                # Optional: Add timestamp display from summary if available
                run_ts_from_summary = summary_data.get("timestamp", None)

            elif isinstance(summary_data, list) and summary_data: # Old format (list of dicts)
                # Try to infer overall status from the list
                # Ensure first item is dict before access
                first_item = summary_data[0] if isinstance(summary_data[0], dict) else {}
                checklist_name_sum = first_item.get("checklist", "N/A") # Get name from first entry

                all_statuses = [item.get("status", "Unknown") for item in summary_data if isinstance(item, dict)]
                if not all_statuses: # If list contains non-dicts
                     overall_status = "Invalid Summary Format"
                elif "Failed" in all_statuses: overall_status = "Failed"
                elif "Partial Success" in all_statuses: overall_status = "Partial Success"
                elif all(s == "Success (Empty)" for s in all_statuses): overall_status = "Success (Empty)"
                elif all(s == "Success" for s in all_statuses): overall_status = "Success"
                else: overall_status = "Mixed/Unknown"
                # Aggregate warnings from all entries in the old list format
                for item in summary_data:
                    if isinstance(item, dict):
                         warnings_sum.extend(item.get("warnings", []))
                run_ts_from_summary = None # Timestamp not usually in old list format
            else: # No summary data or unrecognized format
                 overall_status = "Not Available"
                 run_ts_from_summary = None

            status_icon_map = {
                "Success": "✅", "Partial Success": "⚠️", "Failed": "❌",
                "Success (Empty)": "ℹ️", "Skipped": "➡️", "Mixed/Unknown": "❓",
                "Failed (Save Error)": "❌💾", "Failed (Setup Error)": "❌⚙️",
                "Failed (Critical Error)": "❌🔥"
            }
            final_status_icon = status_icon_map.get(overall_status, "❓")
            summary_title = f"{final_status_icon} Overall Status for '{checklist_name_sum}': **{overall_status}**"

            # Add timestamp to title if available and not viewing history (history already has it)
            if run_ts_from_summary and not st.session_state.viewing_history:
                 summary_title += f" (Run: {run_ts_from_summary})"

            # Expand summary if history, not success, or partial/empty/failed
            expand_summary = st.session_state.viewing_history or overall_status not in ["Success"]

            with st.expander(summary_title, expanded=expand_summary):
                # Display warnings/details
                if warnings_sum:
                    st.caption("Details / Issues Encountered:")
                    # Filter out generic validation header messages if present
                    filtered_warnings = [msg for msg in warnings_sum if not (isinstance(msg, str) and msg.startswith("Validation Issues Found"))]

                    # Provide a generic message if only validation headers were present
                    if not filtered_warnings and any(isinstance(msg, str) and msg.startswith("Validation Issues Found") for msg in warnings_sum):
                        st.warning(" L> Structure or content mismatch found in AI response(s) compared to schema/expected questions.")

                    # Display filtered warnings, applying formatting based on content
                    displayed_warnings_count = 0
                    for i, msg in enumerate(filtered_warnings):
                         msg_str = str(msg).strip()
                         if not msg_str: continue # Skip empty messages

                         # Check for keywords to determine message type
                         is_error = any(term in msg_str.lower() for term in ["critical", "error", "block", "fail", "invalid key", "permission", "stopping", "halted"]) and "warning" not in msg_str.lower()
                         is_warning = any(term in msg_str.lower() for term in ["warn", "missing", "unexpected", "empty list", "mismatch", "recitation", "max_tokens", "timeout", "partial", "cleanup issue"])

                         prefix = f" {displayed_warnings_count+1}. "
                         if is_error: st.error(f"{prefix}{msg_str}")
                         elif is_warning: st.warning(f"{prefix}{msg_str}")
                         elif "skipped" in overall_status.lower(): st.info(f"{prefix}{msg_str}")
                         else: st.caption(f"{prefix}{msg_str}") # Default to caption for info/other messages
                         displayed_warnings_count += 1

                    if displayed_warnings_count == 0 and not filtered_warnings and warnings_sum:
                         # If only validation headers existed and were filtered out, mention it here
                         st.caption("No specific operational issues reported, but potential response validation issues detected (see above).")
                    elif displayed_warnings_count == 0 and warnings_sum:
                         # Should not happen if warnings_sum was non-empty, but as a fallback
                         st.caption("Issues reported but could not be displayed.")

                else:
                    st.caption("No specific issues reported for this run.")
        # --- End Run Status Summary ---


        # --- Display Results ---
        st.markdown("#### Detailed Analysis Results")

        # Check if analysis_results is a list before proceeding
        analysis_results_list = st.session_state.analysis_results if isinstance(st.session_state.analysis_results, list) else None

        if (st.session_state.analysis_complete or st.session_state.viewing_history) and analysis_results_list is not None:
            if not analysis_results_list:
                 st.info("Analysis complete, but the AI returned no results for this checklist and document.")
            else:
                try:
                    # Sort results primarily by Question Number for consistent display
                    results_list = sorted(analysis_results_list, key=lambda x: x.get('Question Number', float('inf')) if isinstance(x, dict) else float('inf'))
                except Exception as sort_err:
                     st.warning(f"Could not sort results by question number: {sort_err}. Displaying in original order.")
                     results_list = analysis_results_list # Use original if sort fails

                # --- Scatter Plot Expander ---
                try:
                    plot_data = []
                    for item in results_list:
                        # Ensure item is a dict before accessing keys
                        if isinstance(item, dict):
                             evidence_list_plot = item.get('Evidence')
                             plot_data.append({
                                 'Question Number': item.get('Question Number', 0),
                                 'Number of Evidence Items': len(evidence_list_plot) if isinstance(evidence_list_plot, list) else 0,
                                 'Question Category': item.get('Question Category', 'Uncategorized'),
                                 'Question': item.get('Question', 'N/A')
                             })
                    if plot_data:
                        df_plot = pd.DataFrame(plot_data)
                        with st.expander("📊 Evidence Count Analysis (Scatter Plot)", expanded=False):
                            fig = px.scatter(df_plot, x='Question Number', y='Number of Evidence Items', color='Question Category',
                                             title="Number of Evidence Clauses Found per Question", labels={'Number of Evidence Items': 'Evidence Count'},
                                             hover_data=['Question'])
                            fig.update_traces(marker=dict(size=10, opacity=0.7), selector=dict(mode='markers'))
                            fig.update_layout(xaxis_title="Question Number", yaxis_title="Number of Evidence Clauses", legend_title_text='Category')
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("This plot shows how many separate evidence clauses the AI referenced for each question. Hover over points for question details.")
                    # else: st.caption("No data available for evidence plot.") # Don't show if no data
                except Exception as plot_err:
                     st.warning(f"Could not generate scatter plot: {plot_err}")
                     print(f"Plotting Error: {plot_err}\n{traceback.format_exc()}")


                # --- Tabbed Results Display ---
                grouped_results = defaultdict(list); categories_ordered = []
                for item in results_list:
                     # Ensure item is a dict
                    if isinstance(item, dict):
                        category = item.get("Question Category", "Uncategorized")
                        if category not in grouped_results: categories_ordered.append(category)
                        grouped_results[category].append(item)

                # Sort categories alphabetically for consistent tab order
                categories_ordered.sort()

                if categories_ordered:
                    # Create tab names, ensuring uniqueness if needed (though less likely now?)
                    tab_names = []
                    name_counts = defaultdict(int)
                    for cat in categories_ordered:
                        name_counts[cat] += 1
                        tab_name = f"{cat} ({name_counts[cat]})" if name_counts[cat] > 1 else cat
                        # Truncate long tab names if necessary
                        tab_name = tab_name[:30] + '...' if len(tab_name) > 30 else tab_name
                        tab_names.append(tab_name)

                    try:
                         category_tabs = st.tabs(tab_names)
                    except Exception as tab_err:
                         st.error(f"Error creating tabs: {tab_err}. Displaying as a list.")
                         category_tabs = None # Fallback

                    if category_tabs and len(category_tabs) == len(categories_ordered): # Ensure tabs were created correctly
                        for i, category in enumerate(categories_ordered):
                            with category_tabs[i]:
                                # Sort items within the tab by question number
                                category_items = sorted(grouped_results[category], key=lambda x: x.get('Question Number', float('inf')) if isinstance(x, dict) else float('inf'))
                                for index, result_item in enumerate(category_items):
                                    # Ensure result_item is a dict
                                    if not isinstance(result_item, dict):
                                        st.warning(f"Skipping invalid result item at index {index} in category '{category}'.")
                                        continue

                                    q_num = result_item.get('Question Number', 'N/A'); question_text = result_item.get('Question', 'N/A')
                                    expander_title = f"**Q{q_num}:** {question_text[:100]}{'...' if len(question_text)>100 else ''}"
                                    # --- UI STRUCTURE within Expander ---
                                    with st.expander(expander_title):
                                        st.markdown(f"**Question:** {question_text}") # Show full question inside
                                        st.markdown(f"**Answer:**")
                                        st.markdown(f"> {result_item.get('Answer', 'N/A')}") # Using blockquote
                                        st.markdown("**Answer Justification:**")
                                        justification_text = result_item.get('Answer Justification', '')
                                        just_key = f"justification_{category}_{q_num}_{index}_{st.session_state.run_key}" # Add run key

                                        if justification_text:
                                            if len(justification_text) < 200:
                                                st.markdown(f"> _{justification_text}_")
                                            else:
                                                st.text_area("Justification Text", value=justification_text, height=100, disabled=True, label_visibility="collapsed", key=just_key)
                                        else:
                                            st.markdown("> _N/A_")


                                        st.markdown("---") # Separator before Evidence

                                        evidence_list = result_item.get('Evidence', [])
                                        if isinstance(evidence_list, list) and evidence_list:
                                            st.markdown("**Evidence:**")
                                            for ev_index, evidence_item in enumerate(evidence_list):
                                                 if not isinstance(evidence_item, dict):
                                                     st.warning(f"Skipping invalid evidence item {ev_index+1} (not a dictionary).")
                                                     continue

                                                 clause_ref = evidence_item.get('Clause Reference', 'N/A')
                                                 search_text = evidence_item.get('Searchable Clause Text', None)
                                                 clause_wording = evidence_item.get('Clause Wording', 'N/A')
                                                 # Ensure base_key is unique across runs/reruns
                                                 base_key = f"ev_{category}_{q_num}_{index}_{ev_index}_{st.session_state.run_key}"

                                                 # Use columns for better layout of buttons/toggles
                                                 ev_cols = st.columns([3, 1]) # Wider column for button, narrower for toggle

                                                 with ev_cols[0]: # Find Button / Clause Ref Display
                                                     if search_text and search_text.strip():
                                                         button_key = f"search_btn_{base_key}"
                                                         button_label = f"📌 **{clause_ref or 'Evidence'}**: Find & View"
                                                         if st.button(button_label, key=button_key, help=f"Search PDF for text related to '{clause_ref or 'this evidence'}' and jump to the page.", use_container_width=True):
                                                             st.session_state.search_trigger = {'text': search_text, 'ref': clause_ref or f"Q{q_num} Ev{ev_index+1}"}
                                                             st.session_state.last_search_result = None
                                                             st.rerun()
                                                     else:
                                                         st.markdown(f"Clause: **{clause_ref}**")
                                                         st.caption("(No searchable text provided by AI)")


                                                 with ev_cols[1]: # Show Wording Toggle
                                                     if clause_wording and clause_wording != 'N/A':
                                                         toggle_key = f"toggle_wording_{base_key}"
                                                         # Use session state directly with get for default
                                                         show_wording = st.toggle("Show Wording", key=toggle_key, value=st.session_state.show_wording_states.get(toggle_key, False), help="Show/hide exact clause wording extracted by AI.")
                                                         if show_wording != st.session_state.show_wording_states.get(toggle_key, False):
                                                             st.session_state.show_wording_states[toggle_key] = show_wording
                                                             st.rerun() # Rerun needed to show/hide
                                                     else:
                                                          st.caption("(No wording provided)")

                                                 # Display wording area if toggled
                                                 if st.session_state.show_wording_states.get(f"toggle_wording_{base_key}", False):
                                                      # Use markdown with code block for better readability/copying
                                                     st.markdown(f"**AI Extracted Wording for '{clause_ref}':**")
                                                     st.markdown(f"```\n{clause_wording}\n```")

                                                 # Add a subtle separator between evidence items
                                                 if ev_index < len(evidence_list) - 1:
                                                     st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
                                        else:
                                            st.markdown("**Evidence:** _None provided by AI._")
                                    # --- END Expander UI ---
                    else: # Fallback if tabs failed or mismatch in lengths
                        if not category_tabs: st.warning("Could not create tabs. Displaying results as a single list.")
                        else: st.warning("Mismatch between tabs and categories. Displaying results as a single list.")
                        for item in results_list:
                             if isinstance(item, dict): st.json(item) # Simple JSON display as fallback
                             else: st.warning(f"Skipping invalid item in results list: {type(item)}")

                else: st.info("No results found or categories could not be determined.")

                # --- Excel Download (Button moved to sidebar) ---
                if not st.session_state.viewing_history:
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("## Export Results")
                    if st.sidebar.button("Prepare Data for Excel Download", key="prep_excel", use_container_width=True):
                        st.session_state.excel_data = None; excel_prep_status = st.sidebar.empty(); excel_prep_status.info("Preparing Excel data...")
                        try:
                            excel_rows = [];
                            # Use the sorted list again for export consistency
                            sorted_results_for_export = sorted(results_list, key=lambda x: x.get('Question Number', float('inf')) if isinstance(x, dict) else float('inf'))
                            for item in sorted_results_for_export:
                                if not isinstance(item, dict): continue # Skip non-dict items

                                references = []; first_search_text = "N/A"; evidence = item.get("Evidence")
                                file_name_for_excel = st.session_state.current_filename # Use current filename
                                gen_time_for_excel = item.get("Generation Time", "N/A") # Get from item if available
                                checklist_name_for_excel = item.get("Checklist Name", "N/A") # Get from item if available

                                if evidence and isinstance(evidence, list):
                                    for i, ev in enumerate(evidence):
                                        if isinstance(ev, dict): references.append(str(ev.get("Clause Reference", "N/A")));
                                        if i == 0 and isinstance(ev, dict): first_search_text = ev.get("Searchable Clause Text", "N/A")

                                excel_row = {
                                    "File Name": file_name_for_excel, "Generation Time": gen_time_for_excel,
                                    "Checklist Name": checklist_name_for_excel,
                                    "Question Number": item.get("Question Number"), "Question Category": item.get("Question Category", "Uncategorized"),
                                    "Question": item.get("Question", "N/A"), "Answer": item.get("Answer", "N/A"),
                                    "Answer Justification": item.get("Answer Justification", "N/A"),
                                    "Clause References (Concatenated)": "; ".join(references) if references else "N/A",
                                    "First Searchable Clause Text": first_search_text
                                }
                                excel_rows.append(excel_row)

                            if not excel_rows: excel_prep_status.warning("No data available to export."); st.session_state.excel_data = None
                            else:
                                df_excel = pd.DataFrame(excel_rows); final_columns = [col for col in EXCEL_COLUMN_ORDER if col in df_excel.columns]; extra_cols = [col for col in df_excel.columns if col not in final_columns]; df_excel = df_excel[final_columns + extra_cols]
                                output = io.BytesIO();
                                with pd.ExcelWriter(output, engine='openpyxl') as writer: df_excel.to_excel(writer, index=False, sheet_name='Analysis Results')
                                st.session_state.excel_data = output.getvalue(); excel_prep_status.success("✅ Excel file ready!"); time.sleep(2); excel_prep_status.empty()
                        except Exception as excel_err: excel_prep_status.error(f"Excel Prep Error: {excel_err}"); print(traceback.format_exc())

                    if st.session_state.excel_data:
                        current_filename_for_download = st.session_state.current_filename or "analysis"
                        safe_base_name = re.sub(r'[^\w\s-]', '', os.path.splitext(current_filename_for_download)[0]).strip().replace(' ', '_'); download_filename = f"Analysis_{safe_base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                        st.sidebar.download_button(label="📥 Download Results as Excel", data=st.session_state.excel_data, file_name=download_filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_excel_final", use_container_width=True)


        elif st.session_state.processing_in_progress: st.info("⏳ Analysis is currently in progress...")
        elif not st.session_state.analysis_complete and st.session_state.pdf_bytes is not None and not st.session_state.viewing_history: st.info("PDF loaded. Select/Create a checklist and click 'Analyse Document' in the sidebar.")


    # --- Column 2: PDF Viewer (Sticky) ---
    with col2:
        st.markdown('<div class="sticky-viewer-content">', unsafe_allow_html=True) # Sticky wrapper start
        st.subheader("📄 PDF Viewer"); viewer_status_placeholder = st.empty()

        # --- Handle Search Trigger ---
        if st.session_state.search_trigger:
            search_info = st.session_state.search_trigger; st.session_state.search_trigger = None # Clear trigger
            search_text_to_find = search_info['text']
            search_ref = search_info['ref']
            if not search_text_to_find or not search_text_to_find.strip():
                 viewer_status_placeholder.warning(f"⚠️ Cannot search: No searchable text provided for '{search_ref}'.")
                 st.session_state.last_search_result = None # Ensure no previous highlight remains
            else:
                with st.spinner(f"🔎 Searching for text related to: '{search_ref}'..."):
                    # Pass actual bytes to the search function
                    found_page, instances, term_used, search_status, all_findings = find_text_in_pdf(
                        st.session_state.pdf_bytes,
                        search_text_to_find
                    )
                if found_page:
                    # Convert fitz.Rect instances to simple tuples for storing in session state
                    instance_tuples = tuple(i.irectuple for i in instances if isinstance(i, fitz.Rect)) if instances else None
                    all_findings_tuples = None
                    if all_findings:
                        # Ensure inner elements are Rects before converting
                        all_findings_tuples = tuple((page, tuple(i.irectuple for i in inst_list if isinstance(i, fitz.Rect))) for page, inst_list in all_findings)


                    st.session_state.last_search_result = {
                        'page': found_page,
                        'instances': instance_tuples, # Store tuples
                        'term': term_used,
                        'status': search_status,
                        'ref': search_ref,
                        'all_findings': all_findings_tuples # Store tuples
                    }
                    st.session_state.current_page = found_page; viewer_status_placeholder.empty(); st.rerun()
                else: st.session_state.last_search_result = None; viewer_status_placeholder.error(search_status)

        # --- Display PDF Page ---
        if st.session_state.pdf_display_ready:
            try:
                # Use cached page count if possible
                @st.cache_data(max_entries=1, show_spinner=False)
                def get_pdf_page_count(_pdf_bytes_hash_for_count):
                    if not st.session_state.pdf_bytes: return 1 # No PDF loaded
                    try:
                        with fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf") as doc:
                            return doc.page_count
                    except Exception as e:
                         print(f"Error getting page count: {e}")
                         return 1 # Fallback

                pdf_hash = st.session_state.get('pdf_bytes_hash')
                total_pages = get_pdf_page_count(pdf_hash) if pdf_hash else 1

            except Exception as pdf_load_err: st.error(f"Error loading PDF for page count: {pdf_load_err}"); total_pages = 1; st.session_state.current_page = 1

            current_display_page = max(1, min(int(st.session_state.get('current_page', 1)), total_pages))
            if current_display_page != st.session_state.get('current_page'):
                 st.session_state.current_page = current_display_page # Correct if out of bounds

            # --- Navigation Controls ---
            nav_cols = st.columns([1, 3, 1])
            with nav_cols[0]: # Previous Button
                if st.button("⬅️ Prev", key="prev_page", disabled=(current_display_page <= 1), use_container_width=True):
                     st.session_state.current_page -= 1; st.session_state.last_search_result = None; st.rerun()
            with nav_cols[2]: # Next Button
                if st.button("Next ➡️", key="next_page", disabled=(current_display_page >= total_pages), use_container_width=True):
                     st.session_state.current_page += 1; st.session_state.last_search_result = None; st.rerun()

            # --- Page Info and Search Context ---
            page_info_text = f"Page {current_display_page} of {total_pages}"; search_context_ref = None
            if st.session_state.last_search_result and st.session_state.last_search_result.get('page') == current_display_page:
                search_context_ref = st.session_state.last_search_result.get('ref', 'Search')
                # Check if 'all_findings' exists and is not None/empty to indicate multi-match context
                if st.session_state.last_search_result.get('all_findings'):
                     page_info_text += f" (🎯 Multi-match: '{search_context_ref}')"
                else: page_info_text += f" (🎯 Ref: '{search_context_ref}')"
            nav_cols[1].markdown(f"<div style='text-align: center; padding-top: 0.5rem;'>{page_info_text}</div>", unsafe_allow_html=True)

            # --- Multi-Match Jump Buttons ---
            # Check if 'all_findings' exists, is not None/empty, and contains actual findings tuples
            if st.session_state.last_search_result and st.session_state.last_search_result.get('all_findings') and isinstance(st.session_state.last_search_result['all_findings'], tuple) and st.session_state.last_search_result['all_findings']:
                multi_findings_tuples = st.session_state.last_search_result['all_findings']
                # Filter out potential empty findings (e.g., page number but empty instance list)
                valid_findings = [f for f in multi_findings_tuples if len(f) == 2 and f[1]] # Ensure it has page and non-empty instances
                found_pages = sorted([f[0] for f in valid_findings])

                if found_pages: # Only show if there are valid pages with findings
                    status_msg = st.session_state.last_search_result.get('status', '');
                    if status_msg: viewer_status_placeholder.info(status_msg) # Show multi-match status

                    st.write("Jump to other matches for this reference:")
                    num_buttons = len(found_pages); btn_cols = st.columns(min(num_buttons, 5)) # Max 5 buttons per row
                    current_search_ref = st.session_state.last_search_result.get('ref', 'unknown')

                    for idx, p_num in enumerate(found_pages):
                        col_idx = idx % len(btn_cols); is_current = (p_num == current_display_page)
                        # Add run_key to jump button key for better uniqueness across runs
                        jump_button_key = f"jump_{p_num}_{current_search_ref}_{st.session_state.run_key}"
                        if btn_cols[col_idx].button(f"Page {p_num}", key=jump_button_key, disabled=is_current, use_container_width=True):
                            st.session_state.current_page = p_num;
                            # Find the corresponding instances tuple for the jumped-to page from the original list
                            new_instances_tuple = next((inst_tuple for pg, inst_tuple in multi_findings_tuples if pg == p_num), None)

                            # Update last_search_result with info for the new page
                            st.session_state.last_search_result['instances'] = new_instances_tuple
                            st.session_state.last_search_result['page'] = p_num
                            term_desc = st.session_state.last_search_result.get('term', 'text')
                            st.session_state.last_search_result['status'] = f"✅ Viewing match for '{term_desc}' on page {p_num}."
                            # Clear the multi-findings marker *after* jump, but keep other details
                            st.session_state.last_search_result['all_findings'] = None
                            st.rerun()

            st.markdown("---")

            # --- Render Page Image ---
            highlights_to_apply_tuples = None; render_status_override = None
            # Check if search result exists, is for the current page, and has instances
            if st.session_state.last_search_result \
               and st.session_state.last_search_result.get('page') == current_display_page \
               and st.session_state.last_search_result.get('instances'):
                 highlights_to_apply_tuples = st.session_state.last_search_result.get('instances') # Use stored tuples
                 # Only override status if it wasn't a multi-match jump status (which clears 'all_findings')
                 if not st.session_state.last_search_result.get('all_findings'):
                     render_status_override = st.session_state.last_search_result.get('status')

            # Use hash for cache invalidation, pass tuple for highlights
            pdf_hash = st.session_state.get('pdf_bytes_hash')
            image_bytes, render_status = render_pdf_page_to_image(
                pdf_hash,
                current_display_page,
                highlight_instances_tuple=highlights_to_apply_tuples,
                dpi=150
            )

            if image_bytes:
                st.image(image_bytes, caption=f"Page {current_display_page} - View", use_container_width=True)
                # Determine final status message to show
                final_status = render_status_override if render_status_override else render_status
                # Don't show status again if multi-match jump buttons were shown (status already displayed above)
                if not (st.session_state.last_search_result and st.session_state.last_search_result.get('all_findings')):
                    if final_status:
                        if "✅" in final_status or "✨" in final_status or "Found" in final_status : viewer_status_placeholder.success(final_status)
                        elif "⚠️" in final_status or "warning" in final_status.lower() or "multiple pages" in final_status.lower(): viewer_status_placeholder.warning(final_status)
                        elif "❌" in final_status or "error" in final_status.lower(): viewer_status_placeholder.error(final_status)
                        else: viewer_status_placeholder.caption(final_status)
                    else: viewer_status_placeholder.empty() # Clear status if none provided
            else:
                 viewer_status_placeholder.error(f"Failed to render page {current_display_page}. {render_status or ''}")
        else:
            # Changed message for clarity when PDF bytes exist but display isn't ready (should be rare)
            if st.session_state.pdf_bytes:
                 st.info("PDF loaded, preparing viewer..."); viewer_status_placeholder.empty()
            # else: PDF bytes are None, which is handled by the outer 'if'


        st.markdown('</div>', unsafe_allow_html=True) # Sticky wrapper end

# --- Fallback message if no PDF loaded (and not viewing history) ---
elif not st.session_state.pdf_bytes and not st.session_state.viewing_history:
     st.info("⬆️ Select or create a checklist, then upload a PDF file using the sidebar to begin. You can also load a previous analysis from the History page.")

# --- History Page Link/Button ---
# Place this at the bottom, outside the main columns if a PDF is loaded, or centered if no PDF
st.divider()
history_cols = st.columns(3)
with history_cols[1]: # Centered column
    if st.button("📜 View Analysis History", key="view_history_button", use_container_width=True):
        try:
            st.switch_page("pages/history.py")
        except Exception as e:
             # Fallback or error message if switching fails (e.g., page doesn't exist)
             st.error(f"Could not navigate to History page. Ensure `pages/history.py` exists. Error: {e}")