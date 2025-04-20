# app.py
# --- COMPLETE FILE (v9 - Dynamic Checklists) ---

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
from datetime import datetime
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

# --- Inject custom CSS (Keep as is) ---
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
        background-color: var(--streamlit-background-color);
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
MODEL_NAME = "gemini-1.5-pro-latest" # Using 1.5 Pro
MAX_VALIDATION_RETRIES = 1
RETRY_DELAY_SECONDS = 3
# PROMPT_FILE = "prompt.txt" # No longer used
LOGO_FILE = "jasper-logo-1.png" # Ensure this file exists
SEARCH_FALLBACK_MIN_LENGTH = 20
SEARCH_PREFIX_MIN_WORDS = 4
CHECKLIST_COLLECTION = "checklists" # Firestore collection for checklists
PROMPT_PLACEHOLDER = "{{QUESTIONS_HERE}}" # Placeholder for injecting questions

# --- Get the absolute path to the directory containing app.py ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(APP_DIR, LOGO_FILE)

# --- Schema Definition (Remains the same for output structure per question) ---
ai_response_schema_dict = {
  "type": "array", "description": "List of question analysis results.",
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
# Updated slightly to be more general
system_instruction_text = """You are an AI assistant specialized in analyzing legal facility agreements. Carefully read the provided document and answer the specific questions listed in the user prompt. Adhere strictly to the requested JSON output schema. Prioritize accuracy and extract evidence directly from the text."""

# --- 3. Helper Function Definitions ---

@st.cache_data(ttl=300, show_spinner="Loading checklists...") # Cache for 5 mins
def load_checklists():
    """Loads checklist names and IDs from Firestore."""
    checklists = {"--- Select Checklist ---": None} # Default option
    try:
        docs = db.collection(CHECKLIST_COLLECTION).stream()
        for doc in docs:
            data = doc.to_dict()
            name = data.get("name", doc.id) # Use name field, fallback to ID
            checklists[name] = doc.id
        # Sort checklists by name, keeping the default option first
        sorted_items = sorted([(name, id) for name, id in checklists.items() if id is not None])
        sorted_checklists = {"--- Select Checklist ---": None}
        sorted_checklists.update(dict(sorted_items))
        return sorted_checklists
    except Exception as e:
        st.error(f"Error loading checklists from Firestore: {e}")
        print(traceback.format_exc())
        return {"--- Select Checklist ---": None, "ERROR": "Could not load"}

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
        df = pd.read_excel(uploaded_file)
        # Validate columns
        required_cols = ["Question Category", "Question", "Answer Options"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return None, f"Excel file is missing required columns: {', '.join(missing_cols)}. Expected columns: {', '.join(required_cols)}."

        # Convert to list of dicts, handling potential NaN values
        questions = df[required_cols].fillna("").to_dict('records')

        # Basic validation on content (optional but recommended)
        valid_questions = []
        for i, q in enumerate(questions):
            if not q.get("Question Category") or not q.get("Question"):
                st.warning(f"Row {i+2}: Skipping question due to missing Category or Question text.")
                continue
            valid_questions.append(q)

        if not valid_questions:
            return None, "No valid questions found in the Excel file (check for missing Category or Question text)."

        return valid_questions, None # Return questions and no error message
    except Exception as e:
        error_msg = f"Error parsing Excel file: {e}"
        print(traceback.format_exc())
        return None, error_msg

def generate_excel_template():
    """Generates an Excel template file in memory."""
    df_template = pd.DataFrame(columns=TEMPLATE_COLUMN_ORDER)
    # Add an example row
    example_row = {"Question Category": "Example Category", "Question": "Is this an example question?", "Answer Options": "Yes, No, Maybe"}
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
        # Clear the cache for load_checklists
        load_checklists.clear()
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
        st.error(f"Prompt Template MUST include the placeholder '{PROMPT_PLACEHOLDER}' where questions should be inserted.")
        return False
    # Note: We allow updating with an empty question list if the user explicitly does so

    data_to_update = {
        "name": checklist_name.strip(),
        "prompt_template": prompt_template,
        "questions": questions_list, # Assume questions might have been edited/cleared
        "last_updated_at": google.cloud.firestore.SERVER_TIMESTAMP,
    }

    try:
        doc_ref = db.collection(CHECKLIST_COLLECTION).document(checklist_id)
        doc_ref.update(data_to_update)
        st.success(f"Checklist '{checklist_name}' updated successfully!")
        # Clear the cache for load_checklists
        load_checklists.clear()
        return True
    except Exception as e:
        st.error(f"Failed to update checklist (ID: {checklist_id}) in Firestore: {e}")
        print(traceback.format_exc())
        return False

def format_questions_for_prompt(questions_list):
    """Formats the list of question dicts into a string for the AI prompt."""
    formatted_string = ""
    current_category = None
    question_number = 0

    # Assign question numbers sequentially
    numbered_questions = []
    for q_dict in questions_list:
        question_number += 1
        q_dict['assigned_number'] = question_number # Store assigned number for validation later
        numbered_questions.append(q_dict)

    # Group by category for display in prompt
    grouped = defaultdict(list)
    for q in numbered_questions:
        grouped[q.get("Question Category", "Uncategorized")].append(q)

    prompt_lines = []
    total_question_count = len(numbered_questions)
    prompt_lines.append(f"**Total Questions in Checklist:** {total_question_count}\n")

    for category, questions_in_category in grouped.items():
        prompt_lines.append(f"\n**Category: {category}**")
        for q in questions_in_category:
            q_num = q['assigned_number']
            q_text = q.get('Question', 'MISSING QUESTION TEXT')
            q_opts = q.get('Answer Options', '')
            prompt_lines.append(f"{q_num}. **Question:** {q_text}")
            if q_opts:
                # Format options clearly
                opts_cleaned = str(q_opts).strip()
                # Simple heuristic: if it looks like a list, format nicely
                if ',' in opts_cleaned or ';' in opts_cleaned or '\n' in opts_cleaned:
                     options = [opt.strip() for opt in re.split(r'[;,|\n]+', opts_cleaned) if opt.strip()]
                     if options:
                         prompt_lines.append(f"   **Answer Options:**")
                         for opt in options:
                             prompt_lines.append(f"     - {opt}")
                     else: # Handle cases like 'Text', 'Date', 'BLANK'
                          prompt_lines.append(f"   **Answer Guidance:** {opts_cleaned}")
                elif opts_cleaned:
                     prompt_lines.append(f"   **Answer Guidance:** {opts_cleaned}")
            # Add a newline for readability between questions
            prompt_lines.append("")


    formatted_string = "\n".join(prompt_lines)
    return formatted_string, numbered_questions # Return formatted string AND questions with assigned numbers

def validate_ai_data(ai_response_data, expected_questions_with_nums):
    """Validates AI response against the schema and expected questions.
       Returns (validated_data, issues_list).
    """
    if not isinstance(ai_response_data, list):
        return None, ["CRITICAL VALIDATION ERROR: AI Response is not a list."]

    validated_data = []
    issues_list = []
    expected_q_nums = {q['assigned_number'] for q in expected_questions_with_nums}
    expected_q_details = {q['assigned_number']: q for q in expected_questions_with_nums}
    found_q_nums = set()

    for index, item in enumerate(ai_response_data):
        q_num = item.get('Question Number')
        q_num_str = f"Q#{q_num}" if q_num is not None else f"Item Index {index}"
        is_outer_valid = True

        if not isinstance(item, dict):
            issues_list.append(f"{q_num_str}: Item is not a dictionary.")
            is_outer_valid = False
            continue # Skip further checks for this item

        # Check Question Number validity
        if not isinstance(q_num, int):
            issues_list.append(f"Item Index {index}: 'Question Number' is missing or not an integer.")
            is_outer_valid = False
        elif q_num not in expected_q_nums:
             issues_list.append(f"{q_num_str}: Unexpected Question Number found in response (was not in the input checklist).")
             is_outer_valid = False
        else:
            found_q_nums.add(q_num)
            # Check if category and question text match expectation (optional but good)
            expected_q_data = expected_q_details.get(q_num)
            if expected_q_data:
                if item.get("Question Category") != expected_q_data.get("Question Category"):
                     issues_list.append(f"{q_num_str}: Question Category mismatch (Expected: '{expected_q_data.get('Question Category')}', Got: '{item.get('Question Category')}')")
                     # Don't fail validation for this, but log it
                if item.get("Question") != expected_q_data.get("Question"):
                     issues_list.append(f"{q_num_str}: Question text mismatch (Expected: '{expected_q_data.get('Question')[:50]}...', Got: '{item.get('Question', '')[:50]}...')")
                     # Don't fail validation for this, but log it

        # Check for required keys
        missing_outer_keys = AI_REQUIRED_KEYS - set(item.keys())
        if missing_outer_keys:
            issues_list.append(f"{q_num_str}: Missing required top-level keys: {missing_outer_keys}")
            is_outer_valid = False

        # Validate Evidence structure (if present)
        evidence_list = item.get("Evidence")
        if "Evidence" in AI_REQUIRED_KEYS and not isinstance(evidence_list, list):
             # Only flag as issue if 'Evidence' is required and not a list
            issues_list.append(f"{q_num_str}: Required 'Evidence' field is not a list (found type: {type(evidence_list).__name__}).")
            is_outer_valid = False
        elif isinstance(evidence_list, list): # Evidence is optional OR present and is a list
            for ev_index, ev_item in enumerate(evidence_list):
                ev_id_str = f"Ev[{ev_index}]"
                if not isinstance(ev_item, dict):
                    issues_list.append(f"{q_num_str} {ev_id_str}: Evidence item is not a dictionary.")
                    is_outer_valid = False; continue
                missing_ev_keys = AI_EVIDENCE_REQUIRED_KEYS - set(ev_item.keys())
                if missing_ev_keys:
                    issues_list.append(f"{q_num_str} {ev_id_str}: Missing required evidence keys: {missing_ev_keys}")
                    is_outer_valid = False
                for key, expected_type in [("Clause Reference", str), ("Clause Wording", str), ("Searchable Clause Text", str)]:
                     # Only check type if key is required AND present
                     if key in AI_EVIDENCE_REQUIRED_KEYS and key in ev_item and not isinstance(ev_item.get(key), expected_type):
                         issues_list.append(f"{q_num_str} {ev_id_str}: Key '{key}' has incorrect type (expected {expected_type.__name__}, got {type(ev_item.get(key)).__name__}).")
                         is_outer_valid = False

        if is_outer_valid:
            validated_data.append(item)

    # Final check: Were all expected questions answered?
    missing_q_nums = expected_q_nums - found_q_nums
    if missing_q_nums:
        issues_list.append(f"Checklist Analysis: Missing answers for expected Question Numbers: {sorted(list(missing_q_nums))}")

    if issues_list:
        issues_list.insert(0, f"Validation Issues Found ({len(validated_data)} items passed validation):")

    # Return validated data (even if empty) or None if critical error occurred
    if validated_data is None and isinstance(ai_response_data, list): return [], issues_list # Empty list if input was list but failed validation
    elif validated_data is None: return None, issues_list # None if input wasn't even a list
    else: return validated_data, issues_list


def generate_checklist_analysis(checklist_prompt_template, checklist_questions, uploaded_file_ref, status_placeholder, api_key_to_use, gen_config_params):
    """Generates analysis for the entire checklist using a specific API key and generation config."""
    try:
        genai.configure(api_key=api_key_to_use)
    except Exception as config_err:
        status_placeholder.error(f"❌ Invalid API Key provided or configuration failed: {config_err}")
        return None, "Failed", [f"Invalid API Key or config error: {config_err}"]

    status_placeholder.info(f"🔄 Preparing prompt and starting analysis...")
    analysis_warnings = []

    try:
        # 1. Format questions and get the numbered list back for validation
        formatted_questions_str, numbered_questions_for_validation = format_questions_for_prompt(checklist_questions)
        if not numbered_questions_for_validation:
            raise ValueError("Failed to format or number questions for the prompt.")

        # 2. Integrate questions into the prompt template
        if PROMPT_PLACEHOLDER not in checklist_prompt_template:
             raise ValueError(f"Prompt template is missing the required placeholder: {PROMPT_PLACEHOLDER}")
        final_prompt_for_api = checklist_prompt_template.replace(PROMPT_PLACEHOLDER, formatted_questions_str)

        # Add a final instruction about the JSON schema (similar to old prompt footer)
        final_instruction = "\n\n**Final Instruction:** Ensure the final output is a valid JSON array containing an object for **all** questions listed above. Each question object must follow the specified schema precisely, including all required keys (`Question Number`, `Question Category`, `Question`, `Answer`, `Answer Justification`, `Evidence`). The `Question Number` must match the number assigned in the list above. Ensure the `Evidence` array contains objects with *all* required keys (`Clause Reference`, `Clause Wording`, `Searchable Clause Text`) or is an empty array (`[]`) if no direct evidence applies (e.g., for 'Information Not Found' or 'N/A' answers). Double-check this structure carefully."
        final_prompt_for_api += final_instruction

        # 3. Setup GenAI Model and Config
        model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction_text)
        generation_config = types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=ai_response_schema_dict,
            temperature=gen_config_params['temperature'],
            top_p=gen_config_params['top_p'],
            top_k=gen_config_params['top_k']
        )
        safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]

        # 4. Call API and Validate (with retries)
        final_validated_data = None
        for attempt in range(1, MAX_VALIDATION_RETRIES + 2):
            if attempt > 1:
                status_placeholder.info(f"⏳ Retrying generation/validation (Attempt {attempt}/{MAX_VALIDATION_RETRIES+1})..."); time.sleep(RETRY_DELAY_SECONDS)
            try:
                if not uploaded_file_ref or not hasattr(uploaded_file_ref, 'name'):
                    raise ValueError("Invalid or missing uploaded file reference for GenAI call.")

                contents = [uploaded_file_ref, final_prompt_for_api]
                status_placeholder.info(f"🧠 Calling AI (Attempt {attempt})...")
                response = model.generate_content(contents=contents, generation_config=generation_config, safety_settings=safety_settings, request_options={'timeout': 600}) # 10 min timeout

                parsed_ai_data = None; validated_ai_data = None; validation_issues = []
                status_placeholder.info(f"🔍 Processing response (Attempt {attempt})...")

                if response.parts:
                    full_response_text = response.text
                    try:
                        # Handle potential markdown ```json ... ``` wrapping
                        match = re.search(r"```json\s*([\s\S]*?)\s*```", full_response_text, re.IGNORECASE | re.DOTALL)
                        json_text = match.group(1).strip() if match else full_response_text.strip()

                        if not json_text: raise json.JSONDecodeError("Extracted JSON content is empty.", json_text, 0)
                        parsed_ai_data = json.loads(json_text)
                        status_placeholder.info(f"✔️ Validating structure...")
                        validated_ai_data, validation_issues = validate_ai_data(parsed_ai_data, numbered_questions_for_validation)
                        analysis_warnings.extend(validation_issues)

                        if validated_ai_data is not None and len(validated_ai_data) > 0:
                            # Check if all questions were answered (validation_issues might contain this)
                            if any("Missing answers" in issue for issue in validation_issues):
                                 status_placeholder.warning(f"⚠️ Validation successful, but some questions might be missing answers. See summary. (Attempt {attempt}).")
                                 # Continue, but consider it a partial success
                                 final_validated_data = validated_ai_data
                                 break
                            else:
                                final_validated_data = validated_ai_data
                                status_placeholder.info(f"✅ Validation successful.")
                                break # Success! Exit retry loop.
                        elif validated_ai_data is not None and len(validated_ai_data) == 0 and not validation_issues:
                            status_placeholder.warning(f"⚠️ AI returned an empty list (Attempt {attempt}). Check prompt/document.")
                            analysis_warnings.append("AI returned empty list.")
                        else: # Validation failed or produced no data
                            error_msg = f"Validation failed. Issues: {validation_issues}"
                            status_placeholder.warning(f"⚠️ {error_msg} (Attempt {attempt}).")
                        if validated_ai_data is None: # Critical validation error (e.g., not a list)
                            analysis_warnings.append("CRITICAL validation error.")

                    except json.JSONDecodeError as json_err:
                        error_msg = f"JSON Decode Error on attempt {attempt}: {json_err}. Raw text received: '{full_response_text[:500]}...'"
                        st.error(error_msg); st.code(full_response_text, language='text')
                        analysis_warnings.append(error_msg)
                    except Exception as parse_validate_err:
                        error_msg = f"Unexpected Error during parsing/validation on attempt {attempt}: {type(parse_validate_err).__name__}: {parse_validate_err}"
                        st.error(error_msg); analysis_warnings.append(error_msg); print(traceback.format_exc())
                else: # No response parts (blocked, etc.)
                    block_reason = "Unknown"; block_message = "N/A"; finish_reason = "Unknown"
                    try:
                        if response.prompt_feedback: block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown'); block_reason = block_reason.name if hasattr(block_reason, 'name') else str(block_reason); block_message = response.prompt_feedback.block_reason_message or "N/A"
                        if response.candidates: finish_reason = getattr(response.candidates[0], 'finish_reason', 'Unknown'); finish_reason = finish_reason.name if hasattr(finish_reason, 'name') else str(finish_reason)
                    except AttributeError: pass # Ignore errors if fields don't exist
                    if finish_reason == "SAFETY": warn_msg = f"API Response Blocked (Attempt {attempt}): Reason: SAFETY. Detail: {block_reason}. Message: {block_message}"; st.error(warn_msg)
                    elif finish_reason == "RECITATION": warn_msg = f"API Response Potentially Blocked (Attempt {attempt}): Finish Reason: RECITATION. Block Reason: {block_reason}."; st.warning(warn_msg)
                    elif finish_reason == "STOP" and not final_validated_data: warn_msg = f"API Response Ended (Attempt {attempt}): Finish Reason: STOP, but no valid data parsed yet."; st.info(warn_msg)
                    elif finish_reason == "MAX_TOKENS": warn_msg = f"API Response Ended (Attempt {attempt}): Finish Reason: MAX_TOKENS. Response might be incomplete."; st.warning(warn_msg)
                    else: warn_msg = f"API Issue (Attempt {attempt}): Finish Reason: {finish_reason}. Block Reason: {block_reason}. Response may be incomplete or empty."; st.warning(warn_msg)
                    analysis_warnings.append(warn_msg)

            except types.StopCandidateException as sce: error_msg = f"Generation Stopped Error (Attempt {attempt}): {sce}."; st.error(error_msg); analysis_warnings.append(error_msg); print(traceback.format_exc())
            except google.api_core.exceptions.GoogleAPIError as api_err:
                 error_msg = f"Google API Error (Attempt {attempt}): {type(api_err).__name__}: {api_err}."; st.error(error_msg); analysis_warnings.append(error_msg); print(traceback.format_exc())
            except Exception as e: error_msg = f"Processing Error during API call/prompt generation (Attempt {attempt}): {type(e).__name__}: {e}"; st.error(error_msg); analysis_warnings.append(error_msg); analysis_warnings.append("Traceback logged to console."); print(traceback.format_exc())

            if final_validated_data is not None:
                break # Exit retry loop if successful

        # After retry loop
        if final_validated_data is not None:
            if any("Missing answers" in issue for issue in analysis_warnings):
                 status_placeholder.warning(f"⚠️ Analysis completed, but some questions may be missing. Check results.")
                 return final_validated_data, "Partial Success", analysis_warnings
            else:
                 status_placeholder.success(f"✅ Analysis completed successfully.")
                 return final_validated_data, "Success", analysis_warnings
        else:
            status_placeholder.error(f"❌ Analysis failed after {attempt} attempts.")
            analysis_warnings.append(f"Failed to get valid response after {MAX_VALIDATION_RETRIES + 1} attempts.")
            return None, "Failed", analysis_warnings

    except Exception as outer_err:
        error_msg = f"Critical Error during setup or execution: {type(outer_err).__name__}: {outer_err}"
        st.error(error_msg); analysis_warnings.append(error_msg); analysis_warnings.append("Traceback logged to console."); print(traceback.format_exc())
        status_placeholder.error(f"❌ Critical failure during analysis.")
        return None, "Failed", analysis_warnings


# --- PDF Search/Render Functions (Keep as is) ---
@st.cache_data(show_spinner=False)
def find_text_in_pdf(_pdf_bytes, search_text):
    """Searches PDF. Returns (first_page_found, instances_on_first_page, term_used, status_msg, all_findings)"""
    if not _pdf_bytes or not search_text: return None, None, None, "Invalid input (PDF bytes or search text missing).", None
    doc = None; search_text_cleaned = search_text.strip(); words = search_text_cleaned.split(); num_words = len(words)
    search_attempts = []
    # --- Build Search Terms List (Prioritized) ---
    if num_words >= SEARCH_PREFIX_MIN_WORDS and num_words > 5: term_10 = ' '.join(words[:10]); search_attempts.append({'term': term_10, 'desc': "first 10 words"})
    # Corrected logic for term_5
    if num_words >= SEARCH_PREFIX_MIN_WORDS:
        term_5 = ' '.join(words[:5])
        if not search_attempts or term_5 != search_attempts[0]['term']: search_attempts.append({'term': term_5, 'desc': "first 5 words"})
    term_full = search_text_cleaned;
    if term_full and not any(term_full == a['term'] for a in search_attempts): search_attempts.append({'term': term_full, 'desc': "full text"})
    sentences = re.split(r'(?<=[.?!])\s+', term_full); term_sentence = sentences[0].strip() if sentences else ""
    if len(term_sentence) >= SEARCH_FALLBACK_MIN_LENGTH and not any(term_sentence == a['term'] for a in search_attempts): search_attempts.append({'term': term_sentence, 'desc': "first sentence fallback"})
    # --- Execute Search Attempts ---
    try:
        doc = fitz.open(stream=_pdf_bytes, filetype="pdf"); search_flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES
        for attempt in search_attempts:
            term = attempt['term']; desc = attempt['desc']; findings_for_term = []
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index); instances = page.search_for(term, flags=search_flags, quads=False)
                if instances: findings_for_term.append((page_index + 1, instances))
            if findings_for_term:
                doc.close(); first_page_found = findings_for_term[0][0]; instances_on_first_page = findings_for_term[0][1]
                if len(findings_for_term) == 1: status = f"✅ Found using '{desc}' on page {first_page_found}."; return first_page_found, instances_on_first_page, term, status, None
                else: pages_found = sorted([f[0] for f in findings_for_term]); status = f"⚠️ Found matches using '{desc}' on multiple pages: {pages_found}. Showing first match on page {first_page_found}."; return first_page_found, instances_on_first_page, term, status, findings_for_term
        doc.close(); tried_descs = [a['desc'] for a in search_attempts]; return None, None, None, f"❌ Text not found (tried methods: {', '.join(tried_descs)}).", None
    except Exception as e:
        if doc: doc.close(); print(f"ERROR searching PDF: {e}\n{traceback.format_exc()}"); return None, None, None, f"❌ Error during PDF search: {e}", None

def render_pdf_page_to_image(_pdf_bytes, page_number, highlight_instances=None, dpi=150):
    """Renders PDF page to PNG image bytes, applying highlights. Returns (image_bytes, status_msg)."""
    if not _pdf_bytes or page_number < 1: return None, "Invalid input for rendering (PDF bytes missing or invalid page number)."
    doc = None; image_bytes = None; render_status_message = f"Rendered page {page_number}."
    try:
        doc = fitz.open(stream=_pdf_bytes, filetype="pdf"); page_index = page_number - 1
        if page_index < 0 or page_index >= doc.page_count: doc.close(); return None, f"Page number {page_number} is out of range (Total pages: {doc.page_count})."
        page = doc.load_page(page_index); highlight_applied_count = 0
        if highlight_instances:
            try:
                for inst in highlight_instances:
                    if isinstance(inst, (fitz.Rect, fitz.Quad)):
                        highlight = page.add_highlight_annot(inst)
                        if highlight: highlight.set_colors(stroke=fitz.utils.getColor("yellow")); highlight.set_opacity(0.4); highlight.update(); highlight_applied_count += 1
                        else: print(f"WARN: Failed to add highlight annotation for instance: {inst} on page {page_number}")
                if highlight_applied_count > 0: render_status_message = f"Rendered page {page_number} with {highlight_applied_count} highlight(s)."
                elif highlight_instances: render_status_message = f"Rendered page {page_number}, but no valid highlights applied from provided instances."
            except Exception as highlight_err: print(f"ERROR applying highlights on page {page_number}: {highlight_err}\n{traceback.format_exc()}"); render_status_message = f"⚠️ Error applying highlights: {highlight_err}"
        pix = page.get_pixmap(dpi=dpi, alpha=False); image_bytes = pix.tobytes("png")
    except Exception as e: print(f"ERROR rendering page {page_number}: {e}\n{traceback.format_exc()}"); render_status_message = f"❌ Error rendering page {page_number}: {e}"; image_bytes = None
    finally:
        if doc: doc.close()
    return image_bytes, render_status_message

# --- GCS Upload/Download Functions (Keep as is) ---
def upload_to_gcs(bucket_name, source_bytes, destination_blob_name, status_placeholder=None):
    """Uploads bytes to GCS bucket."""
    if status_placeholder: status_placeholder.info(f"☁️ Uploading PDF to Google Cloud Storage...")
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(source_bytes, content_type='application/pdf')
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
        pdf_bytes = blob.download_as_bytes()
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
    current_temp = st.session_state.get('ai_temperature', 0.2)
    current_top_p = st.session_state.get('ai_top_p', 0.95)
    current_top_k = st.session_state.get('ai_top_k', 40)
    # current_steve_mode = st.session_state.get('steve_test_mode', False) # Removed

    # Clear most state variables
    keys_to_reset = [
        'pdf_bytes', 'pdf_display_ready', 'analysis_results', 'analysis_complete',
        'processing_in_progress', 'current_page', 'run_status_summary',
        'excel_data', 'search_trigger', 'last_search_result', 'show_wording_states',
        'viewing_history', 'history_filename', 'history_timestamp', 'current_filename',
        'load_history_id', 'run_key',
        'selected_checklist_id', 'selected_checklist_name', # NEW Checklist state
        'current_checklist_prompt', 'current_checklist_questions', # NEW Checklist state
        'new_checklist_name', 'uploaded_checklist_file', 'checklist_action' # UI state
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
    # st.session_state.steve_test_mode = current_steve_mode # Removed


# --- 4. Initialize Session State ---
def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    state_defaults = {
        'show_wording_states': defaultdict(bool),
        'current_page': 1,
        'analysis_results': None,
        'pdf_bytes': None,
        'pdf_display_ready': False,
        'processing_in_progress': False,
        'analysis_complete': False,
        'run_key': 0,
        'run_status_summary': [],
        'excel_data': None,
        'search_trigger': None,
        'last_search_result': None,
        'api_key': None,
        'load_history_id': None,
        'viewing_history': False,
        'history_filename': None,
        'history_timestamp': None,
        'current_filename': None,
        # 'steve_test_mode': False, # Removed
        'ai_temperature': 0.2, # AI Temperature
        'ai_top_p': 0.95,      # AI Top-P
        'ai_top_k': 40,        # AI Top-K
        # --- New Checklist State ---
        'selected_checklist_id': None,
        'selected_checklist_name': None,
        'current_checklist_prompt': "", # Prompt for the selected/new checklist
        'current_checklist_questions': [], # Questions list for selected/new checklist
        'new_checklist_name': "", # For the create new form
        'uploaded_checklist_file': None, # Holds the uploaded file object
        'checklist_action': None, # Tracks if user wants to create/edit
        'available_checklists': {}, # Stores loaded checklist names/IDs
    }
    for key, default_value in state_defaults.items():
        if key not in st.session_state:
            # Use deepcopy for mutable defaults like defaultdict
            st.session_state[key] = copy.deepcopy(default_value) if isinstance(default_value, (dict, list, set, defaultdict)) else default_value
    # Ensure available_checklists is loaded if empty
    if not st.session_state.available_checklists or "--- Select Checklist ---" not in st.session_state.available_checklists :
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
            timestamp = run_data.get("analysis_timestamp") # Firestore timestamp object
            run_summary = run_data.get("run_status", []) # Load summary if saved
            checklist_name_hist = run_data.get("checklist_name", "Unknown") # Load checklist name

            if not gcs_pdf_path:
                 st.error("❌ History record is missing the PDF file path (gcs_pdf_path). Cannot load.")
            elif not results:
                 st.error("❌ History record is missing analysis results. Cannot load.")
            else:
                # Download PDF from GCS
                if gcs_pdf_path.startswith("gs://"):
                    path_parts = gcs_pdf_path[5:].split("/", 1)
                    hist_bucket_name = path_parts[0]
                    hist_blob_name = path_parts[1]
                else: # Assume path format is FOLDER/FILENAME.PDF
                     hist_bucket_name = GCS_BUCKET_NAME # Use configured bucket
                     hist_blob_name = gcs_pdf_path # Treat the whole thing as the blob name relative to bucket root

                with st.spinner(f"Downloading PDF from gs://{hist_bucket_name}/{hist_blob_name}..."):
                    pdf_bytes_from_hist = download_from_gcs(hist_bucket_name, hist_blob_name)

                if pdf_bytes_from_hist:
                    # Reset state before loading historical data BUT preserve some settings
                    reset_app_state()

                    # Load historical data into session state
                    st.session_state.pdf_bytes = pdf_bytes_from_hist
                    st.session_state.analysis_results = results
                    st.session_state.run_status_summary = run_summary
                    st.session_state.analysis_complete = True # Mark as complete (for display purposes)
                    st.session_state.pdf_display_ready = True
                    st.session_state.viewing_history = True # Set history mode flag
                    st.session_state.history_filename = filename
                    st.session_state.history_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp)
                    st.session_state.current_page = 1 # Reset to first page
                    st.session_state.current_filename = filename # Store filename for potential export
                    # Store the checklist name used for this historical run
                    st.session_state.history_checklist_name = checklist_name_hist

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

# --- Sidebar Setup ---
st.sidebar.title("Controls")

# --- Add Logo ---
if os.path.exists(LOGO_PATH):
    try: logo = Image.open(LOGO_PATH); st.sidebar.image(logo, use_column_width=True)
    except Exception as img_err: st.sidebar.warning(f"Could not load logo: {img_err}")
else: st.sidebar.warning(f"Logo file '{LOGO_FILE}' not found.")
st.sidebar.markdown("---") # Separator after logo

# --- Display History Mode Banner / Title ---
if st.session_state.viewing_history:
    hist_ts_str = st.session_state.history_timestamp or "N/A"
    hist_checklist = st.session_state.get('history_checklist_name', 'Unknown')
    st.title("JASPER - Historical Analysis Review")
    st.info(f"📜 **Viewing Historical Analysis:** File: **{st.session_state.history_filename}**\n"
            f"Checklist Used: **{hist_checklist}** (Generated: {hist_ts_str})")
    if st.button("⬅️ Exit History View / Start New Analysis", key="clear_history_view"):
        reset_app_state()
        st.rerun()
    st.markdown("---") # Separator

else: # Not viewing history
    st.title("JASPER - Just A Smart Platform for Extraction and Review")
    if not st.session_state.analysis_complete:
        st.markdown("Select or upload a checklist, upload a PDF, enter API key, adjust settings, click 'Analyse'.")

    # --- Conditional Controls (Only show if NOT viewing history) ---

    # --- Checklist Management ---
    st.sidebar.markdown("### 1. Checklist Setup")
    checklist_status_placeholder = st.sidebar.empty()

    # Load available checklists if needed
    if not st.session_state.available_checklists or "--- Select Checklist ---" not in st.session_state.available_checklists:
         st.session_state.available_checklists = load_checklists()

    # Checklist Selection Dropdown
    selected_checklist_name = st.sidebar.selectbox(
        "Select Existing Checklist or Create New",
        options=list(st.session_state.available_checklists.keys()) + ["--- Create New Checklist ---"],
        key='checklist_selector',
        index=list(st.session_state.available_checklists.values()).index(st.session_state.selected_checklist_id) if st.session_state.selected_checklist_id in st.session_state.available_checklists.values() else 0,
        help="Choose a previously saved checklist or start creating a new one."
    )

    # Handle selection change
    if selected_checklist_name == "--- Create New Checklist ---":
        if st.session_state.checklist_action != 'create':
            # Reset fields for creation mode
            st.session_state.checklist_action = 'create'
            st.session_state.selected_checklist_id = None
            st.session_state.selected_checklist_name = None
            st.session_state.new_checklist_name = ""
            st.session_state.current_checklist_prompt = f"Analyse the document based on the following questions:\n\n{PROMPT_PLACEHOLDER}\n\nProvide answers in the required JSON format." # Default prompt
            st.session_state.current_checklist_questions = []
            st.session_state.uploaded_checklist_file = None
            st.rerun() # Rerun to update UI for creation mode
    elif selected_checklist_name != "--- Select Checklist ---":
        newly_selected_id = st.session_state.available_checklists.get(selected_checklist_name)
        if newly_selected_id != st.session_state.selected_checklist_id:
            # Load the selected checklist's details
            checklist_data = load_checklist_details(newly_selected_id)
            if checklist_data:
                st.session_state.selected_checklist_id = newly_selected_id
                st.session_state.selected_checklist_name = checklist_data.get("name", selected_checklist_name)
                st.session_state.current_checklist_prompt = checklist_data.get("prompt_template", "")
                st.session_state.current_checklist_questions = checklist_data.get("questions", [])
                st.session_state.checklist_action = 'selected' # Indicate an existing one is selected
                st.session_state.new_checklist_name = "" # Clear create form field
                st.session_state.uploaded_checklist_file = None # Clear upload field
                checklist_status_placeholder.success(f"Loaded checklist: '{st.session_state.selected_checklist_name}'")
                time.sleep(1); checklist_status_placeholder.empty()
                st.rerun() # Update UI
            else:
                checklist_status_placeholder.error(f"Failed to load details for '{selected_checklist_name}'.")
                st.session_state.selected_checklist_id = None # Reset selection
                st.session_state.checklist_action = None
    elif selected_checklist_name == "--- Select Checklist ---" and st.session_state.selected_checklist_id is not None:
         # User explicitly selected the placeholder, reset state
         st.session_state.selected_checklist_id = None
         st.session_state.selected_checklist_name = None
         st.session_state.current_checklist_prompt = ""
         st.session_state.current_checklist_questions = []
         st.session_state.checklist_action = None
         st.rerun()


    # --- Checklist Creation/Editing Form ---
    if st.session_state.checklist_action == 'create':
        st.sidebar.markdown("**Create New Checklist:**")
        st.session_state.new_checklist_name = st.sidebar.text_input(
            "New Checklist Name",
            value=st.session_state.new_checklist_name,
            key="new_checklist_name_input",
            placeholder="Enter a unique name"
        )
        uploaded_excel = st.sidebar.file_uploader(
            "Upload Questions (.xlsx)",
            type="xlsx",
            key="checklist_uploader",
            help=f"Excel file must have columns: {', '.join(TEMPLATE_COLUMN_ORDER)}"
        )
        # Parse uploaded file immediately if present
        if uploaded_excel is not None and uploaded_excel != st.session_state.get('uploaded_checklist_file_obj'):
            st.session_state.uploaded_checklist_file_obj = uploaded_excel # Store file object
            parsed_questions, error = parse_checklist_excel(uploaded_excel)
            if error:
                checklist_status_placeholder.error(error)
                st.session_state.current_checklist_questions = []
            else:
                checklist_status_placeholder.success(f"Parsed {len(parsed_questions)} questions from Excel.")
                st.session_state.current_checklist_questions = parsed_questions
                time.sleep(1); checklist_status_placeholder.empty()
                # Don't rerun here, let user edit prompt

        st.session_state.current_checklist_prompt = st.sidebar.text_area(
            "Checklist Prompt Template",
            value=st.session_state.current_checklist_prompt,
            key="new_checklist_prompt_input",
            height=150,
            help=f"Define the prompt text. MUST include '{PROMPT_PLACEHOLDER}' where the questions will be automatically inserted."
        )
        if st.sidebar.button("💾 Save New Checklist", key="save_new_checklist", use_container_width=True, type="primary",
                             disabled=not (st.session_state.new_checklist_name and st.session_state.current_checklist_prompt and st.session_state.current_checklist_questions)):
            saved_id, success = save_checklist_to_firestore(
                st.session_state.new_checklist_name,
                st.session_state.current_checklist_prompt,
                st.session_state.current_checklist_questions
            )
            if success:
                st.session_state.available_checklists = load_checklists() # Refresh list
                st.session_state.selected_checklist_id = saved_id # Auto-select the new one
                st.session_state.selected_checklist_name = st.session_state.new_checklist_name
                st.session_state.checklist_action = 'selected' # Switch mode
                st.rerun()

    elif st.session_state.checklist_action == 'selected':
        st.sidebar.markdown(f"**Selected Checklist:** {st.session_state.selected_checklist_name}")
        st.session_state.current_checklist_prompt = st.sidebar.text_area(
            "Checklist Prompt Template",
            value=st.session_state.current_checklist_prompt,
            key="edit_checklist_prompt_input",
            height=150,
            help=f"Edit the prompt template. MUST include '{PROMPT_PLACEHOLDER}'. Save changes below."
        )
        # Maybe add ability to view/edit questions later? For now, just prompt.
        st.sidebar.caption(f"Contains {len(st.session_state.current_checklist_questions)} questions (defined at creation).")

        if st.sidebar.button("💾 Save Prompt Changes", key="update_checklist", use_container_width=True,
                             disabled=not st.session_state.selected_checklist_id):
            success = update_checklist_in_firestore(
                st.session_state.selected_checklist_id,
                st.session_state.selected_checklist_name, # Keep name same on update for now
                st.session_state.current_checklist_prompt,
                st.session_state.current_checklist_questions # Keep questions same on prompt update
            )
            if success:
                 checklist_status_placeholder.success("Prompt updated.")
                 time.sleep(1); checklist_status_placeholder.empty()
                 # No rerun needed unless name changed, which isn't supported here yet

    # --- Download Template Button ---
    excel_template_bytes = generate_excel_template()
    st.sidebar.download_button(
        label="📄 Download Checklist Template (.xlsx)",
        data=excel_template_bytes,
        file_name="checklist_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_template",
        use_container_width=True,
        help="Download an Excel template with the required columns for uploading questions."
    )
    st.sidebar.markdown("---") # Separator

    # --- Analysis Setup ---
    st.sidebar.markdown("### 2. Analysis Setup")

    # --- API Key Input ---
    api_key_input = st.sidebar.text_input("Enter Google AI Gemini API Key", type="password", key="api_key_input", help="Your API key is used only for this session and is not stored.", value=st.session_state.get("api_key", ""))
    if api_key_input and api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input

    if not st.session_state.api_key and not st.session_state.analysis_complete and not st.session_state.processing_in_progress : st.sidebar.warning("API Key required to run analysis.", icon="🔑")

    # --- File Upload ---
    uploaded_file_obj = st.sidebar.file_uploader("Upload Facility Agreement PDF", type="pdf", key=f"pdf_uploader_{st.session_state.get('run_key', 0)}")
    if uploaded_file_obj is not None:
        new_file_bytes = uploaded_file_obj.getvalue()
        if new_file_bytes != st.session_state.get('pdf_bytes'):
            # Reset state but preserve API key/AI params/Checklist Selection
            current_checklist_id = st.session_state.selected_checklist_id
            current_checklist_name = st.session_state.selected_checklist_name
            current_prompt = st.session_state.current_checklist_prompt
            current_questions = st.session_state.current_checklist_questions
            current_action = st.session_state.checklist_action

            reset_app_state() # Resets most things

            # Restore checklist state
            st.session_state.selected_checklist_id = current_checklist_id
            st.session_state.selected_checklist_name = current_checklist_name
            st.session_state.current_checklist_prompt = current_prompt
            st.session_state.current_checklist_questions = current_questions
            st.session_state.checklist_action = current_action

            # Load new PDF
            st.session_state.pdf_bytes = new_file_bytes
            st.session_state.pdf_display_ready = True
            st.session_state.current_filename = uploaded_file_obj.name
            st.toast("✅ New PDF file loaded. Viewer ready.", icon="📄")
            st.rerun()
    elif not st.session_state.pdf_bytes:
         st.sidebar.info("Upload a PDF to enable analysis.")

    st.sidebar.markdown("### 3. Analysis Options")
    # --- Steve Test Mode Toggle (Removed) ---

    # --- AI Parameter Controls ---
    with st.sidebar.expander("Advanced AI Settings"):
        st.session_state.ai_temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, step=0.05, value=st.session_state.ai_temperature, help="Controls randomness.")
        st.session_state.ai_top_p = st.slider("Top-P", min_value=0.0, max_value=1.0, step=0.05, value=st.session_state.ai_top_p, help="Nucleus sampling.")
        st.session_state.ai_top_k = st.number_input("Top-K", min_value=1, step=1, value=st.session_state.ai_top_k, help="Considers top K tokens.")

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
        not st.session_state.analysis_complete
    )

    tooltip_parts = []
    if st.session_state.analysis_complete: tooltip_parts.append("Analysis complete for the current file.")
    if st.session_state.processing_in_progress: tooltip_parts.append("Analysis is currently running.")
    if not st.session_state.pdf_bytes: tooltip_parts.append("Upload a PDF file.")
    if not st.session_state.api_key: tooltip_parts.append("Enter your Gemini API key.")
    if not st.session_state.selected_checklist_id: tooltip_parts.append("Select a checklist.")
    if st.session_state.selected_checklist_id and not st.session_state.current_checklist_questions: tooltip_parts.append("Selected checklist has no questions.")
    if st.session_state.selected_checklist_id and not st.session_state.current_checklist_prompt: tooltip_parts.append("Selected checklist has no prompt template.")
    if st.session_state.selected_checklist_id and st.session_state.current_checklist_prompt and PROMPT_PLACEHOLDER not in st.session_state.current_checklist_prompt: tooltip_parts.append(f"Prompt template missing '{PROMPT_PLACEHOLDER}'.")

    analyse_button_tooltip = " ".join(tooltip_parts) if tooltip_parts else "Start analyzing the document using the selected checklist"

    if st.sidebar.button("✨ Analyse Document", key="analyse_button", disabled=not can_analyse, help=analyse_button_tooltip, use_container_width=True, type="primary"):
        # --- Start Analysis Process ---
        st.session_state.processing_in_progress = True
        st.session_state.analysis_complete = False
        st.session_state.analysis_results = [] # Clear previous results
        st.session_state.run_key += 1 # Increment run key for this analysis run
        st.session_state.run_status_summary = []
        st.session_state.excel_data = None
        st.session_state.search_trigger = None
        st.session_state.last_search_result = None
        st.session_state.show_wording_states = defaultdict(bool) # Reset toggles

        current_api_key = st.session_state.api_key
        # Get AI parameters from session state
        current_gen_config_params = {
            'temperature': st.session_state.ai_temperature,
            'top_p': st.session_state.ai_top_p,
            'top_k': st.session_state.ai_top_k
        }
        # Get selected checklist data
        checklist_prompt = st.session_state.current_checklist_prompt
        checklist_questions = st.session_state.current_checklist_questions
        checklist_id = st.session_state.selected_checklist_id
        checklist_name = st.session_state.selected_checklist_name

        run_start_time = datetime.now()
        run_timestamp_str = run_start_time.strftime("%Y-%m-%d %H:%M:%S")

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

        status_container = st.container()
        progress_bar = status_container.progress(0, text="Initializing analysis...")
        status_text = status_container.empty()

        temp_dir = "temp_uploads"
        safe_base_name = re.sub(r'[^\w\-.]', '_', base_file_name)
        temp_file_path = os.path.join(APP_DIR, temp_dir, f"{st.session_state.run_key}_{safe_base_name}")
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
                    gemini_uploaded_file_ref = genai.upload_file(path=temp_file_path)
                    st.toast(f"File '{gemini_uploaded_file_ref.display_name}' uploaded to cloud.", icon="☁️"); break
                except Exception as upload_err:
                    err_str = str(upload_err).lower()
                    if "api key" in err_str or "authenticat" in err_str or "permission" in err_str:
                        st.error(f"❌ File upload failed due to API key/permission issue: {upload_err}")
                        st.error("Please verify the API key has File API permissions enabled.")
                        raise
                    elif upload_attempt < 2:
                        st.warning(f"Upload attempt {upload_attempt+1} failed: {upload_err}. Retrying...")
                        time.sleep(2 + upload_attempt)
                    else:
                        st.error(f"Upload failed after multiple attempts: {upload_err}")
                        raise
            if not gemini_uploaded_file_ref: raise Exception("Failed to upload file to Google Cloud AI after retries.")
            progress_bar.progress(15, text="File uploaded. Starting analysis...")

            # --- Run Analysis for the whole checklist ---
            analysis_data, analysis_status, analysis_warnings = generate_checklist_analysis(
                checklist_prompt, checklist_questions, gemini_uploaded_file_ref, status_text, current_api_key, current_gen_config_params
            )

            overall_status = analysis_status
            run_warnings.extend(analysis_warnings)
            st.session_state.run_status_summary.append({"checklist": checklist_name, "status": analysis_status, "warnings": analysis_warnings})


            # --- Finalize Analysis ---
            if analysis_status in ["Success", "Partial Success"] and analysis_data:
                all_validated_data = analysis_data
                # Add common fields
                for item in all_validated_data:
                     item["File Name"] = base_file_name
                     item["Generation Time"] = run_timestamp_str
                     item["Checklist Name"] = checklist_name

                st.session_state.analysis_results = all_validated_data
                progress_bar.progress(100, text="Analysis process finished!")
                if analysis_status == "Success":
                    status_text.success("🏁 Analysis finished successfully!")
                else: # Partial Success
                     status_text.warning("🏁 Analysis finished, but some questions might be missing. Check results/summary.")
            else:
                 status_text.error("🏁 Analysis finished, but failed to generate valid results.")
                 st.session_state.analysis_results = [] # Ensure it's an empty list

            st.session_state.analysis_complete = True

            # --- Save to GCS and Firestore ---
            if st.session_state.analysis_results and st.session_state.pdf_bytes: # Save even partial results
                try:
                    timestamp = datetime.now()
                    firestore_doc_id = f"{safe_base_name}_{timestamp.isoformat()}"
                    gcs_blob_name = f"{GCS_PDF_FOLDER}/{firestore_doc_id}.pdf"
                    gcs_file_path = upload_to_gcs(GCS_BUCKET_NAME, st.session_state.pdf_bytes, gcs_blob_name, status_text)
                    status_text.info("💾 Saving results and PDF reference to database...")
                    doc_ref = db.collection("analysis_runs").document(firestore_doc_id)
                    doc_ref.set({
                        "filename": base_file_name,
                        "analysis_timestamp": timestamp,
                        "results": st.session_state.analysis_results, # Save the potentially partial results
                        "run_status": st.session_state.run_status_summary,
                        "gcs_pdf_path": gcs_file_path,
                        "checklist_name": checklist_name, # Store checklist name used
                        "checklist_id": checklist_id # Store checklist ID used
                    })
                    status_text.success("💾 Results and PDF link saved successfully.")
                    time.sleep(1)
                except Exception as db_gcs_err:
                    st.error(f"❌ Failed to save results/PDF to cloud: {db_gcs_err}")
                    print(f"DB/GCS Save Error: {db_gcs_err}\n{traceback.format_exc()}")
                    st.session_state.run_status_summary.append({
                        "checklist": "Cloud Save", "status": "Failed",
                        "warnings": [f"Error saving to GCS/Firestore: {db_gcs_err}"]})
                    overall_status = "Failed" # Mark overall as failed if save fails

        except Exception as main_err:
            st.error(f"❌ CRITICAL ERROR during analysis workflow: {main_err}"); print(traceback.format_exc())
            overall_status = "Failed"; st.session_state.analysis_complete = False
            st.session_state.run_status_summary.append({"checklist": "Overall Process Control", "status": "Critical Error", "warnings": [str(main_err), "Analysis halted. See logs."]})
            status_text.error(f"Analysis stopped due to critical error: {main_err}")
        finally:
            # --- Cleanup ---
            st.session_state.processing_in_progress = False
            time.sleep(4); status_text.empty(); progress_bar.empty()
            if gemini_uploaded_file_ref and hasattr(gemini_uploaded_file_ref, 'name'):
                try: status_text.info(f"☁️ Deleting temporary Gemini cloud file: {gemini_uploaded_file_ref.name}..."); genai.delete_file(name=gemini_uploaded_file_ref.name); st.toast("Gemini cloud file deleted.", icon="🗑️"); time.sleep(1); status_text.empty()
                except Exception as del_err: st.sidebar.warning(f"Gemini cloud cleanup issue: {del_err}", icon="⚠️"); status_text.warning(f"Could not delete Gemini cloud file: {del_err}"); print(f"WARN: Failed to delete cloud file {gemini_uploaded_file_ref.name}: {del_err}")
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except Exception as local_del_err: st.sidebar.warning(f"Local temp file cleanup issue: {local_del_err}", icon="⚠️"); print(f"WARN: Failed to delete local temp file {temp_file_path}: {local_del_err}")

        st.rerun() # Rerun to reflect final state

    # else for history mode
    elif st.session_state.viewing_history:
        # Minimal sidebar when viewing history
        st.sidebar.info("📜 Viewing historical data.")
        st.sidebar.markdown("---")
        st.sidebar.markdown("Click 'Exit History View' above to start a new analysis.")


# --- 7. Display Area (Results and PDF Viewer) ---
if st.session_state.pdf_bytes is not None:
    col1, col2 = st.columns([6, 4]) # Results | PDF Viewer

    # --- Column 1: Analysis Results ---
    with col1:
        # --- Run Status Summary ---
        if st.session_state.run_status_summary:
            final_status_icon = "❓"
            summary_text = "Analysis Summary"
            # Since there's only one "run" entry now (for the whole checklist)
            if st.session_state.run_status_summary:
                run_info = st.session_state.run_status_summary[0]
                status = run_info.get("status", "Unknown")
                checklist_name_sum = run_info.get("checklist", "N/A")
                warnings_sum = run_info.get("warnings", [])

                if status == "Success": final_status_icon = "✅"
                elif status == "Partial Success": final_status_icon = "⚠️"
                elif status == "Failed": final_status_icon = "❌"
                elif status == "Skipped": final_status_icon = "➡️" # Should not happen now but keep
                summary_text = f"{final_status_icon} Checklist '{checklist_name_sum}': {status}"

                expand_summary = st.session_state.viewing_history or (final_status_icon != "✅")
                with st.expander(summary_text, expanded=expand_summary):
                    # Display warnings/details
                    if warnings_sum:
                        # Filter out generic validation header message if present
                        filtered_warnings = [msg for msg in warnings_sum if not (isinstance(msg, str) and msg.startswith("Validation Issues Found"))]
                        if filtered_warnings:
                            st.caption("Details / Issues Encountered:")
                            for msg in filtered_warnings:
                                msg_str = str(msg)
                                if any(term in msg_str for term in ["CRITICAL", "Error", "Block", "Fail"]) and "permission" not in msg_str.lower(): st.error(f" L> {msg_str}")
                                elif any(term in msg_str.lower() for term in ["warn", "missing", "unexpected", "empty list", "validation issue", "permission issue", "recitation", "max_tokens", "timeout", "partial"]): st.warning(f" L> {msg_str}")
                                elif "Skipped" in status: st.info(f" L> {msg_str}")
                                else: st.caption(f" L> {msg_str}")
                        elif not filtered_warnings and warnings_sum: # Only generic header was present
                            st.caption("Validation issues found (structure or content mismatch).")

                    else:
                        st.caption("No specific issues reported for this run.")
            else:
                 st.expander("📊 Analysis Run Summary", expanded=False).info("No summary data available.")


        st.subheader("Analysis Results")

        if (st.session_state.analysis_complete or st.session_state.viewing_history) and st.session_state.analysis_results:
            try:
                results_list = sorted(st.session_state.analysis_results, key=lambda x: x.get('Question Number', float('inf')))
            except Exception as sort_err:
                 st.warning(f"Could not sort results by question number: {sort_err}. Displaying in original order.")
                 results_list = st.session_state.analysis_results

            # --- Scatter Plot Expander (Keep as is) ---
            try:
                plot_data = []
                for item in results_list:
                    plot_data.append({
                        'Question Number': item.get('Question Number', 0),
                        'Number of Evidence Items': len(item.get('Evidence', [])),
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
                        fig.update_layout(xaxis_title="Question Number", yaxis_title="Number of Evidence Clauses")
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("This plot shows how many separate evidence clauses the AI referenced for each question. Hover over points for question details.")
            except Exception as plot_err:
                 st.warning(f"Could not generate scatter plot: {plot_err}")
                 print(f"Plotting Error: {plot_err}\n{traceback.format_exc()}")


            # --- Tabbed Results Display (Keep structure, works with dynamic categories) ---
            grouped_results = defaultdict(list); categories_ordered = []
            for item in results_list:
                category = item.get("Question Category", "Uncategorized")
                if category not in grouped_results: categories_ordered.append(category)
                grouped_results[category].append(item)

            if categories_ordered:
                category_tabs = st.tabs(categories_ordered)
                for i, category in enumerate(categories_ordered):
                    with category_tabs[i]:
                        category_items = grouped_results[category]
                        for index, result_item in enumerate(category_items):
                            q_num = result_item.get('Question Number', 'N/A'); question_text = result_item.get('Question', 'N/A')
                            # --- UI STRUCTURE within Expander (Keep as is) ---
                            with st.expander(f"**Q{q_num}:** {question_text}"):
                                st.markdown(f"**Answer:**")
                                st.markdown(f"> {result_item.get('Answer', 'N/A')}") # Using blockquote
                                st.markdown("**Answer Justification:**")
                                justification_text = result_item.get('Answer Justification', '')
                                just_key = f"justification_{category}_{q_num}_{index}"
                                st.text_area("Justification Text", value=justification_text, height=100, disabled=True, label_visibility="collapsed", key=just_key)
                                st.markdown("---") # Separator before Evidence

                                evidence_list = result_item.get('Evidence', [])
                                if evidence_list:
                                    st.markdown("**Evidence:**")
                                    with st.container():
                                        for ev_index, evidence_item in enumerate(evidence_list):
                                            clause_ref = evidence_item.get('Clause Reference', 'N/A')
                                            search_text = evidence_item.get('Searchable Clause Text', None)
                                            clause_wording = evidence_item.get('Clause Wording', 'N/A')
                                            base_key = f"ev_{category}_{q_num}_{index}_{ev_index}" # Unique base for keys

                                            ev_cols = st.columns([3, 1])
                                            with ev_cols[0]:
                                                if search_text:
                                                    button_key = f"search_btn_{base_key}"
                                                    button_label = f"Clause: **{clause_ref or 'Link'}** (Find & View)"
                                                    if st.button(button_label, key=button_key, help=f"Search for text related to '{clause_ref or 'this evidence'}' and view the page."):
                                                        st.session_state.search_trigger = {'text': search_text, 'ref': clause_ref}
                                                        st.session_state.last_search_result = None
                                                        st.rerun()
                                                elif clause_ref != 'N/A':
                                                    st.markdown(f"- Clause: **{clause_ref}** (No searchable text)")
                                                else:
                                                    st.caption("No clause reference provided.")

                                            with ev_cols[1]:
                                                if clause_wording != 'N/A':
                                                    toggle_key = f"toggle_wording_{base_key}"
                                                    show_wording = st.toggle("Show Wording", key=toggle_key, value=st.session_state.show_wording_states.get(toggle_key, False), help="Show/hide exact clause wording.")
                                                    if show_wording != st.session_state.show_wording_states.get(toggle_key, False):
                                                        st.session_state.show_wording_states[toggle_key] = show_wording
                                                        st.rerun() # Rerun needed to show/hide

                                            if st.session_state.show_wording_states.get(f"toggle_wording_{base_key}", False):
                                                st.text_area(f"AI Extracted Wording for '{clause_ref}':", value=clause_wording, height=100, disabled=True, key=f"wording_area_{base_key}")

                                            if ev_index < len(evidence_list) - 1:
                                                st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
                                else:
                                    st.markdown("**Evidence:** None provided.")
                            # --- END Expander UI ---
            else: st.warning("Analysis generated results, but they could not be grouped by category. Displaying raw list."); st.json(results_list)

            # --- Excel Download ---
            st.sidebar.markdown("---"); st.sidebar.markdown("## Export Results")
            if st.sidebar.button("Prepare Data for Excel Download", key="prep_excel", use_container_width=True):
                st.session_state.excel_data = None; excel_prep_status = st.sidebar.empty(); excel_prep_status.info("Preparing Excel data...")
                try:
                    excel_rows = [];
                    for item in results_list:
                        references = []; first_search_text = "N/A"; evidence = item.get("Evidence")
                        file_name_for_excel = st.session_state.history_filename if st.session_state.viewing_history else st.session_state.get("current_filename", "N/A")
                        gen_time_for_excel = st.session_state.history_timestamp if st.session_state.viewing_history else item.get("Generation Time", "N/A")
                        # Get checklist name - prefer history if viewing, else from current item
                        checklist_name_for_excel = st.session_state.history_checklist_name if st.session_state.viewing_history else item.get("Checklist Name", "N/A")


                        if evidence and isinstance(evidence, list):
                            for i, ev in enumerate(evidence):
                                if isinstance(ev, dict): references.append(str(ev.get("Clause Reference", "N/A")));
                                if i == 0 and isinstance(ev, dict): first_search_text = ev.get("Searchable Clause Text", "N/A")

                        excel_row = {
                            "File Name": file_name_for_excel, "Generation Time": gen_time_for_excel,
                            "Checklist Name": checklist_name_for_excel, # Added Checklist Name
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
                        st.session_state.excel_data = output.getvalue(); excel_prep_status.success("✅ Excel file ready for download!"); time.sleep(2); excel_prep_status.empty()
                except Exception as excel_err: excel_prep_status.error(f"Excel Prep Error: {excel_err}"); print(traceback.format_exc())

            if st.session_state.excel_data:
                current_filename_for_download = st.session_state.history_filename if st.session_state.viewing_history else st.session_state.get("current_filename", "analysis_results")
                safe_base_name = re.sub(r'[^\w\s-]', '', os.path.splitext(current_filename_for_download)[0]).strip().replace(' ', '_'); download_filename = f"Analysis_{safe_base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                st.sidebar.download_button(label="📥 Download Results as Excel", data=st.session_state.excel_data, file_name=download_filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_excel_final", use_container_width=True)

        elif st.session_state.analysis_complete and not st.session_state.analysis_results: st.info("Analysis process completed, but no valid results were generated. Check the run summary above for potential issues.")
        elif st.session_state.processing_in_progress: st.info("Analysis is currently in progress...")
        elif not st.session_state.analysis_complete and st.session_state.pdf_bytes is not None and not st.session_state.viewing_history: st.info("PDF loaded. Select a checklist and click 'Analyse Document' in the sidebar.")


    # --- Column 2: PDF Viewer (Sticky - Keep structure as is) ---
    with col2:
        st.markdown('<div class="sticky-viewer-content">', unsafe_allow_html=True) # Sticky wrapper start
        st.subheader("📄 PDF Viewer"); viewer_status_placeholder = st.empty()

        if st.session_state.search_trigger:
            search_info = st.session_state.search_trigger; st.session_state.search_trigger = None # Clear trigger
            with st.spinner(f"🔎 Searching for text related to: '{search_info['ref']}'..."):
                found_page, instances, term_used, search_status, all_findings = find_text_in_pdf(st.session_state.pdf_bytes, search_info['text'])
            if found_page:
                st.session_state.last_search_result = {'page': found_page, 'instances': instances, 'term': term_used, 'status': search_status, 'ref': search_info['ref'], 'all_findings': all_findings}
                st.session_state.current_page = found_page; viewer_status_placeholder.empty(); st.rerun()
            else: st.session_state.last_search_result = None; viewer_status_placeholder.error(search_status)

        if st.session_state.pdf_display_ready:
            try:
                with fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf") as doc: total_pages = doc.page_count
            except Exception as pdf_load_err: st.error(f"Error loading PDF for page count: {pdf_load_err}"); total_pages = 1; st.session_state.current_page = 1

            current_display_page = max(1, min(int(st.session_state.get('current_page', 1)), total_pages))
            if current_display_page != st.session_state.get('current_page'):
                 st.session_state.current_page = current_display_page

            nav_cols = st.columns([1, 3, 1])
            with nav_cols[0]:
                if st.button("⬅️ Prev", key="prev_page", disabled=(current_display_page <= 1), use_container_width=True): st.session_state.current_page -= 1; st.session_state.last_search_result = None; st.rerun()
            with nav_cols[2]:
                if st.button("Next ➡️", key="next_page", disabled=(current_display_page >= total_pages), use_container_width=True): st.session_state.current_page += 1; st.session_state.last_search_result = None; st.rerun()

            page_info_text = f"Page {current_display_page} of {total_pages}"; search_context_ref = None
            if st.session_state.last_search_result and st.session_state.last_search_result['page'] == current_display_page:
                search_context_ref = st.session_state.last_search_result.get('ref', 'Search')
                if st.session_state.last_search_result.get('all_findings'): page_info_text += f" (🎯 Multi-match: '{search_context_ref}')"
                else: page_info_text += f" (🎯 Ref: '{search_context_ref}')"

            nav_cols[1].markdown(f"<div style='text-align: center; padding-top: 0.5rem;'>{page_info_text}</div>", unsafe_allow_html=True)

            if st.session_state.last_search_result and st.session_state.last_search_result.get('all_findings'):
                multi_findings = st.session_state.last_search_result['all_findings']; found_pages = sorted([f[0] for f in multi_findings])
                status_msg = st.session_state.last_search_result.get('status', '');
                if status_msg: viewer_status_placeholder.info(status_msg)
                st.write("Jump to other matches for this reference:")
                num_buttons = len(found_pages); btn_cols = st.columns(min(num_buttons, 5))
                current_search_ref = st.session_state.last_search_result.get('ref', 'unknown')

                for idx, p_num in enumerate(found_pages):
                    col_idx = idx % len(btn_cols); is_current = (p_num == current_display_page)
                    jump_button_key = f"jump_{p_num}_{current_search_ref}"
                    if btn_cols[col_idx].button(f"Page {p_num}", key=jump_button_key, disabled=is_current, use_container_width=True):
                        st.session_state.current_page = p_num; new_instances = next((inst for pg, inst in multi_findings if pg == p_num), None)
                        st.session_state.last_search_result['instances'] = new_instances
                        st.session_state.last_search_result['page'] = p_num
                        term_desc = st.session_state.last_search_result.get('term', 'text')
                        st.session_state.last_search_result['status'] = f"✅ Viewing match for '{term_desc}' on page {p_num}."
                        st.session_state.last_search_result['all_findings'] = None # Clear after jump
                        st.rerun()

            st.markdown("---")
            highlights_to_apply = None; render_status_override = None
            if st.session_state.last_search_result and st.session_state.last_search_result.get('page') == current_display_page:
                highlights_to_apply = st.session_state.last_search_result.get('instances')
                if not st.session_state.last_search_result.get('all_findings'):
                    render_status_override = st.session_state.last_search_result.get('status')

            image_bytes, render_status = render_pdf_page_to_image(st.session_state.pdf_bytes, current_display_page, highlight_instances=highlights_to_apply, dpi=150)

            if image_bytes:
                st.image(image_bytes, caption=f"Page {current_display_page} - View", use_container_width=True)
                final_status = render_status_override if render_status_override else render_status
                if not (st.session_state.last_search_result and st.session_state.last_search_result.get('all_findings')):
                    if final_status:
                        if "✅" in final_status or "✨" in final_status or "Found" in final_status : viewer_status_placeholder.success(final_status)
                        elif "⚠️" in final_status or "warning" in final_status.lower() or "multiple pages" in final_status.lower(): viewer_status_placeholder.warning(final_status)
                        elif "❌" in final_status or "error" in final_status.lower(): viewer_status_placeholder.error(final_status)
                        else: viewer_status_placeholder.caption(final_status)
                    else: viewer_status_placeholder.empty()
            else:
                 viewer_status_placeholder.error(f"Failed to render page {current_display_page}. {render_status or ''}")
        else:
            st.info("PDF loaded, preparing viewer..."); viewer_status_placeholder.empty()
        st.markdown('</div>', unsafe_allow_html=True) # Sticky wrapper end

# --- Fallback message if no PDF loaded (and not viewing history) ---
elif not st.session_state.pdf_bytes and not st.session_state.viewing_history:
     st.info("⬆️ Select/Create a checklist, then upload a PDF file using the sidebar to begin, or load an analysis from the History page.")