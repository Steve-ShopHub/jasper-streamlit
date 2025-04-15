# app.py
# --- COMPLETE FILE ---

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

# --- 1. SET PAGE CONFIG (MUST BE FIRST st COMMAND) ---
st.set_page_config(
    layout="wide",
    page_title="JASPER - Agreement Analyzer",
    page_icon="📄" # Optional: Set an emoji icon
)

# --- Inject custom CSS for sticky column (AFTER set_page_config) ---
# This targets the container Streamlit usually wraps column content in.
st.markdown("""
<style>
    /* Define a class for the sticky container */
    .sticky-viewer-content {
        position: sticky;
        top: 55px; /* Adjust vertical offset from top (consider Streamlit header) */
        z-index: 101; /* Ensure it's above other elements */
        /* background-color: white; */ /* Inherit from theme */
        padding-bottom: 1rem; /* Add some space at the bottom */
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Configuration & Setup ---
API_KEY_ENV_VAR = "GEMINI_API_KEY"
MODEL_NAME = "gemini-2.5-pro-preview-03-25"
MAX_VALIDATION_RETRIES = 1
RETRY_DELAY_SECONDS = 3
PROMPT_FILE = "prompt.txt"
LOGO_FILE = "jasper-logo-1.png"

# Get the absolute path to the directory containing app.py
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load Prompt Text from File ---
try:
    prompt_path = os.path.join(APP_DIR, PROMPT_FILE)
    with open(prompt_path, 'r', encoding='utf-8') as f:
        full_prompt_text = f.read()
except FileNotFoundError:
    st.error(f"Error: Prompt file '{PROMPT_FILE}' not found in the application directory ({APP_DIR}).")
    st.stop()
except Exception as e:
    st.error(f"Error reading prompt file '{PROMPT_FILE}': {e}")
    st.stop()

# --- Logo (in Sidebar) ---
try:
    logo_path = os.path.join(APP_DIR, LOGO_FILE)
    st.sidebar.image(logo_path, width=150)
    st.sidebar.markdown("---")
except FileNotFoundError:
    st.sidebar.warning(f"Logo '{LOGO_FILE}' not found.")
except Exception as e:
    st.sidebar.error(f"Error loading logo: {e}")

# --- Schema Definition (Version 2 - Supporting Multiple Evidence Items) ---
ai_response_schema_dict = {
  "type": "array",
  "description": "A list of questions, answers, justifications, and supporting evidence derived from the facility agreement analysis.",
  "items": {
    "type": "object",
    "description": "Represents a single question's analysis, including potentially multiple pieces of evidence.",
    "properties": {
      "Question Number": {"type": "integer"},
      "Question Category": {"type": "string"},
      "Question": {"type": "string"},
      "Answer": {"type": "string"},
      "Answer Justification": {
          "type": "string",
          "description": "Overall justification for the answer, considering *all* supporting evidence found."
      },
      "Evidence": {
        "type": "array",
        "description": "A list of evidence items supporting the answer. Each item links a clause to its wording and the specific page number where it was found. This array should be empty if the Answer is 'Information Not Found' or 'N/A'.",
        "items": {
          "type": "object",
          "properties": {
            "Clause Reference": {
                "type": "string",
                "description": "Specific clause number(s) or section(s) (e.g., 'Clause 14.1(a)', 'Section 5')."
            },
            "Clause Wording": {
                "type": "string",
                "description": "Exact, full text of the referenced clause(s) or relevant sentence(s)."
            },
            "Page Number": {
                "type": "integer",
                "description": "The physical page number in the document where this clause reference/wording is primarily located (PDF viewer page 1 = first page). Use 0 if not applicable/determinable."
            }
          },
          "required": ["Clause Reference", "Clause Wording", "Page Number"]
        }
      }
    },
    "required": [
      "Question Number", "Question Category", "Question", "Answer", "Answer Justification", "Evidence"
    ]
  }
}

# Keys required for validation
AI_REQUIRED_KEYS = set(ai_response_schema_dict['items']['required'])
AI_EVIDENCE_REQUIRED_KEYS = set(ai_response_schema_dict['items']['properties']['Evidence']['items']['required'])

# --- Excel Column Order ---
EXCEL_COLUMN_ORDER = [
    "File Name", "Generation Time", "Question Number", "Question Category",
    "Question", "Answer", "Answer Justification",
    "Clause References (Concatenated)", "Page Numbers (Concatenated)",
    "First Clause Wording Found"
]

# --- Section Definitions ---
SECTIONS_TO_RUN = {
    "agreement_details": (1, 4),
    "eligibility": (5, 36),
    "confidentiality": (37, 63),
    "additional_borrowers": (64, 66),
    "interest_rate_provisions": (67, 71),
    "prepayment_interest": (72, 78)
}

# --- System Instruction ---
system_instruction_text = """You are analysing a facility agreement to understand whether the asset can be included within a Significant Risk Transfer or not (or with conditions, requirements, or exceptions) at NatWest. Your output must be precise, factual, and directly supported by evidence from the provided document(s). You must answer with UK spelling, not US. (e.g. 'analyse' is correct while 'analyze' is not). Adhere strictly to the JSON schema provided, ensuring every object in the output array contains all required keys."""


# --- 3. Helper Function Definitions ---

def filter_prompt_by_section(initial_full_prompt, section):
    """Filters the main prompt to include only questions for a specific section."""
    if section not in SECTIONS_TO_RUN:
        raise ValueError(f"Invalid section specified: {section}. Must be one of {list(SECTIONS_TO_RUN.keys())}")
    start_q, end_q = SECTIONS_TO_RUN[section]
    questions_start_marker = "**Questions to Answer:**"
    questions_end_marker = "**Final Instruction:**"
    try:
        start_index = initial_full_prompt.index(questions_start_marker)
        end_index = initial_full_prompt.index(questions_end_marker)
        prompt_header = initial_full_prompt[:start_index]
        full_questions_block = initial_full_prompt[start_index + len(questions_start_marker):end_index].strip()
        prompt_footer = initial_full_prompt[end_index:]
    except ValueError:
        st.error("Could not find question block markers ('**Questions to Answer:**' or '**Final Instruction:**') in the main prompt text. Check prompt.txt.")
        raise ValueError("Could not find prompt markers.")

    # Split questions based on the numbered format
    question_entries = re.split(r'\n(?=\s*\d+\.\s*?\*\*Question Category:)', full_questions_block)
    # Add fallback split patterns if the primary one fails or doesn't match all entries
    if not question_entries or len(question_entries) < 2:
        question_entries = re.split(r'\n(?=\d+\.\s)', full_questions_block)
        if not question_entries or len(question_entries) < 2:
             question_entries = re.split(r'\n(?=\*\*Question Category:)', full_questions_block) # Least specific

    filtered_question_texts = []
    processed_q_nums = set()

    for entry in question_entries:
        entry = entry.strip()
        if not entry: continue
        # Match digits at the very beginning, possibly after whitespace, followed by a dot.
        match = re.match(r'^\s*(\d+)\.', entry)
        if match:
            q_num = int(match.group(1))
            if start_q <= q_num <= end_q:
                filtered_question_texts.append(entry)
                processed_q_nums.add(q_num)
        # else: # Optionally warn about entries that look like questions but weren't parsed
            # if "**Question Category:**" in entry or "**Question:**" in entry:
            #      try: st.warning(f"Could not parse question number from entry starting: {entry[:100]}...")
            #      except Exception: pass # Avoid errors if st called too early

    # Check if expected questions are missing after parsing
    expected_q_nums = set(range(start_q, end_q + 1))
    missing_q_nums = expected_q_nums - processed_q_nums
    if missing_q_nums:
         try: st.warning(f"Parsing might have missed expected question numbers in range {start_q}-{end_q} for section '{section}': {sorted(list(missing_q_nums))}")
         except Exception: pass

    if not filtered_question_texts:
        # Use st.error for critical failures that should stop processing for this section
        st.error(f"No questions found for section '{section}' in range {start_q}-{end_q}. Check prompt formatting and split logic in filter_prompt_by_section.")
        raise ValueError(f"Failed to extract questions for section '{section}'.")

    filtered_questions_string = "\n\n".join(filtered_question_texts) # Use double newline for better separation

    # Add section focus note to the prompt header
    task_end_marker = "specified section."
    insert_pos = prompt_header.find(task_end_marker)
    if insert_pos != -1:
         insert_pos += len(task_end_marker)
         section_note = f"\n\n**Current Focus:** Answering ONLY questions for the '{section.upper()}' section (Questions {start_q}-{end_q}). The list below contains ONLY these questions.\n"
         final_header = prompt_header[:insert_pos] + section_note + prompt_header[insert_pos:]
    else: # Fallback if marker not found
         final_header = prompt_header + f"\n\n**Current Focus:** Answering ONLY questions for the '{section.upper()}' section (Questions {start_q}-{end_q}).\n"

    final_prompt_for_api = f"{final_header}{questions_start_marker}\n\n{filtered_questions_string}\n\n{prompt_footer}"
    return final_prompt_for_api


def validate_ai_data(data, section_name):
    """Validates the structure of the AI response against the schema. Returns validated data (list) and a list of warning messages."""
    if not isinstance(data, list):
        st.error(f"Critical Validation Error: AI response for section '{section_name}' is not a list.")
        return None, [f"CRITICAL: Response for '{section_name}' was not a list."]

    validated_data = []
    issues_list = [] # Collect all validation issues

    for index, item in enumerate(data):
        q_num_str = f"Q#{item.get('Question Number', f'Item Index {index}')}" # Identify item
        is_outer_valid = True # Assume valid initially

        if not isinstance(item, dict):
            issues_list.append(f"{q_num_str}: Entry is not a dictionary.")
            continue # Skip to next item if not a dict

        # Check outer keys
        item_keys = set(item.keys())
        missing_outer_keys = AI_REQUIRED_KEYS - item_keys
        if missing_outer_keys:
            issues_list.append(f"{q_num_str}: Missing required keys: {sorted(list(missing_outer_keys))}")
            is_outer_valid = False

        # Check Evidence structure
        evidence_list = item.get("Evidence")
        if not isinstance(evidence_list, list):
             issues_list.append(f"{q_num_str}: 'Evidence' field is missing or not an array.")
             is_outer_valid = False
        else:
            # Validate items within the Evidence array
            for ev_index, evidence_item in enumerate(evidence_list):
                is_evidence_item_valid = True
                if not isinstance(evidence_item, dict):
                    issues_list.append(f"{q_num_str}, Evidence[{ev_index}]: Item is not a dictionary.")
                    is_outer_valid = False; continue # Invalidate outer, skip rest of this evidence item

                evidence_keys = set(evidence_item.keys())
                missing_evidence_keys = AI_EVIDENCE_REQUIRED_KEYS - evidence_keys
                if missing_evidence_keys:
                    issues_list.append(f"{q_num_str}, Evidence[{ev_index}]: Missing keys: {sorted(list(missing_evidence_keys))}")
                    is_outer_valid = False; is_evidence_item_valid = False

                # Check specific types within evidence
                page_num = evidence_item.get("Page Number")
                if not isinstance(page_num, int):
                    issues_list.append(f"{q_num_str}, Evidence[{ev_index}]: 'Page Number' is not an integer (found {type(page_num).__name__}).")
                    is_outer_valid = False; is_evidence_item_valid = False

                if not isinstance(evidence_item.get("Clause Reference"), str):
                    issues_list.append(f"{q_num_str}, Evidence[{ev_index}]: 'Clause Reference' is not a string.")
                    is_outer_valid = False; is_evidence_item_valid = False

                if not isinstance(evidence_item.get("Clause Wording"), str):
                    issues_list.append(f"{q_num_str}, Evidence[{ev_index}]: 'Clause Wording' is not a string.")
                    is_outer_valid = False; is_evidence_item_valid = False


        if is_outer_valid:
            validated_data.append(item)

    # Prepend summary header if there were issues
    if issues_list:
        issues_list.insert(0, f"Validation Issues Found [Section: {section_name}] ({len(validated_data)} items passed):")
        # Display immediate warning in the main app area during development/run
        st.warning("Validation Issues Detected:\n" + "\n".join([f"- {issue}" for issue in issues_list[1:]]))

    return validated_data, issues_list # Return validated items and list of all issue strings


def generate_section_analysis(section, uploaded_file_ref, status_placeholder):
    """Generates analysis for a section, handles retries, validation. Returns (data, status, warnings)."""
    status_placeholder.info(f"🔄 Starting section: {section}...")
    section_warnings = []
    model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction_text)
    generation_config = types.GenerationConfig(
        response_mime_type="application/json",
        response_schema=ai_response_schema_dict, # Use the correct, updated schema
        temperature=0.0, top_p=0.05, top_k=1
    )
    final_validated_data = None

    for attempt in range(1, 1 + MAX_VALIDATION_RETRIES + 1):
        if attempt > 1:
             status_placeholder.info(f"⏳ Retrying validation for '{section}' (Attempt {attempt})...")
             time.sleep(RETRY_DELAY_SECONDS)
        try:
            prompt_for_api = filter_prompt_by_section(full_prompt_text, section)
            contents = [uploaded_file_ref, prompt_for_api]

            status_placeholder.info(f"🧠 Calling AI for section: {section} (Attempt {attempt})...")
            response = model.generate_content(contents=contents, generation_config=generation_config)
            parsed_ai_data = None; validated_ai_data = None; validation_issues = []

            status_placeholder.info(f"🔍 Processing AI response for: {section}...")
            if response.parts:
                full_response_text = response.text
                try:
                    parsed_ai_data = json.loads(full_response_text)
                    status_placeholder.info(f"✔️ Validating response structure for: {section}...")
                    validated_ai_data, validation_issues = validate_ai_data(parsed_ai_data, section) # Use updated validation
                    section_warnings.extend(validation_issues) # Collect any validation messages

                    if validated_ai_data is not None and len(validated_ai_data) > 0:
                        # Validation passed and yielded data
                        final_validated_data = validated_ai_data
                        break # Success, exit retry loop
                    elif validated_ai_data is None: # Critical validation error (e.g., not a list)
                         error_msg = f"CRITICAL Validation Error for section '{section}'. Response format unusable."
                         # st.error(error_msg) # Error shown by validate_ai_data
                         section_warnings.append(error_msg) # Add to summary
                         break # Don't retry critical errors
                    else: # Validation ran but returned 0 items (e.g., all failed structure check)
                        status_placeholder.warning(f"⚠️ Validation found 0 valid items for {section} (Attempt {attempt}).")
                        # Loop continues if retries remain

                except json.JSONDecodeError as json_err:
                    error_msg = f"JSON Parsing Error (Attempt {attempt}, Section '{section}'): {json_err}"
                    st.error(error_msg) # Show immediate error
                    section_warnings.append(error_msg) # Add to summary
                    # Loop continues if retries remain

            else: # Handle empty response / blocking
                block_reason = "Unknown"; block_message = "N/A"; safety_ratings_str = "N/A"
                try: # Safe access to feedback attributes
                   if response.prompt_feedback:
                     if response.prompt_feedback.block_reason: block_reason = response.prompt_feedback.block_reason.name
                     block_message = response.prompt_feedback.block_reason_message or "N/A"
                     if response.prompt_feedback.safety_ratings:
                          ratings = [f"{sr.category.name}: {sr.probability.name}" for sr in response.prompt_feedback.safety_ratings]
                          safety_ratings_str = ", ".join(ratings)
                except AttributeError: pass # Ignore if attributes don't exist
                warn_msg = f"API returned empty response for {section} (Attempt {attempt}). Reason: {block_reason}. Safety: {safety_ratings_str}"
                st.warning(warn_msg) # Show immediate warning
                section_warnings.append(warn_msg) # Add to summary
                # Loop continues if retries remain

        # --- Handle specific API/Generation Exceptions ---
        except types.generation_types.BlockedPromptException as bpe:
             error_msg = f"API Error (Attempt {attempt}): Prompt blocked for section '{section}'. Reason: {bpe}"
             st.error(error_msg); section_warnings.append(error_msg)
        except types.generation_types.StopCandidateException as sce:
             error_msg = f"API Error (Attempt {attempt}): Generation stopped unexpectedly for section '{section}'. Reason: {sce}"
             st.error(error_msg); section_warnings.append(error_msg)
        except ValueError as ve: # Catch prompt filtering errors
             error_msg = f"Prompt Error (Attempt {attempt}, Section '{section}'): {ve}"; st.error(error_msg); section_warnings.append(error_msg)
             break # Stop processing this section if prompt filtering failed
        except Exception as e: # Catch any other errors during processing
            error_msg = f"Unexpected Error (Attempt {attempt}, Section '{section}'): {type(e).__name__} - {e}"
            st.error(error_msg); section_warnings.append(error_msg)
            # st.error(traceback.format_exc()); # Keep traceback for dev if needed
            section_warnings.append(f"Check server logs for detailed traceback.")
            # Decide whether to continue retrying or break on general errors

        if final_validated_data is not None: break # Exit loop if successful

    # --- Return results and status summary ---
    if final_validated_data is not None:
        status_placeholder.info(f"✅ Section '{section}' processing complete.")
        st.toast(f"Section completed: {section}", icon="✅")
        return final_validated_data, "Success", section_warnings
    else:
        status_placeholder.error(f"❌ Failed to get valid data for section '{section}' after {attempt} attempt(s).")
        return None, "Failed", section_warnings


@st.cache_data(show_spinner=False)
def render_pdf_page_to_image(_pdf_bytes, page_number, clause_ref=None, clause_wording=None, dpi=150):
    """Renders page to image, highlighting based on clause_ref first, then clause_wording snippet."""
    if not _pdf_bytes or page_number < 1: return None
    try:
        doc = fitz.open(stream=_pdf_bytes, filetype="pdf")
        page_index = page_number - 1
        if page_index < 0 or page_index >= doc.page_count:
            # st.warning(f"Render warning: Page number {page_number} out of range (1-{doc.page_count}).")
            doc.close(); return None # Silently fail if page out of range

        page = doc.load_page(page_index)
        text_instances = []
        search_term_used = None
        highlight_applied = False

        # --- Try highlighting based on Clause Reference FIRST ---
        if clause_ref and clause_ref != 'N/A':
            search_term_ref = str(clause_ref).strip()
            # Simple cleaning: remove potential leading/trailing brackets/periods if exact match fails
            cleaned_ref = re.sub(r"^[\[\(]*(.*?)[\]\)\.]*$", r"\1", search_term_ref).strip()
            if len(cleaned_ref) > 1:
                 try:
                     # Try exact match first
                     text_instances = page.search_for(cleaned_ref, flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES, quads=False)
                     if text_instances: search_term_used = f"Reference: '{cleaned_ref}'"
                     # Add more robust search logic if needed (e.g., splitting numbers/letters)
                 except Exception: pass # Ignore search errors

        # --- If reference search failed or ref not provided, try Clause Wording snippet ---
        if not text_instances and clause_wording:
            search_term_wording = str(clause_wording).strip()
            # Use a reasonably long snippet, avoiding initial/trailing spaces/punctuation
            search_snippet = search_term_wording[:100].strip(' .,:;-\'"”“()[]') # Increased length slightly
            if len(search_snippet) > 5: # Need a meaningful snippet
                 try:
                     text_instances = page.search_for(search_snippet, flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES, quads=False)
                     if text_instances: search_term_used = f"Wording snippet: '{search_snippet}...'"
                 except Exception: pass

        # --- Add Annotations if found ---
        if text_instances:
            highlight_count = 0
            try:
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.update() # Apply the annotation
                    highlight_count += 1
                if highlight_count > 0: highlight_applied = True
            except Exception as highlight_err:
                 st.error(f"Error applying highlights on page {page_number}: {highlight_err}")

        # --- Render page ---
        pix = page.get_pixmap(dpi=dpi)
        doc.close() # Close doc *after* getting pixmap
        img_bytes = pix.tobytes("png")

        # --- Report status AFTER closing doc ---
        if highlight_applied:
            st.toast(f"Highlighted {highlight_count} instance(s) on page {page_number} based on {search_term_used}.", icon="✨")
        elif clause_ref or clause_wording: # If highlight was attempted but failed
             st.toast(f"Note: Could not find exact text to highlight on page {page_number}.", icon="⚠️")

        return img_bytes
    except Exception as e:
        st.error(f"Error rendering PDF page {page_number}: {e}")
        # st.error(traceback.format_exc()) # Optional traceback for render errors
        return None


# --- 4. Initialize Session State ---
if 'current_page' not in st.session_state: st.session_state.current_page = 1
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None # Stores the final list of dicts
if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None # Raw bytes of uploaded PDF
if 'pdf_bytes_processed' not in st.session_state: st.session_state.pdf_bytes_processed = None # Flag to check if current bytes are processed
if 'processing_complete' not in st.session_state: st.session_state.processing_complete = False # Analysis finished?
if 'run_key' not in st.session_state: st.session_state.run_key = 0 # To reset file uploader widget state
if 'text_to_highlight' not in st.session_state: st.session_state.text_to_highlight = None # Wording for highlight
if 'ref_to_highlight' not in st.session_state: st.session_state.ref_to_highlight = None # Reference for highlight
if 'run_status_summary' not in st.session_state: st.session_state.run_status_summary = [] # Status messages from last run
if 'excel_data' not in st.session_state: st.session_state.excel_data = None # Prepared Excel bytes


# --- 5. Streamlit UI Logic ---

# Main Title
st.title("JASPER - Facility Agreement Analyzer")
st.markdown("Upload a PDF agreement, click 'Analyze Document' in the sidebar. Results are grouped by category below. Click clause references to view the relevant page and highlight.")

# Sidebar Setup
st.sidebar.markdown("## Controls")

# --- File Upload (in Sidebar) ---
uploaded_file_obj = st.sidebar.file_uploader(
    "Upload Facility Agreement PDF", type="pdf", key=f"pdf_uploader_{st.session_state.run_key}"
)

# Process new file upload
if uploaded_file_obj is not None:
    uploaded_bytes = uploaded_file_obj.getvalue()
    if uploaded_bytes != st.session_state.get('pdf_bytes_processed'):
        st.session_state.pdf_bytes = uploaded_bytes
        st.session_state.pdf_bytes_processed = uploaded_bytes # Mark as processed
        # Reset relevant states
        st.session_state.analysis_results = None; st.session_state.processing_complete = False
        st.session_state.current_page = 1; st.session_state.text_to_highlight = None
        st.session_state.ref_to_highlight = None; st.session_state.run_status_summary = []
        st.session_state.excel_data = None
        st.toast("✅ New PDF file loaded.", icon="📄")
        st.rerun()
elif 'pdf_bytes_processed' in st.session_state: # Clear processed flag if file removed
     st.session_state.pdf_bytes_processed = None


# --- Analysis Trigger (in Sidebar) ---
analyze_disabled = st.session_state.processing_complete or st.session_state.pdf_bytes is None
if st.sidebar.button("✨ Analyze Document", key="analyze_button", disabled=analyze_disabled, use_container_width=True, type="primary"):
    # Reset states for the new run
    st.session_state.analysis_results = None; st.session_state.processing_complete = False
    st.session_state.current_page = 1; st.session_state.run_key += 1 # Increment run key
    st.session_state.run_status_summary = []; st.session_state.excel_data = None
    st.session_state.text_to_highlight = None; st.session_state.ref_to_highlight = None

    run_start_time = datetime.now(); run_timestamp_str = run_start_time.strftime("%Y-%m-%d %H:%M:%S")
    base_file_name = getattr(uploaded_file_obj, 'name', 'uploaded_file') # Get filename safely

    # Placeholders for status updates in the main area
    status_container = st.container() # Use a container to group status messages
    progress_bar = status_container.progress(0, text="Initializing...")
    status_text = status_container.empty() # Main status text within container

    temp_dir = "temp_uploads"; os.makedirs(os.path.join(APP_DIR, temp_dir), exist_ok=True) # Ensure temp dir exists relative to app
    temp_file_path = os.path.join(APP_DIR, temp_dir, f"{run_start_time.strftime('%Y%m%d%H%M%S')}_{base_file_name}")
    gemini_uploaded_file_ref = None; all_validated_data = []; overall_success = True

    try:
        status_text.info("💾 Saving file temporarily..."); progress_bar.progress(5, text="Saving...")
        with open(temp_file_path, "wb") as f: f.write(st.session_state.pdf_bytes)

        status_text.info("🚀 Uploading to Google Cloud..."); progress_bar.progress(10, text="Uploading...")
        gemini_uploaded_file_ref = genai.upload_file(path=temp_file_path)
        # Consider removing toast if Gemini name isn't useful to end user
        # st.toast(f"☁️ Gemini File ID: {gemini_uploaded_file_ref.name}", icon="🔗")
        progress_bar.progress(15, text="Uploaded.")

        num_sections = len(SECTIONS_TO_RUN)
        progress_per_section = (95 - 15) / num_sections if num_sections > 0 else 0

        # --- Process Sections ---
        for i, section_name in enumerate(SECTIONS_TO_RUN.keys()):
            current_progress = int(15 + (i * progress_per_section))
            progress_bar.progress(current_progress, text=f"Starting {section_name}...")
            # Pass the status_text placeholder to the function for detailed updates
            section_data, section_status, section_warnings = generate_section_analysis(
                section_name, gemini_uploaded_file_ref, status_text
            )
            # Store detailed status summary
            st.session_state.run_status_summary.append({
                "section": section_name, "status": section_status, "warnings": section_warnings
            })
            if section_status == "Success" and section_data:
                # Add metadata to validated data
                for item in section_data:
                    item["File Name"] = base_file_name; item["Generation Time"] = run_timestamp_str
                all_validated_data.extend(section_data)
            else:
                overall_success = False # Mark run as not fully successful

        # Store final results in session state
        st.session_state.analysis_results = all_validated_data

        progress_bar.progress(100, text="Analysis Complete!")
        if overall_success:
             status_text.success("🏁 Analysis finished successfully!")
        else:
             status_text.warning("🏁 Analysis finished with some issues (see summary below).")
        st.session_state.processing_complete = True

    except Exception as main_err: # Catch critical errors during setup/loop
         st.error(f"❌ CRITICAL ERROR during analysis: {main_err}"); st.error(traceback.format_exc());
         overall_success = False; st.session_state.processing_complete = False # Allow retry
         st.session_state.run_status_summary.append({"section": "Overall Process", "status": "Critical Error", "warnings": [str(main_err), "Check server logs for traceback."]})
         status_text.error("Analysis failed due to a critical error.") # Update status
    finally: # Cleanup
                # status_text.empty() # Optionally clear final status message
                progress_bar.empty() # Clear progress bar

                # --- CORRECTED CLEANUP LOGIC ---
                if gemini_uploaded_file_ref and hasattr(gemini_uploaded_file_ref, 'name'): # Also check if it has the 'name' attribute
                    try:
                        st.info(f"Cleaning up Gemini file: {gemini_uploaded_file_ref.name}...")
                        genai.delete_file(name=gemini_uploaded_file_ref.name)
                        st.toast("☁️ Temporary cloud file deleted.", icon="🗑️")
                    except Exception as delete_err:
                        st.warning(f"⚠️ Could not delete Gemini file: {delete_err}") # Log warning if deletion fails

                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        # st.info("💾 Local temporary file deleted.") # Maybe too verbose
                    except Exception as remove_err:
                        st.warning(f"⚠️ Could not delete local temp file: {remove_err}")
                # --- END CORRECTION ---

    st.rerun() # Rerun to update display based on results


# --- Run Status Summary Expander (Displayed near the top) ---
if st.session_state.run_status_summary:
    final_status = "✅ Success"
    if any(s['status'] != "Success" for s in st.session_state.run_status_summary): final_status = "⚠️ Completed with Issues"
    if any("Critical" in s['status'] for s in st.session_state.run_status_summary): final_status = "❌ Failed"

    with st.expander(f"📊 Last Analysis Run Summary ({final_status})", expanded=(final_status != "✅ Success")):
        for item in st.session_state.run_status_summary:
            icon = "✅" if item['status'] == "Success" else "❌" if "Fail" in item['status'] or "Error" in item['status'] else "⚠️"
            st.markdown(f"**{item['section']}**: {icon} {item['status']}")
            if item['warnings']:
                 with st.container(): # Indent warnings
                     for msg in item['warnings']:
                         # Display validation summaries differently maybe?
                         if isinstance(msg, str) and msg.startswith("Validation Issues"):
                              st.warning(f"{msg}") # Use warning box
                         elif isinstance(msg, list): # Handle case where validate returns list
                              st.warning("\n".join(msg))
                         else:
                              st.caption(f" L> {msg}") # Use caption


# --- Display Area (Results and PDF Viewer) ---
if st.session_state.analysis_results is not None:
    col1, col2 = st.columns([6, 4]) # Results | PDF Viewer

# --- Column 1: Analysis Results (Grouped by Category using Tabs) ---
    with col1:
        st.subheader("Analysis Results")
        results_list = st.session_state.analysis_results

        if not results_list:
            st.info("Analysis complete, but no valid results were generated. Check the summary above for details.")
        else:
            # Group results by category
            grouped_results = defaultdict(list)
            categories_ordered = []
            for item in results_list:
                category = item.get("Question Category", "Uncategorized")
                if category not in grouped_results: categories_ordered.append(category)
                grouped_results[category].append(item)

            # Create tabs for each category
            if categories_ordered:
                category_tabs = st.tabs(categories_ordered)
                for i, category in enumerate(categories_ordered):
                    with category_tabs[i]:
                        category_items = grouped_results[category]
                        category_items.sort(key=lambda x: x.get('Question Number', float('inf'))) # Sort by Q#

                        for index, result_item in enumerate(category_items):
                            q_num = result_item.get('Question Number', 'N/A'); question_text = result_item.get('Question', 'N/A')
                            # --- REMOVE KEY FROM THIS LINE ---
                            with st.expander(f"**Q{q_num}:** {question_text}"):
                            # --- END REMOVAL ---
                                st.markdown(f"**Answer:** {result_item.get('Answer', 'N/A')}")
                                st.markdown("---")

                                # Evidence Section
                                evidence_list = result_item.get('Evidence', [])
                                if evidence_list:
                                    st.markdown("**Evidence:** (Click reference to view page & highlight)")
                                    for ev_index, evidence_item in enumerate(evidence_list):
                                        clause_ref = evidence_item.get('Clause Reference', 'N/A')
                                        page_num = evidence_item.get('Page Number', 0)
                                        clause_wording = evidence_item.get('Clause Wording', None)

                                        # Button to jump to page
                                        if page_num and page_num > 0:
                                            # Keep key for button, ensure it's unique
                                            button_key = f"page_btn_{category}_{q_num}_{index}_{ev_index}"
                                            button_label = f"Clause: **{clause_ref or 'Link'}** (View Page {page_num})"
                                            if st.button(button_label, key=button_key, help=f"View page {page_num} & highlight: {clause_ref or 'related text'}"):
                                                st.session_state.current_page = page_num
                                                st.session_state.ref_to_highlight = clause_ref
                                                st.session_state.text_to_highlight = clause_wording
                                                st.rerun() # Rerun needed to update viewer
                                        elif clause_ref != 'N/A':
                                            st.markdown(f"- Clause: **{clause_ref}** (Page: N/A or 0)")

                                        # Display Clause Wording
                                        if clause_wording:
                                             st.markdown(f"**Wording for '{clause_ref or 'Evidence'}':**")
                                             wording_key = f"wording_{category}_{q_num}_{index}_{ev_index}" # Unique key for text area
                                             st.text_area(label="Clause Wording", value=clause_wording, height=150, disabled=True, label_visibility="collapsed", key=wording_key)
                                             st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)

                                else: st.markdown("**Evidence:** None provided.")

                                # Justification Section
                                st.markdown("**Answer Justification:**")
                                just_key = f"justification_{category}_{q_num}_{index}" # Unique key
                                st.text_area(label="Justification", value=result_item.get('Answer Justification', ''), height=100, disabled=True, label_visibility="collapsed", key=just_key)

            else:
                 st.warning("Results generated, but could not group by category.")


            # --- Excel Download (Moved to Sidebar) ---
            st.sidebar.markdown("---")
            st.sidebar.markdown("## Export")
            if st.sidebar.button("Prepare Data for Excel Download", key="prep_excel", use_container_width=True):
                st.session_state.excel_data = None; excel_prep_status = st.sidebar.empty()
                excel_prep_status.info("Preparing Excel...")
                try:
                    excel_rows = [];
                    for item in results_list: # Use original full list
                        references = []; pages = []; first_wording = "N/A"
                        evidence = item.get("Evidence");
                        if evidence:
                            for i, ev in enumerate(evidence):
                                references.append(str(ev.get("Clause Reference", "N/A")))
                                pages.append(str(ev.get("Page Number", "0")))
                                if i == 0: first_wording = ev.get("Clause Wording", "N/A")
                        excel_row = {"File Name": item.get("File Name", ""), "Generation Time": item.get("Generation Time", ""),
                                     "Question Number": item.get("Question Number"), "Question Category": item.get("Question Category"),
                                     "Question": item.get("Question"), "Answer": item.get("Answer"), "Answer Justification": item.get("Answer Justification"),
                                     "Clause References (Concatenated)": "; ".join(references) if references else "N/A",
                                     "Page Numbers (Concatenated)": "; ".join(pages) if pages else "N/A",
                                     "First Clause Wording Found": first_wording}
                        excel_rows.append(excel_row)
                    df_excel = pd.DataFrame(excel_rows);
                    for col in EXCEL_COLUMN_ORDER: # Apply column order
                         if col not in df_excel.columns: df_excel[col] = None
                    df_excel = df_excel[EXCEL_COLUMN_ORDER]
                    output = io.BytesIO();
                    with pd.ExcelWriter(output, engine='openpyxl') as writer: df_excel.to_excel(writer, index=False, sheet_name='Analysis')
                    st.session_state.excel_data = output.getvalue()
                    excel_prep_status.success("Excel ready!")
                    time.sleep(2); excel_prep_status.empty() # Clear message
                except Exception as excel_err: excel_prep_status.error(f"Excel Error: {excel_err}")

            # Display download button in sidebar if data is ready
            if st.session_state.excel_data:
                 display_file_name = getattr(uploaded_file_obj, 'name', 'analysis') # Try to get current filename
                 safe_base_name = re.sub(r'[^\w\s-]', '', display_file_name.split('.')[0]).strip()
                 download_filename = f"Analysis_{safe_base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                 st.sidebar.download_button(
                     label="📥 Download Results as Excel", data=st.session_state.excel_data,
                     file_name=download_filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     key="download_excel_final", use_container_width=True
                 )

    # --- Column 2: Page Viewer ---
    with col2:
        # Wrap content in the styled div for sticky effect
        st.markdown('<div class="sticky-viewer-content">', unsafe_allow_html=True)

        st.subheader("📄 Page Viewer")
        if st.session_state.pdf_bytes:
            # ... (Keep the page number calculation, Nav buttons, Image Rendering logic) ...
            try:
                with fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf") as doc: total_pages = doc.page_count
            except Exception: total_pages = 1
            current_display_page = max(1, min(int(st.session_state.get('current_page', 1)), total_pages))

            nav_cols = st.columns([1, 3, 1])
            with nav_cols[0]:
                if st.button("⬅️ Prev", key="prev_page", disabled=(current_display_page <= 1)):
                    st.session_state.current_page = max(1, current_display_page - 1); st.session_state.text_to_highlight = None; st.session_state.ref_to_highlight = None; st.rerun()
            with nav_cols[1]: st.markdown(f"<div style='text-align: center; padding-top: 0.5rem;'>Page {current_display_page} of {total_pages}</div>", unsafe_allow_html=True)
            with nav_cols[2]:
                if st.button("Next ➡️", key="next_page", disabled=(current_display_page >= total_pages)):
                    st.session_state.current_page = min(total_pages, current_display_page + 1); st.session_state.text_to_highlight = None; st.session_state.ref_to_highlight = None; st.rerun()

            st.markdown("---") # Separator
            image_bytes = render_pdf_page_to_image(st.session_state.pdf_bytes, current_display_page, clause_ref=st.session_state.get('ref_to_highlight'), clause_wording=st.session_state.get('text_to_highlight'))
            if image_bytes:
                st.image(image_bytes, caption=f"Page {current_display_page}", use_column_width='always')
                # Optional caption about highlighting attempt
                # if st.session_state.get('ref_to_highlight') or st.session_state.get('text_to_highlight'):
                #      st.caption(f"Attempting highlight for Ref: '{st.session_state.get('ref_to_highlight','N/A')}' / Wording: '{st.session_state.get('text_to_highlight', '')[:50]}...'")
            else: st.warning(f"Could not render page {current_display_page}.")
        else: st.info("Upload a PDF and run analysis to view pages.")

        # Close the sticky div
        st.markdown('</div>', unsafe_allow_html=True)


# --- Fallback messages if analysis hasn't run or no PDF ---
elif st.session_state.pdf_bytes is not None and not st.session_state.processing_complete:
     st.info("PDF loaded. Click 'Analyze Document' in the sidebar.")
elif st.session_state.pdf_bytes is None:
     st.info("⬆️ Upload a PDF file using the sidebar to begin.")