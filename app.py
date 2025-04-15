# app.py
# --- COMPLETE FILE (v8 - Added Context Caching) ---

import streamlit as st
import pandas as pd
import os
import io
import base64
import fitz   # PyMuPDF
import google.generativeai as genai
# Use specific import for types to avoid conflicts if any
from google.generativeai import types as genai_types
import sys
import json
import re
import time
from datetime import datetime, timedelta, timezone # For cache TTL/expiry
import traceback
from collections import defaultdict # To group questions by category
import plotly.express as px
import google.cloud.firestore
import google.cloud.storage
import google.oauth2.service_account
from google.api_core.exceptions import NotFound as GcpNotFound # Alias for clarity
from google.api_core.exceptions import GoogleAPIError # For catching API errors
import copy # Needed for deep copying state defaults

# --- 1. SET PAGE CONFIG (MUST BE FIRST st COMMAND) ---
st.set_page_config(
    layout="wide",
    page_title="JASPER - Extraction and Review",
    page_icon="📄" # Optional: Set an emoji icon
)

# --- Inject custom CSS for sticky column (AFTER set_page_config) ---
st.markdown("""
<style>
    /* Define a class for the sticky container */
    .sticky-viewer-content {
        position: sticky;
        top: 55px; /* Adjust vertical offset from top */
        z-index: 101; /* Ensure it's above other elements */
        padding-bottom: 1rem; /* Add some space at the bottom */
        /* Ensure background color matches theme for sticky element */
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

    # Configure Google Generative AI (required before creating client)
    if "api_key" not in st.secrets:
         # Attempt to get from session state if available (e.g. user input)
         if st.session_state.get("api_key"):
              genai.configure(api_key=st.session_state.api_key)
         else:
              # If not in secrets and not in session state, raise error or handle appropriately
              st.error("❌ Google AI API Key not found in secrets and not provided by user.")
              # Optionally, stop execution or prompt user again
              # st.stop() # Or allow UI to load for key input
              # For now, let it proceed so user can input key
    else:
        genai.configure(api_key=st.secrets["api_key"]) # Prioritize secrets if available

    # Create the Generative AI Client (used for caching and potentially generation)
    # This assumes configuration is done. We might need to re-configure if API key changes.
    # We'll handle re-configuration within the analysis button logic where the key is confirmed.
    # client = genai.Client() # Instantiate client globally? Or per run? Let's do per run for key changes.


except KeyError as e:
    st.error(f"❌ Configuration Error: Missing key '{e}' in Streamlit secrets (`secrets.toml`). Please check your configuration.")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to initialize cloud clients or configure GenAI: {e}")
    print(traceback.format_exc())
    st.stop()

# --- 2. Configuration & Setup ---
MODEL_NAME = "gemini-2.5-pro-preview-03-25" # Ensure this model supports caching
CACHE_TTL_SECONDS = 900 # 15 minutes for the cache Time-To-Live
MAX_VALIDATION_RETRIES = 1
RETRY_DELAY_SECONDS = 3
PROMPT_FILE = "prompt.txt"
LOGO_FILE = "jasper-logo-1.png" # Ensure this file exists if used later
SEARCH_FALLBACK_MIN_LENGTH = 20
SEARCH_PREFIX_MIN_WORDS = 4

# --- Get the absolute path to the directory containing app.py ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load Prompt Text from File ---
try:
    prompt_path = os.path.join(APP_DIR, PROMPT_FILE)
    with open(prompt_path, 'r', encoding='utf-8') as f:
        full_prompt_text = f.read()
except FileNotFoundError:
    st.error(f"Error: Prompt file '{PROMPT_FILE}' not found in the application directory ({APP_DIR}). Please ensure it exists.")
    st.stop()
except Exception as e:
    st.error(f"Error reading prompt file '{PROMPT_FILE}': {e}")
    st.stop()

# --- Schema Definition ---
# Using genai_types for clarity
ai_response_schema = genai_types.Schema(
    type=genai_types.Type.ARRAY,
    description="List of question analysis results.",
    items=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            'Question Number': genai_types.Schema(type=genai_types.Type.INTEGER),
            'Question Category': genai_types.Schema(type=genai_types.Type.STRING),
            'Question': genai_types.Schema(type=genai_types.Type.STRING),
            'Answer': genai_types.Schema(type=genai_types.Type.STRING),
            'Answer Justification': genai_types.Schema(type=genai_types.Type.STRING),
            'Evidence': genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        'Clause Reference': genai_types.Schema(type=genai_types.Type.STRING),
                        'Clause Wording': genai_types.Schema(type=genai_types.Type.STRING),
                        'Searchable Clause Text': genai_types.Schema(type=genai_types.Type.STRING)
                    },
                    required=['Clause Reference', 'Clause Wording', 'Searchable Clause Text']
                )
            )
        },
        required=['Question Number', 'Question Category', 'Question', 'Answer', 'Answer Justification', 'Evidence']
    )
)
# Keep the dictionary version for validation logic if needed, or adapt validation to use genai_types
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
EXCEL_COLUMN_ORDER = [ "File Name", "Generation Time", "Question Number", "Question Category", "Question", "Answer", "Answer Justification", "Clause References (Concatenated)", "First Searchable Clause Text"]

# --- Section Definitions & Dependencies ---
ALL_SECTIONS = {
    "agreement_details": (1, 4),
    "eligibility_part_1": (5, 20),        # Prerequisite
    "eligibility_part_2": (21, 34),       # Prerequisite
    "eligibility_summary": (35, 36),      # Dependent
    "confidentiality": (37, 63),
    "additional_borrowers": (64, 66),
    "interest_rate_provisions": (67, 71),
    "prepayment_fee": (72, 78)
}
DEPENDENT_SECTIONS = {
    "eligibility_summary": ["eligibility_part_1", "eligibility_part_2"]
}

# --- System Instruction ---
system_instruction_text = """You are an AI assistant specialized in analyzing legal facility agreements. Carefully read the provided document (available in context) and answer the specific questions listed in the user prompt for the designated section only. Adhere strictly to the requested JSON output schema. Prioritize accuracy and extract evidence directly from the text."""

# --- 3. Helper Function Definitions ---

# filter_prompt_by_section remains largely the same
def filter_prompt_by_section(initial_full_prompt, section, exclude_final_instruction=False):
    """Filters the full prompt to include only questions for the specified section.
       Can optionally exclude the final instruction block.
    """
    if section not in ALL_SECTIONS: raise ValueError(f"Invalid section specified: {section}")
    start_q, end_q = ALL_SECTIONS[section]; questions_start_marker = "**Questions to Answer:**"; questions_end_marker = "**Final Instruction:**"

    try:
        start_index = initial_full_prompt.index(questions_start_marker)
        end_index = initial_full_prompt.index(questions_end_marker)
    except ValueError:
        raise ValueError(f"Could not find '{questions_start_marker}' or '{questions_end_marker}' markers in the prompt file.")

    prompt_header = initial_full_prompt[:start_index]
    full_questions_block = initial_full_prompt[start_index + len(questions_start_marker):end_index].strip()
    prompt_footer = initial_full_prompt[end_index:] # Includes "**Final Instruction:**" marker

    question_entries = re.split(r'\n\s*(?=\d+\.\s*?\*\*Question Category:)', full_questions_block)
    filtered_question_texts = []

    for entry in question_entries:
        entry = entry.strip()
        if not entry: continue
        match = re.match(r'^\s*(\d+)\.', entry)
        if match:
            try:
                q_num = int(match.group(1))
                if start_q <= q_num <= end_q:
                    filtered_question_texts.append(entry)
            except ValueError:
                continue # Skip entry if number parsing fails

    if not filtered_question_texts: raise ValueError(f"No questions found for section '{section}' (range {start_q}-{end_q}) in the parsed questions block.")

    filtered_questions_string = "\n\n".join(filtered_question_texts)
    section_note = f"\n\n**Current Focus:** You MUST answer ONLY the questions listed below for the '{section.upper()}' section (Questions {start_q}-{end_q}). Ignore all other questions for this specific task.\n"

    # Construct the final prompt
    final_prompt_parts = [prompt_header, section_note, questions_start_marker, "\n\n", filtered_questions_string]
    if not exclude_final_instruction:
        final_prompt_parts.append("\n\n")
        final_prompt_parts.append(prompt_footer)

    final_prompt_for_api = "".join(final_prompt_parts)
    return final_prompt_for_api

# validate_ai_data remains the same, using the dictionary schema for validation
def validate_ai_data(data, section_name):
    """Validates AI response against the schema dict. Returns (validated_data, issues_list)."""
    if not isinstance(data, list): return None, [f"CRITICAL VALIDATION ERROR: Response for section '{section_name}' is not a list."]
    validated_data = []; issues_list = []
    is_dependent_section = section_name in DEPENDENT_SECTIONS
    expected_q_nums_in_section = set()

    if section_name in ALL_SECTIONS:
        start_q, end_q = ALL_SECTIONS[section_name]
        expected_q_nums_in_section = set(range(start_q, end_q + 1))
    else:
        issues_list.append(f"Warning: Section '{section_name}' not found in master section definitions (ALL_SECTIONS). Cannot validate question range.")

    found_q_nums = set()
    for index, item in enumerate(data):
        q_num = item.get('Question Number'); q_num_str = f"Q#{q_num}" if q_num is not None else f"Item Index {index}"; is_outer_valid = True
        if not isinstance(item, dict): issues_list.append(f"{q_num_str}: Item is not a dictionary."); is_outer_valid = False; continue
        if isinstance(q_num, int): found_q_nums.add(q_num)
        else: issues_list.append(f"Item Index {index}: 'Question Number' is missing or not an integer."); is_outer_valid = False
        missing_outer_keys = AI_REQUIRED_KEYS - set(item.keys())
        if missing_outer_keys: issues_list.append(f"{q_num_str}: Missing required top-level keys: {missing_outer_keys}"); is_outer_valid = False
        evidence_list = item.get("Evidence")
        if not isinstance(evidence_list, list):
            if "Evidence" in AI_REQUIRED_KEYS: issues_list.append(f"{q_num_str}: 'Evidence' field is not a list (found type: {type(evidence_list).__name__})."); is_outer_valid = False
        else:
            for ev_index, ev_item in enumerate(evidence_list):
                ev_id_str = f"Ev[{ev_index}]"
                if not isinstance(ev_item, dict): issues_list.append(f"{q_num_str} {ev_id_str}: Evidence item is not a dictionary."); is_outer_valid = False; continue
                missing_ev_keys = AI_EVIDENCE_REQUIRED_KEYS - set(ev_item.keys())
                if missing_ev_keys: issues_list.append(f"{q_num_str} {ev_id_str}: Missing required evidence keys: {missing_ev_keys}"); is_outer_valid = False
                for key, expected_type in [("Clause Reference", str), ("Clause Wording", str), ("Searchable Clause Text", str)]:
                     if key in AI_EVIDENCE_REQUIRED_KEYS and key in ev_item and not isinstance(ev_item.get(key), expected_type): issues_list.append(f"{q_num_str} {ev_id_str}: Key '{key}' has incorrect type (expected {expected_type.__name__}, got {type(ev_item.get(key)).__name__})."); is_outer_valid = False
        if is_outer_valid: validated_data.append(item)

    if expected_q_nums_in_section:
        missing_q_nums = expected_q_nums_in_section - found_q_nums
        if missing_q_nums: issues_list.append(f"Section '{section_name}': Missing answers for expected Question Numbers: {sorted(list(missing_q_nums))}")
        unexpected_q_nums = found_q_nums - expected_q_nums_in_section
        if unexpected_q_nums: issues_list.append(f"Section '{section_name}': Received unexpected Question Numbers: {sorted(list(unexpected_q_nums))}")

    if issues_list: issues_list.insert(0, f"Validation Issues Found [Section: {section_name}] ({len(validated_data)} items passed validation):")

    if validated_data is None and isinstance(data, list): return [], issues_list
    elif validated_data is None: return None, issues_list
    else: return validated_data, issues_list


def generate_section_analysis(section, status_placeholder, api_key_to_use, cache_name=None):
    """Generates analysis for a standard (non-dependent) section, potentially using cache."""
    # Configure API key for this specific call attempt (needed if key changes mid-session)
    # Note: Re-configuring might be slightly inefficient but ensures correct key use.
    try: genai.configure(api_key=api_key_to_use)
    except Exception as config_err: status_placeholder.error(f"❌ Invalid API Key provided or configuration failed: {config_err}"); return None, "Failed", [f"Invalid API Key or config error: {config_err}"]

    status_placeholder.info(f"🔄 Starting Analysis: {section}{' (using cache)' if cache_name else ''}...");
    section_warnings = []
    try:
        # Define generation config including the schema
        generation_config = genai_types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=ai_response_schema, # Use the genai_types schema
            temperature=0.0,
        )
        safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]

        # Instantiate the model
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=system_instruction_text,
            generation_config=generation_config, # Attach schema here
            safety_settings=safety_settings
        )

        final_validated_data = None
        for attempt in range(1, MAX_VALIDATION_RETRIES + 2):
            if attempt > 1: status_placeholder.info(f"⏳ Retrying generation/validation for '{section}' (Attempt {attempt}/{MAX_VALIDATION_RETRIES+1})..."); time.sleep(RETRY_DELAY_SECONDS)
            try:
                prompt_for_api = filter_prompt_by_section(full_prompt_text, section)
                # Contents only include the prompt text now, file is in cache (if used)
                contents = [prompt_for_api]
                # Prepare request options for caching
                request_options = {'timeout': 600}
                if cache_name:
                    request_options['cached_content'] = cache_name

                status_placeholder.info(f"🧠 Calling AI for '{section}' (Attempt {attempt})...")
                response = model.generate_content(
                    contents=contents,
                    # generation_config passed during model init handles schema
                    request_options=request_options
                )
                parsed_ai_data = None; validated_ai_data = None; validation_issues = []
                status_placeholder.info(f"🔍 Processing response for '{section}'...")
                if response.parts:
                    full_response_text = response.text
                    try:
                        match = re.search(r"```json\s*([\s\S]*?)\s*```", full_response_text, re.IGNORECASE | re.DOTALL)
                        json_text = match.group(1).strip() if match else full_response_text.strip()
                        if not json_text: raise json.JSONDecodeError("Extracted JSON content is empty.", json_text, 0)
                        parsed_ai_data = json.loads(json_text)
                        status_placeholder.info(f"✔️ Validating structure for '{section}'...")
                        # Use dictionary schema for validation ease for now
                        validated_ai_data, validation_issues = validate_ai_data(parsed_ai_data, section); section_warnings.extend(validation_issues)
                        if validated_ai_data is not None and len(validated_ai_data) > 0: final_validated_data = validated_ai_data; status_placeholder.info(f"✅ Validation successful for '{section}'."); break
                        elif validated_ai_data is not None and len(validated_ai_data) == 0 and not validation_issues: status_placeholder.warning(f"⚠️ AI returned an empty list for '{section}' (Attempt {attempt})."); section_warnings.append(f"AI returned empty list for '{section}'.")
                        else: error_msg = f"Validation failed for '{section}'. Issues: {validation_issues}"; status_placeholder.warning(f"⚠️ {error_msg} (Attempt {attempt}).");
                        if validated_ai_data is None: section_warnings.append(f"CRITICAL validation error for '{section}'.")
                    # ... (rest of parsing/validation error handling remains the same) ...
                    except json.JSONDecodeError as json_err: error_msg = f"JSON Decode Error on attempt {attempt} for '{section}': {json_err}. Raw text received: '{full_response_text[:500]}...'"; st.error(error_msg); st.code(full_response_text, language='text'); section_warnings.append(error_msg)
                    except Exception as parse_validate_err: error_msg = f"Unexpected Error during parsing/validation on attempt {attempt} for '{section}': {type(parse_validate_err).__name__}: {parse_validate_err}"; st.error(error_msg); section_warnings.append(error_msg); print(traceback.format_exc())
                else:
                    # ... (response blocking/safety handling remains the same) ...
                    block_reason = "Unknown"; block_message = "N/A"; finish_reason = "Unknown"
                    try:
                        if response.prompt_feedback: block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown'); block_reason = block_reason.name if hasattr(block_reason, 'name') else str(block_reason); block_message = response.prompt_feedback.block_reason_message or "N/A"
                        if response.candidates: finish_reason = getattr(response.candidates[0], 'finish_reason', 'Unknown'); finish_reason = finish_reason.name if hasattr(finish_reason, 'name') else str(finish_reason)
                    except AttributeError: pass # Ignore errors if fields don't exist
                    if finish_reason == "SAFETY": warn_msg = f"API Response Blocked (Attempt {attempt}, Section: {section}): Reason: SAFETY. Detail: {block_reason}. Message: {block_message}"; st.error(warn_msg)
                    elif finish_reason == "RECITATION": warn_msg = f"API Response Potentially Blocked (Attempt {attempt}, Section: {section}): Finish Reason: RECITATION. Block Reason: {block_reason}."; st.warning(warn_msg)
                    elif finish_reason == "STOP" and not final_validated_data: warn_msg = f"API Response Ended (Attempt {attempt}, Section: {section}): Finish Reason: STOP, but no valid data parsed yet."; st.info(warn_msg)
                    elif finish_reason == "MAX_TOKENS": warn_msg = f"API Response Ended (Attempt {attempt}, Section: {section}): Finish Reason: MAX_TOKENS. Response might be incomplete."; st.warning(warn_msg)
                    else: warn_msg = f"API Issue (Attempt {attempt}, Section: {section}): Finish Reason: {finish_reason}. Block Reason: {block_reason}. Response may be incomplete or empty."; st.warning(warn_msg)
                    section_warnings.append(warn_msg)

            # ... (rest of API call error handling remains the same) ...
            except types.StopCandidateException as sce: error_msg = f"Generation Stopped Error (Attempt {attempt}, Section: {section}): {sce}."; st.error(error_msg); section_warnings.append(error_msg); print(traceback.format_exc())
            except GoogleAPIError as api_err:
                 error_msg = f"Google API Error (Attempt {attempt}, Section: {section}): {type(api_err).__name__}: {api_err}."; st.error(error_msg); section_warnings.append(error_msg); print(traceback.format_exc())
                 # Check if it's a caching specific error? e.g., cache not found, expired?
                 if "cache" in str(api_err).lower():
                      st.warning("Potential issue with context cache. Trying without cache if possible.")
                      # Ideally, we'd retry without the cache, but that complicates the loop.
                      # For now, just log and let the retry happen (or fail).
            except Exception as e: error_msg = f"Processing Error during API call/prompt generation (Attempt {attempt}, Section: {section}): {type(e).__name__}: {e}"; st.error(error_msg); section_warnings.append(error_msg); section_warnings.append("Traceback logged to console."); print(traceback.format_exc())

            if final_validated_data is not None: break # Exit retry loop if successful

        # After retry loop
        if final_validated_data is not None: status_placeholder.success(f"✅ Analysis completed successfully for: {section}."); return final_validated_data, "Success", section_warnings
        else: status_placeholder.error(f"❌ Analysis failed for: {section} after {attempt} attempts."); section_warnings.append(f"Failed to get valid response for section '{section}' after {MAX_VALIDATION_RETRIES + 1} attempts."); return None, "Failed", section_warnings
    except Exception as outer_err: error_msg = f"Critical Error during setup or execution for section '{section}': {type(outer_err).__name__}: {outer_err}"; st.error(error_msg); section_warnings.append(error_msg); section_warnings.append("Traceback logged to console."); print(traceback.format_exc()); status_placeholder.error(f"❌ Critical failure processing section: {section}."); return None, "Failed", section_warnings


# format_results_for_summary remains the same
def format_results_for_summary(prerequisite_results_map):
    """Formats results from prerequisite sections for inclusion in a summary prompt."""
    context_str = ""
    for section_name, results in prerequisite_results_map.items():
        if results: # Only include context if results exist
            context_str += f"\n\n--- Context from Section: {section_name} ---\n"
            for item in results:
                q_num = item.get('Question Number', 'N/A')
                answer = item.get('Answer', 'N/A')
                context_str += f"Q{q_num}: {answer}\n"
            context_str += "--- End Context ---\n"
    return context_str


def generate_summary_with_context(section, prerequisite_results_map, status_placeholder, api_key_to_use, cache_name=None):
    """Generates analysis for a dependent section, injecting context and potentially using cache."""
    try: genai.configure(api_key=api_key_to_use)
    except Exception as config_err: status_placeholder.error(f"❌ Invalid API Key provided or configuration failed: {config_err}"); return None, "Failed", [f"Invalid API Key or config error: {config_err}"]

    status_placeholder.info(f"🔄 Starting Summary Analysis: {section}{' (using cache)' if cache_name else ''} (with context)...");
    section_warnings = []
    try:
        # Define generation config including the schema
        generation_config = genai_types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=ai_response_schema, # Use the genai_types schema
            temperature=0.0,
        )
        safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]

        # Instantiate the model
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=system_instruction_text, # System instruction applies to cached context too
            generation_config=generation_config, # Attach schema here
            safety_settings=safety_settings
        )

        final_validated_data = None

        # 1. Filter prompt to get ONLY the summary questions
        summary_questions_prompt_part = filter_prompt_by_section(full_prompt_text, section, exclude_final_instruction=True) # Exclude generic footer

        # 2. Format the prerequisite results
        context_string = format_results_for_summary(prerequisite_results_map)
        if not context_string:
             status_placeholder.warning(f"⚠️ Cannot generate summary for '{section}' as prerequisite context is missing or empty.")
             return None, "Failed", [f"Missing prerequisite context for {section}"]

        # 3. Construct the full prompt with context
        questions_start_marker = "**Questions to Answer:**"
        questions_end_marker = "**Final Instruction:**"
        start_index = full_prompt_text.index(questions_start_marker)
        end_index = full_prompt_text.index(questions_end_marker)
        prompt_header = full_prompt_text[:start_index]
        prompt_footer = full_prompt_text[end_index:]

        prompt_for_api = (
            f"{prompt_header}\n"
            f"**Context from Prerequisite Sections:**\n"
            f"{context_string}\n"
            f"**Task Specific to this Section ({section}):**\n"
            f"Based ONLY on the document (available in the context cache if used) AND the context provided above from prerequisite sections, "
            f"answer the following specific questions for the '{section}' section.\n"
            f"{summary_questions_prompt_part}\n"
            f"{prompt_footer}"
        )

        # 4. Call API and Validate (similar loop as standard generation)
        for attempt in range(1, MAX_VALIDATION_RETRIES + 2):
            if attempt > 1: status_placeholder.info(f"⏳ Retrying summary generation/validation for '{section}' (Attempt {attempt}/{MAX_VALIDATION_RETRIES+1})..."); time.sleep(RETRY_DELAY_SECONDS)
            try:
                # Contents only include the prompt text
                contents = [prompt_for_api]
                request_options = {'timeout': 600}
                if cache_name:
                    request_options['cached_content'] = cache_name

                status_placeholder.info(f"🧠 Calling AI for summary '{section}' (Attempt {attempt})...")
                response = model.generate_content(
                    contents=contents,
                    # generation_config from model init handles schema
                    request_options=request_options
                )
                parsed_ai_data = None; validated_ai_data = None; validation_issues = []
                status_placeholder.info(f"🔍 Processing summary response for '{section}'...")
                if response.parts:
                    full_response_text = response.text
                    try:
                        match = re.search(r"```json\s*([\s\S]*?)\s*```", full_response_text, re.IGNORECASE | re.DOTALL)
                        json_text = match.group(1).strip() if match else full_response_text.strip()
                        if not json_text: raise json.JSONDecodeError("Extracted JSON content is empty.", json_text, 0)
                        parsed_ai_data = json.loads(json_text)
                        status_placeholder.info(f"✔️ Validating summary structure for '{section}'...")
                        validated_ai_data, validation_issues = validate_ai_data(parsed_ai_data, section); section_warnings.extend(validation_issues)
                        if validated_ai_data is not None and len(validated_ai_data) > 0: final_validated_data = validated_ai_data; status_placeholder.info(f"✅ Summary validation successful for '{section}'."); break
                        elif validated_ai_data is not None and len(validated_ai_data) == 0 and not validation_issues: status_placeholder.warning(f"⚠️ AI returned an empty list for summary '{section}' (Attempt {attempt})."); section_warnings.append(f"AI returned empty list for summary '{section}'.")
                        else: error_msg = f"Summary validation failed for '{section}'. Issues: {validation_issues}"; status_placeholder.warning(f"⚠️ {error_msg} (Attempt {attempt}).");
                        if validated_ai_data is None: section_warnings.append(f"CRITICAL summary validation error for '{section}'.")
                    # ... (rest of summary parsing/validation error handling) ...
                    except json.JSONDecodeError as json_err: error_msg = f"JSON Decode Error (Summary) on attempt {attempt} for '{section}': {json_err}. Raw text: '{full_response_text[:500]}...'"; st.error(error_msg); st.code(full_response_text, language='text'); section_warnings.append(error_msg)
                    except Exception as parse_validate_err: error_msg = f"Unexpected Error during summary parsing/validation on attempt {attempt} for '{section}': {type(parse_validate_err).__name__}: {parse_validate_err}"; st.error(error_msg); section_warnings.append(error_msg); print(traceback.format_exc())
                else:
                    # ... (summary response blocking/safety handling) ...
                    block_reason = "Unknown"; block_message = "N/A"; finish_reason = "Unknown"
                    try:
                        if response.prompt_feedback: block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown'); block_reason = block_reason.name if hasattr(block_reason, 'name') else str(block_reason); block_message = response.prompt_feedback.block_reason_message or "N/A"
                        if response.candidates: finish_reason = getattr(response.candidates[0], 'finish_reason', 'Unknown'); finish_reason = finish_reason.name if hasattr(finish_reason, 'name') else str(finish_reason)
                    except AttributeError: pass
                    if finish_reason == "SAFETY": warn_msg = f"API Response Blocked (Summary Attempt {attempt}, Section: {section}): Reason: SAFETY. Detail: {block_reason}. Message: {block_message}"; st.error(warn_msg)
                    elif finish_reason == "RECITATION": warn_msg = f"API Response Potentially Blocked (Summary Attempt {attempt}, Section: {section}): Finish Reason: RECITATION. Block Reason: {block_reason}."; st.warning(warn_msg)
                    elif finish_reason == "STOP" and not final_validated_data: warn_msg = f"API Response Ended (Summary Attempt {attempt}, Section: {section}): Finish Reason: STOP, but no valid data parsed yet."; st.info(warn_msg)
                    elif finish_reason == "MAX_TOKENS": warn_msg = f"API Response Ended (Summary Attempt {attempt}, Section: {section}): Finish Reason: MAX_TOKENS. Response might be incomplete."; st.warning(warn_msg)
                    else: warn_msg = f"API Issue (Summary Attempt {attempt}, Section: {section}): Finish Reason: {finish_reason}. Block Reason: {block_reason}. Response may be incomplete or empty."; st.warning(warn_msg)
                    section_warnings.append(warn_msg)

            # ... (rest of summary API call error handling) ...
            except types.StopCandidateException as sce: error_msg = f"Generation Stopped Error (Summary Attempt {attempt}, Section: {section}): {sce}."; st.error(error_msg); section_warnings.append(error_msg); print(traceback.format_exc())
            except GoogleAPIError as api_err:
                 error_msg = f"Google API Error (Summary Attempt {attempt}, Section: {section}): {type(api_err).__name__}: {api_err}."; st.error(error_msg); section_warnings.append(error_msg); print(traceback.format_exc())
                 if "cache" in str(api_err).lower():
                      st.warning("Potential issue with context cache during summary generation.")
            except Exception as e: error_msg = f"Processing Error (Summary) during API call/prompt generation (Attempt {attempt}, Section: {section}): {type(e).__name__}: {e}"; st.error(error_msg); section_warnings.append(error_msg); section_warnings.append("Traceback logged to console."); print(traceback.format_exc())

            if final_validated_data is not None: break # Exit retry loop if successful

        # 5. Return results or failure status
        if final_validated_data is not None: status_placeholder.success(f"✅ Summary analysis completed successfully for: {section}."); return final_validated_data, "Success", section_warnings
        else: status_placeholder.error(f"❌ Summary analysis failed for: {section} after {attempt} attempts."); section_warnings.append(f"Failed to get valid summary response for section '{section}' after {MAX_VALIDATION_RETRIES + 1} attempts."); return None, "Failed", section_warnings
    except Exception as outer_err: error_msg = f"Critical Error during summary setup or execution for section '{section}': {type(outer_err).__name__}: {outer_err}"; st.error(error_msg); section_warnings.append(error_msg); section_warnings.append("Traceback logged to console."); print(traceback.format_exc()); status_placeholder.error(f"❌ Critical failure processing summary section: {section}."); return None, "Failed", section_warnings


# PDF processing and GCS functions remain the same
@st.cache_data(show_spinner=False)
def find_text_in_pdf(_pdf_bytes, search_text):
    """Searches PDF. Returns (first_page_found, instances_on_first_page, term_used, status_msg, all_findings)"""
    if not _pdf_bytes or not search_text: return None, None, None, "Invalid input (PDF bytes or search text missing).", None
    doc = None; search_text_cleaned = search_text.strip(); words = search_text_cleaned.split(); num_words = len(words)
    search_attempts = []
    if num_words >= SEARCH_PREFIX_MIN_WORDS and num_words > 5: term_10 = ' '.join(words[:10]); search_attempts.append({'term': term_10, 'desc': "first 10 words"})
    if num_words >= SEARCH_PREFIX_MIN_WORDS:
        term_5 = ' '.join(words[:5])
        if not search_attempts or term_5 != search_attempts[0]['term']: search_attempts.append({'term': term_5, 'desc': "first 5 words"})
    term_full = search_text_cleaned;
    if term_full and not any(term_full == a['term'] for a in search_attempts): search_attempts.append({'term': term_full, 'desc': "full text"})
    sentences = re.split(r'(?<=[.?!])\s+', term_full); term_sentence = sentences[0].strip() if sentences else ""
    if len(term_sentence) >= SEARCH_FALLBACK_MIN_LENGTH and not any(term_sentence == a['term'] for a in search_attempts): search_attempts.append({'term': term_sentence, 'desc': "first sentence fallback"})
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
        raise

def download_from_gcs(bucket_name, source_blob_name):
    """Downloads a blob from the bucket."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        pdf_bytes = blob.download_as_bytes()
        return pdf_bytes
    except GcpNotFound: # Use aliased exception
        st.error(f"❌ Error: PDF file not found in Google Cloud Storage at gs://{bucket_name}/{source_blob_name}")
        return None
    except Exception as e:
        st.error(f"❌ Failed to download PDF from GCS: {e}")
        print(f"GCS Download Error: {e}\n{traceback.format_exc()}")
        return None

# reset_app_state remains the same
def reset_app_state():
    """Resets the session state for a new analysis or clearing history."""
    st.session_state.pdf_bytes = None
    st.session_state.pdf_display_ready = False
    st.session_state.analysis_results = None
    st.session_state.analysis_complete = False
    st.session_state.processing_in_progress = False
    st.session_state.current_page = 1
    st.session_state.run_status_summary = []
    st.session_state.excel_data = None
    st.session_state.search_trigger = None
    st.session_state.last_search_result = None
    st.session_state.show_wording_states = defaultdict(bool)
    st.session_state.viewing_history = False
    st.session_state.history_filename = None
    st.session_state.history_timestamp = None
    st.session_state.current_filename = None


# --- 4. Initialize Session State ---
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
    'api_key': st.secrets.get("api_key", None), # Initialize from secrets if present
    'selected_sections_to_run': list(ALL_SECTIONS.keys())[:1],
    'load_history_id': None,
    'viewing_history': False,
    'history_filename': None,
    'history_timestamp': None,
    'current_filename': None
}
for key, default_value in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(default_value) if isinstance(default_value, (dict, list, set, defaultdict)) else default_value


# --- 5. Load History Data (if triggered) ---
# This section remains the same
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
            timestamp = run_data.get("analysis_timestamp")
            run_summary = run_data.get("run_status", [])
            if not gcs_pdf_path: st.error("❌ History record missing PDF path."); raise ValueError("Missing GCS Path")
            if not results: st.error("❌ History record missing results."); raise ValueError("Missing Results")

            if gcs_pdf_path.startswith("gs://"):
                path_parts = gcs_pdf_path[5:].split("/", 1)
                hist_bucket_name = path_parts[0]; hist_blob_name = path_parts[1]
            else: hist_bucket_name = GCS_BUCKET_NAME; hist_blob_name = gcs_pdf_path

            with st.spinner(f"Downloading PDF from gs://{hist_bucket_name}/{hist_blob_name}..."):
                pdf_bytes_from_hist = download_from_gcs(hist_bucket_name, hist_blob_name)

            if pdf_bytes_from_hist:
                reset_app_state()
                st.session_state.pdf_bytes = pdf_bytes_from_hist
                st.session_state.analysis_results = results
                st.session_state.run_status_summary = run_summary
                st.session_state.analysis_complete = True
                st.session_state.pdf_display_ready = True
                st.session_state.viewing_history = True
                st.session_state.history_filename = filename
                st.session_state.history_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp)
                st.session_state.current_page = 1
                st.session_state.current_filename = filename
                st.success(f"✅ Loaded history for '{filename}' ({st.session_state.history_timestamp}).")
                time.sleep(1); st.rerun()
            else: st.session_state.viewing_history = False # Failed download
        else: st.error(f"❌ History record ID '{history_id}' not found."); st.session_state.viewing_history = False
    except Exception as e: st.error(f"❌ Error loading history: {e}"); print(f"History Load Error: {e}\n{traceback.format_exc()}"); st.session_state.viewing_history = False


# --- 6. Streamlit UI Logic ---
st.title("JASPER - Just A Smart Platform for Extraction and Review")

# --- Display History Mode Banner ---
# This section remains the same
if st.session_state.viewing_history:
    hist_ts_str = st.session_state.history_timestamp or "N/A"
    st.info(f"📜 **Viewing Historical Analysis:** File: **{st.session_state.history_filename}** (Generated: {hist_ts_str})")
    if st.button("⬅️ Exit History View / Start New Analysis", key="clear_history_view"):
        reset_app_state()
        st.rerun()
    st.markdown("---")
elif not st.session_state.analysis_complete:
    st.markdown("Upload a PDF agreement, **enter your Gemini API Key**, **select sections**, click 'Analyse'. Results grouped below. Click clause references to view & highlight.")

# --- Sidebar Setup ---
st.sidebar.markdown("## Controls")

# --- Conditional Controls (Only show if NOT viewing history) ---
if not st.session_state.viewing_history:
    st.sidebar.markdown("### 1. API Key")
    api_key_input = st.sidebar.text_input("Enter your Google AI Gemini API Key", type="password", key="api_key_input", help="Your API key is used only for this session and is not stored.", value=st.session_state.get("api_key", ""))
    if api_key_input and api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        # Reconfigure GenAI client if key changes
        try:
             genai.configure(api_key=st.session_state.api_key)
             st.toast("API Key updated.", icon="🔑")
        except Exception as config_err:
             st.sidebar.error(f"Invalid API Key: {config_err}")
             # Optionally clear the key from state if invalid
             # st.session_state.api_key = None

    if not st.session_state.api_key and not st.session_state.analysis_complete and not st.session_state.processing_in_progress : st.sidebar.warning("API Key required to run analysis.", icon="🔑")

    st.sidebar.markdown("### 2. Upload PDF")
    uploaded_file_obj = st.sidebar.file_uploader("Upload Facility Agreement PDF", type="pdf", key=f"pdf_uploader_{st.session_state.get('run_key', 0)}")
    if uploaded_file_obj is not None:
        new_file_bytes = uploaded_file_obj.getvalue()
        if new_file_bytes != st.session_state.get('pdf_bytes'):
             current_api_key_before_reset = st.session_state.api_key
             current_sections_before_reset = st.session_state.selected_sections_to_run
             reset_app_state()
             st.session_state.pdf_bytes = new_file_bytes
             st.session_state.pdf_display_ready = True
             st.session_state.current_filename = uploaded_file_obj.name
             st.session_state.api_key = current_api_key_before_reset
             st.session_state.selected_sections_to_run = current_sections_before_reset
             st.toast("✅ New PDF file loaded.", icon="📄"); st.rerun()
    elif not st.session_state.pdf_bytes: st.sidebar.info("Upload a PDF to enable analysis.")

    st.sidebar.markdown("### 3. Select Sections")
    selected_sections = st.sidebar.multiselect("Choose sections:", options=list(ALL_SECTIONS.keys()), default=st.session_state.selected_sections_to_run, key="section_selector", help="Select which parts of the document you want to analyse in this run.")
    if selected_sections != st.session_state.selected_sections_to_run: st.session_state.selected_sections_to_run = selected_sections

    st.sidebar.markdown("### 4. Run Analysis")
    can_analyse = (st.session_state.pdf_bytes is not None and st.session_state.api_key is not None and not st.session_state.processing_in_progress and not st.session_state.analysis_complete and st.session_state.selected_sections_to_run)
    analyse_button_tooltip = "Analysis complete." if st.session_state.analysis_complete else "Running..." if st.session_state.processing_in_progress else "Upload PDF." if not st.session_state.pdf_bytes else "Enter API key." if not st.session_state.api_key else "Select sections." if not st.session_state.selected_sections_to_run else "Start analysis"
    if st.sidebar.button("✨ Analyse Document", key="analyse_button", disabled=not can_analyse, help=analyse_button_tooltip, use_container_width=True, type="primary"):
        # --- Start Analysis Process with Caching ---
        if not st.session_state.api_key: st.error("API Key missing."); st.stop()
        if not st.session_state.pdf_bytes: st.error("PDF missing."); st.stop()
        if not st.session_state.selected_sections_to_run: st.error("No sections selected."); st.stop()

        st.session_state.processing_in_progress = True
        st.session_state.analysis_complete = False
        st.session_state.analysis_results = []
        st.session_state.run_key += 1
        st.session_state.run_status_summary = []
        st.session_state.excel_data = None
        st.session_state.search_trigger = None
        st.session_state.last_search_result = None
        st.session_state.show_wording_states = defaultdict(bool)

        current_api_key = st.session_state.api_key # Use confirmed key
        run_start_time = datetime.now()
        run_timestamp_str = run_start_time.strftime("%Y-%m-%d %H:%M:%S")
        if not st.session_state.current_filename: st.session_state.current_filename = f"analysis_{run_start_time.strftime('%Y%m%d%H%M%S')}.pdf"; st.warning("Filename fallback used.")
        base_file_name = st.session_state.current_filename

        # --- Configure GenAI and Create Client for this run ---
        try:
             genai.configure(api_key=current_api_key)
             client = genai.Client() # Create client instance for this run
             st.toast("API Key validated, Client created.", icon="🔑")
        except Exception as config_err:
             st.error(f"❌ Failed to configure Gemini API/Client: {config_err}"); st.session_state.processing_in_progress = False; st.stop()

        status_container = st.container()
        progress_bar = status_container.progress(0, text="Initializing analysis...")
        status_text = status_container.empty()

        temp_dir = "temp_uploads"
        safe_base_name = re.sub(r'[^\w\-.]', '_', base_file_name)
        temp_file_path = os.path.join(APP_DIR, temp_dir, f"{st.session_state.run_key}_{safe_base_name}")
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        gemini_uploaded_file_ref = None # Stores the File API object
        cache = None # Stores the Cache object
        cache_name = None # Stores the cache resource name string
        all_validated_data = []
        overall_success = True
        gcs_file_path = None
        section_results_map = {}

        try:
            # --- File Upload & Caching ---
            status_text.info("💾 Saving file temporarily..."); progress_bar.progress(2, text="Saving temp file...");
            with open(temp_file_path, "wb") as f: f.write(st.session_state.pdf_bytes)

            status_text.info("☁️ Uploading file to Google Cloud AI (File API)..."); progress_bar.progress(5, text="Uploading...")
            # Use client instance for file operations
            gemini_uploaded_file_ref = client.upload_file(path=temp_file_path)
            st.toast(f"File '{gemini_uploaded_file_ref.display_name}' uploaded.", icon="☁️")

            # --- Wait for File Processing (Important for Caching) ---
            status_text.info("⏳ Waiting for file processing before caching...")
            progress_bar.progress(8, text="Processing file...")
            while gemini_uploaded_file_ref.state.name == "PROCESSING":
                time.sleep(2) # Wait longer for potentially large files
                gemini_uploaded_file_ref = client.get_file(name=gemini_uploaded_file_ref.name)
                if gemini_uploaded_file_ref.state.name == "FAILED":
                     raise GoogleAPIError(f"File processing failed: {gemini_uploaded_file_ref.name}")
            status_text.info(f"✅ File '{gemini_uploaded_file_ref.name}' processed.")
            progress_bar.progress(10, text="File processed.")


            # --- Create Context Cache ---
            try:
                status_text.info("🧠 Creating context cache for the document...")
                progress_bar.progress(12, text="Creating cache...")
                cache = client.create_cache(
                     model=MODEL_NAME, # Use the same model as for generation
                     contents=[gemini_uploaded_file_ref], # Cache the processed file reference
                     ttl=timedelta(seconds=CACHE_TTL_SECONDS),
                     display_name=f"Cache for {safe_base_name} - Run {st.session_state.run_key}" # Optional: Add display name
                )
                cache_name = cache.name
                st.toast(f"Context cache created.", icon="🧠")
                status_text.info(f"✅ Context cache created: {cache_name}")
                progress_bar.progress(15, text="Cache created.")
            except GoogleAPIError as cache_err:
                # Handle potential errors like token limits for caching
                st.warning(f"⚠️ Failed to create context cache: {cache_err}. Proceeding without cache.")
                print(f"Cache Creation Warning: {cache_err}")
                cache_name = None # Fallback to non-cached mode
                progress_bar.progress(15, text="Cache creation failed, proceeding without.")
            except Exception as general_cache_err:
                st.warning(f"⚠️ Unexpected error creating context cache: {general_cache_err}. Proceeding without cache.")
                print(f"Cache Creation Error: {general_cache_err}\n{traceback.format_exc()}")
                cache_name = None
                progress_bar.progress(15, text="Cache creation error, proceeding without.")


            # --- Analysis Loop (Two Passes with Cache) ---
            selected_sections_to_run = st.session_state.selected_sections_to_run
            processed_sections = set()
            sections_to_process_pass1 = []
            sections_to_process_pass2 = []
            for section_name in selected_sections_to_run:
                if section_name in DEPENDENT_SECTIONS: sections_to_process_pass2.append(section_name)
                else: sections_to_process_pass1.append(section_name)

            total_sections = len(selected_sections_to_run)
            progress_per_section = (95 - 15) / total_sections if total_sections > 0 else 0
            sections_completed_count = 0

            # Pass 1: Independent Sections
            status_text.info(f"🚀 Starting Pass 1: Independent Sections{(' using cache ' + cache_name) if cache_name else ''}")
            for section_name in sections_to_process_pass1:
                if section_name not in ALL_SECTIONS: st.warning(f"Skipping invalid section '{section_name}'."); continue
                current_progress = int(15 + (sections_completed_count * progress_per_section)); progress_bar.progress(current_progress, text=f"Analysing Section: {section_name}...")
                # Pass API key and cache_name
                section_data, section_status, section_warnings = generate_section_analysis(section_name, status_text, current_api_key, cache_name=cache_name)
                st.session_state.run_status_summary.append({"section": section_name, "status": section_status, "warnings": section_warnings})

                if section_status == "Success" and section_data:
                    for item in section_data: item["File Name"] = base_file_name; item["Generation Time"] = run_timestamp_str
                    section_results_map[section_name] = section_data
                    all_validated_data.extend(section_data)
                else:
                    section_results_map[section_name] = None
                    overall_success = False

                processed_sections.add(section_name)
                sections_completed_count += 1
                progress_bar.progress(int(15 + (sections_completed_count * progress_per_section)), text=f"Completed: {section_name}")

            # Pass 2: Dependent Sections
            status_text.info(f"🚀 Starting Pass 2: Dependent Sections{(' using cache ' + cache_name) if cache_name else ''}")
            for section_name in sections_to_process_pass2:
                if section_name not in ALL_SECTIONS: st.warning(f"Skipping invalid dependent section '{section_name}'."); continue
                if section_name in processed_sections: continue

                prerequisites = DEPENDENT_SECTIONS.get(section_name, [])
                prereqs_met = True
                prerequisite_results_for_context = {}
                for prereq_section in prerequisites:
                    if prereq_section not in selected_sections_to_run:
                        st.warning(f"Skipping '{section_name}': Prerequisite '{prereq_section}' not selected.")
                        st.session_state.run_status_summary.append({"section": section_name, "status": "Skipped", "warnings": [f"Prerequisite '{prereq_section}' not selected."]})
                        prereqs_met = False; break
                    if prereq_section not in section_results_map or section_results_map[prereq_section] is None:
                        st.warning(f"Skipping '{section_name}': Prerequisite '{prereq_section}' failed/missing.")
                        st.session_state.run_status_summary.append({"section": section_name, "status": "Skipped", "warnings": [f"Prerequisite '{prereq_section}' failed or missing."]})
                        prereqs_met = False; break
                    prerequisite_results_for_context[prereq_section] = section_results_map[prereq_section]

                if not prereqs_met:
                    overall_success = False
                    sections_completed_count += 1
                    progress_bar.progress(int(15 + (sections_completed_count * progress_per_section)), text=f"Skipped: {section_name}")
                    continue

                current_progress = int(15 + (sections_completed_count * progress_per_section)); progress_bar.progress(current_progress, text=f"Analysing Summary: {section_name}...")
                # Pass API key and cache_name
                section_data, section_status, section_warnings = generate_summary_with_context(section_name, prerequisite_results_for_context, status_text, current_api_key, cache_name=cache_name)
                st.session_state.run_status_summary.append({"section": section_name, "status": section_status, "warnings": section_warnings})

                if section_status == "Success" and section_data:
                    for item in section_data: item["File Name"] = base_file_name; item["Generation Time"] = run_timestamp_str
                    all_validated_data.extend(section_data)
                else: overall_success = False

                processed_sections.add(section_name)
                sections_completed_count += 1
                progress_bar.progress(int(15 + (sections_completed_count * progress_per_section)), text=f"Completed: {section_name}")

            # --- Finalize Analysis ---
            st.session_state.analysis_results = all_validated_data
            progress_bar.progress(100, text="Analysis process finished!")
            if overall_success and all_validated_data: status_text.success("🏁 Analysis finished successfully!")
            elif all_validated_data: status_text.warning("🏁 Analysis finished, but some sections had issues/were skipped.")
            else: status_text.error("🏁 Analysis finished, but failed to generate any valid results.")
            st.session_state.analysis_complete = True

            # --- Save to GCS and Firestore ---
            if all_validated_data and st.session_state.pdf_bytes:
                try:
                    timestamp = datetime.now(timezone.utc) # Use timezone-aware timestamp
                    firestore_doc_id = f"{safe_base_name}_{timestamp.isoformat()}"
                    gcs_blob_name = f"{GCS_PDF_FOLDER}/{firestore_doc_id}.pdf"
                    gcs_file_path = upload_to_gcs(GCS_BUCKET_NAME, st.session_state.pdf_bytes, gcs_blob_name, status_text)
                    status_text.info("💾 Saving results and PDF reference to database...")
                    doc_ref = db.collection("analysis_runs").document(firestore_doc_id)
                    doc_ref.set({
                        "filename": base_file_name, "analysis_timestamp": timestamp,
                        "results": all_validated_data, "run_status": st.session_state.run_status_summary,
                        "gcs_pdf_path": gcs_file_path
                    })
                    status_text.success("💾 Results and PDF link saved successfully.")
                    time.sleep(1)
                except Exception as db_gcs_err:
                    st.error(f"❌ Failed to save results/PDF to cloud: {db_gcs_err}")
                    print(f"DB/GCS Save Error: {db_gcs_err}\n{traceback.format_exc()}")
                    st.session_state.run_status_summary.append({ "section": "Cloud Save", "status": "Failed", "warnings": [f"Error saving to GCS/Firestore: {db_gcs_err}"]})
                    overall_success = False

        except Exception as main_err:
            st.error(f"❌ CRITICAL ERROR during analysis workflow: {main_err}"); print(traceback.format_exc())
            overall_success = False; st.session_state.analysis_complete = False
            st.session_state.run_status_summary.append({"section": "Overall Process Control", "status": "Critical Error", "warnings": [str(main_err), "Analysis halted."]})
            status_text.error(f"Analysis stopped due to critical error: {main_err}")
        finally:
            # --- Cleanup ---
            st.session_state.processing_in_progress = False
            time.sleep(2); status_text.empty(); progress_bar.empty() # Shorten sleep

            # Delete Context Cache
            if cache_name:
                try:
                    status_text.info(f"🗑️ Deleting context cache: {cache_name}...")
                    client.delete_cache(name=cache_name)
                    st.toast("Context cache deleted.", icon="🗑️"); time.sleep(0.5)
                    status_text.empty()
                except Exception as del_cache_err:
                    st.sidebar.warning(f"Cache cleanup issue: {del_cache_err}", icon="⚠️")
                    print(f"WARN: Failed to delete context cache {cache_name}: {del_cache_err}")
                    status_text.warning(f"Could not delete context cache: {del_cache_err}") # Show warning longer

            # Delete Uploaded File (File API)
            if gemini_uploaded_file_ref and hasattr(gemini_uploaded_file_ref, 'name'):
                try:
                     # Check state before deleting - avoid deleting if failed/still processing?
                     # For simplicity, attempt delete regardless for now.
                    status_text.info(f"☁️ Deleting temporary cloud file: {gemini_uploaded_file_ref.name}...");
                    client.delete_file(name=gemini_uploaded_file_ref.name)
                    st.toast("Cloud file deleted.", icon="🗑️"); time.sleep(0.5)
                    status_text.empty()
                except Exception as del_err:
                    st.sidebar.warning(f"Cloud file cleanup issue: {del_err}", icon="⚠️")
                    print(f"WARN: Failed to delete cloud file {gemini_uploaded_file_ref.name}: {del_err}")
                    # Don't leave status message hanging if cache deletion also failed
                    if not cache_name: status_text.warning(f"Could not delete cloud file: {del_err}")


            # Delete Local Temp File
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except Exception as local_del_err: st.sidebar.warning(f"Local temp file cleanup issue: {local_del_err}", icon="⚠️"); print(f"WARN: Failed to delete local temp file {temp_file_path}: {local_del_err}")

            # Clear status text finally
            time.sleep(1)
            status_text.empty()

        # Rerun after analysis and cleanup to update UI
        st.rerun()
else:
    # Sidebar for history view
    st.sidebar.info("📜 Viewing historical data.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Click 'Exit History View' above to start a new analysis.")


# --- 7. Display Area (Results and PDF Viewer) ---
# This section remains the same - no changes needed here for caching logic
if st.session_state.pdf_bytes is not None:
    col1, col2 = st.columns([6, 4]) # Results | PDF Viewer

    # --- Column 1: Analysis Results ---
    with col1:
        # --- Run Status Summary ---
        if st.session_state.run_status_summary:
            final_status = "✅ Success"; has_failures = any(s['status'] == "Failed" or "Error" in s['status'] or "Skipped" in s['status'] for s in st.session_state.run_status_summary); has_warnings = any(s['status'] != "Success" and not has_failures for s in st.session_state.run_status_summary)
            if has_failures: final_status = "❌ Failed / Skipped"
            elif has_warnings: final_status = "⚠️ Issues"
            expand_summary = st.session_state.viewing_history or (final_status != "✅ Success")
            with st.expander(f"📊 Analysis Run Summary ({final_status})", expanded=expand_summary):
                for item in st.session_state.run_status_summary:
                    icon = "✅" if item['status'] == "Success" else "❌" if item['status'] == "Failed" or "Error" in item['status'] else "➡️" if "Skipped" in item['status'] else "⚠️"
                    status_text_display = item['status']
                    st.markdown(f"**{item['section']}**: {icon} {status_text_display}")
                    if item['warnings']:
                        filtered_warnings = [msg for msg in item['warnings'] if not (isinstance(msg, str) and msg.startswith("Validation Issues Found"))]
                        if filtered_warnings:
                            with st.container(): st.caption("Details:")
                            for msg in filtered_warnings:
                                msg_str = str(msg);
                                if any(term in msg_str for term in ["CRITICAL", "Error", "Block", "Fail"]) and "permission" not in msg_str.lower(): st.error(f" L> {msg_str}")
                                elif any(term in msg_str.lower() for term in ["warn", "missing", "unexpected", "empty list", "validation issue", "permission issue", "recitation", "max_tokens", "timeout", "cache"]): st.warning(f" L> {msg_str}")
                                elif "Skipped" in status_text_display: st.info(f" L> {msg_str}")
                                else: st.caption(f" L> {msg_str}")

        st.subheader("Analysis Results")

        # --- Display Results ---
        if (st.session_state.analysis_complete or st.session_state.viewing_history) and st.session_state.analysis_results:
            try: results_list = sorted(st.session_state.analysis_results, key=lambda x: x.get('Question Number', float('inf')))
            except Exception as sort_err: st.warning(f"Sort error: {sort_err}"); results_list = st.session_state.analysis_results

            # --- Scatter Plot ---
            try:
                plot_data = [{'Question Number': item.get('Question Number', 0), 'Number of Evidence Items': len(item.get('Evidence', [])), 'Question Category': item.get('Question Category', 'Uncategorized'), 'Question': item.get('Question', 'N/A')} for item in results_list]
                if plot_data:
                    df_plot = pd.DataFrame(plot_data)
                    with st.expander("📊 Evidence Count Analysis (Scatter Plot)", expanded=False):
                        fig = px.scatter(df_plot, x='Question Number', y='Number of Evidence Items', color='Question Category', title="Number of Evidence Clauses Found per Question", labels={'Number of Evidence Items': 'Evidence Count'}, hover_data=['Question'])
                        fig.update_traces(marker=dict(size=10, opacity=0.7), selector=dict(mode='markers'))
                        fig.update_layout(xaxis_title="Question Number", yaxis_title="Number of Evidence Clauses")
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Hover over points for question details.")
            except Exception as plot_err: st.warning(f"Plot error: {plot_err}"); print(f"Plotting Error: {plot_err}\n{traceback.format_exc()}")

            # --- Tabbed Results ---
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
                            with st.expander(f"**Q{q_num}:** {question_text}"):
                                st.markdown(f"**Answer:**"); st.markdown(f"> {result_item.get('Answer', 'N/A')}"); st.markdown("---")
                                evidence_list = result_item.get('Evidence', [])
                                if evidence_list:
                                    st.markdown("**Evidence:**")
                                    for ev_index, evidence_item in enumerate(evidence_list):
                                        clause_ref = evidence_item.get('Clause Reference', 'N/A'); search_text = evidence_item.get('Searchable Clause Text', None); clause_wording = evidence_item.get('Clause Wording', 'N/A'); base_key = f"ev_{category}_{q_num}_{index}_{ev_index}"
                                        ev_cols = st.columns([3, 1])
                                        with ev_cols[0]:
                                            if search_text:
                                                button_key = f"search_btn_{base_key}"; button_label = f"Clause: **{clause_ref or 'Link'}** (Find & View)"
                                                if st.button(button_label, key=button_key, help=f"Search for '{clause_ref or 'evidence'}'"): st.session_state.search_trigger = {'text': search_text, 'ref': clause_ref}; st.session_state.last_search_result = None; st.rerun()
                                            elif clause_ref != 'N/A': st.markdown(f"- Clause: **{clause_ref}** (No searchable text)")
                                            else: st.caption("No clause reference.")
                                        with ev_cols[1]:
                                            if clause_wording != 'N/A':
                                                toggle_key = f"toggle_wording_{base_key}"; show_wording = st.toggle("Show Wording", key=toggle_key, value=st.session_state.show_wording_states.get(toggle_key, False), help="Show/hide extracted wording.")
                                                if show_wording != st.session_state.show_wording_states.get(toggle_key, False): st.session_state.show_wording_states[toggle_key] = show_wording; st.rerun()
                                        if st.session_state.show_wording_states.get(f"toggle_wording_{base_key}", False): st.text_area(f"Wording for '{clause_ref}':", value=clause_wording, height=150, disabled=True, key=f"wording_area_{base_key}")
                                        st.markdown("---")
                                else: st.markdown("**Evidence:** None provided.")
                                st.markdown("---"); st.markdown("**Answer Justification:**")
                                justification_text = result_item.get('Answer Justification', ''); just_key = f"justification_{category}_{q_num}_{index}"
                                st.text_area("Justification", value=justification_text, height=100, disabled=True, label_visibility="collapsed", key=just_key)
            else: st.warning("No results to display in tabs."); st.json(results_list or "{}")

            # --- Excel Download ---
            st.sidebar.markdown("---"); st.sidebar.markdown("## Export Results")
            if st.sidebar.button("Prepare Data for Excel Download", key="prep_excel", use_container_width=True):
                st.session_state.excel_data = None; excel_prep_status = st.sidebar.empty(); excel_prep_status.info("Preparing Excel...")
                try:
                    excel_rows = []; file_name_for_excel = st.session_state.history_filename if st.session_state.viewing_history else st.session_state.get("current_filename", "N/A"); gen_time_for_excel = st.session_state.history_timestamp if st.session_state.viewing_history else (results_list[0].get("Generation Time", "N/A") if results_list else "N/A")
                    for item in results_list:
                        references = []; first_search_text = "N/A"; evidence = item.get("Evidence")
                        if evidence:
                            references = [str(ev.get("Clause Reference", "N/A")) for ev in evidence if isinstance(ev, dict)]
                            first_search_text = evidence[0].get("Searchable Clause Text", "N/A") if isinstance(evidence[0], dict) else "N/A"
                        excel_row = { "File Name": file_name_for_excel, "Generation Time": gen_time_for_excel, "Question Number": item.get("Question Number"), "Question Category": item.get("Question Category", "Uncategorized"), "Question": item.get("Question", "N/A"), "Answer": item.get("Answer", "N/A"), "Answer Justification": item.get("Answer Justification", "N/A"), "Clause References (Concatenated)": "; ".join(references) if references else "N/A", "First Searchable Clause Text": first_search_text }
                        excel_rows.append(excel_row)
                    if not excel_rows: excel_prep_status.warning("No data."); st.session_state.excel_data = None
                    else:
                        df_excel = pd.DataFrame(excel_rows); final_columns = [col for col in EXCEL_COLUMN_ORDER if col in df_excel.columns]; extra_cols = [col for col in df_excel.columns if col not in final_columns]; df_excel = df_excel[final_columns + extra_cols]
                        output = io.BytesIO();
                        with pd.ExcelWriter(output, engine='openpyxl') as writer: df_excel.to_excel(writer, index=False, sheet_name='Analysis Results')
                        st.session_state.excel_data = output.getvalue(); excel_prep_status.success("✅ Excel ready!"); time.sleep(2); excel_prep_status.empty()
                except Exception as excel_err: excel_prep_status.error(f"Excel Error: {excel_err}"); print(traceback.format_exc())
            if st.session_state.excel_data:
                dl_fn = st.session_state.history_filename if st.session_state.viewing_history else st.session_state.get("current_filename", "analysis_results")
                safe_fn = re.sub(r'[^\w\s-]', '', os.path.splitext(dl_fn)[0]).strip().replace(' ', '_'); download_filename = f"Analysis_{safe_fn}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                st.sidebar.download_button(label="📥 Download Results as Excel", data=st.session_state.excel_data, file_name=download_filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_excel_final", use_container_width=True)

        # --- Fallback messages ---
        elif st.session_state.analysis_complete and not st.session_state.analysis_results: st.info("Analysis completed, but no valid results generated.")
        elif st.session_state.processing_in_progress: st.info("Analysis running...")
        elif not st.session_state.analysis_complete and st.session_state.pdf_bytes is not None and not st.session_state.viewing_history: st.info("PDF loaded. Configure and click 'Analyse Document'.")


    # --- Column 2: PDF Viewer ---
    with col2:
        st.markdown('<div class="sticky-viewer-content">', unsafe_allow_html=True)
        st.subheader("📄 PDF Viewer"); viewer_status_placeholder = st.empty()
        if st.session_state.search_trigger:
            search_info = st.session_state.pop('search_trigger'); # Pop trigger
            with st.spinner(f"🔎 Searching for: '{search_info['ref']}'..."): found_page, instances, term_used, search_status, all_findings = find_text_in_pdf(st.session_state.pdf_bytes, search_info['text'])
            if found_page: st.session_state.last_search_result = {'page': found_page, 'instances': instances, 'term': term_used, 'status': search_status, 'ref': search_info['ref'], 'all_findings': all_findings}; st.session_state.current_page = found_page; viewer_status_placeholder.empty(); st.rerun()
            else: st.session_state.last_search_result = None; viewer_status_placeholder.error(search_status)

        if st.session_state.pdf_display_ready:
            try: doc = fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf"); total_pages = doc.page_count; doc.close()
            except Exception as pdf_load_err: st.error(f"PDF load error: {pdf_load_err}"); total_pages = 1; st.session_state.current_page = 1
            current_display_page = max(1, min(int(st.session_state.get('current_page', 1)), total_pages))
            if current_display_page != st.session_state.get('current_page'): st.session_state.current_page = current_display_page

            nav_cols = st.columns([1, 3, 1])
            if nav_cols[0].button("⬅️ Prev", key="prev_page", disabled=(current_display_page <= 1), use_container_width=True): st.session_state.current_page -= 1; st.session_state.last_search_result = None; st.rerun()
            if nav_cols[2].button("Next ➡️", key="next_page", disabled=(current_display_page >= total_pages), use_container_width=True): st.session_state.current_page += 1; st.session_state.last_search_result = None; st.rerun()

            page_info_text = f"Page {current_display_page} of {total_pages}"; search_ref_disp = None
            if st.session_state.last_search_result and st.session_state.last_search_result['page'] == current_display_page:
                search_ref_disp = st.session_state.last_search_result.get('ref', 'Search')
                page_info_text += f" (🎯 Ref: '{search_ref_disp}')" if not st.session_state.last_search_result.get('all_findings') else f" (🎯 Multi: '{search_ref_disp}')"
            nav_cols[1].markdown(f"<div style='text-align: center; padding-top: 0.5rem;'>{page_info_text}</div>", unsafe_allow_html=True)

            if st.session_state.last_search_result and st.session_state.last_search_result.get('all_findings'):
                multi_findings = st.session_state.last_search_result['all_findings']; found_pages = sorted([f[0] for f in multi_findings])
                status_msg = st.session_state.last_search_result.get('status', '');
                if status_msg: viewer_status_placeholder.info(status_msg)
                st.write("Jump to other matches:")
                btn_cols = st.columns(min(len(found_pages), 5))
                current_sr = st.session_state.last_search_result.get('ref', 'unk')
                for idx, p_num in enumerate(found_pages):
                    col_idx = idx % len(btn_cols)
                    if btn_cols[col_idx].button(f"Pg {p_num}", key=f"jump_{p_num}_{current_sr}", disabled=(p_num == current_display_page), use_container_width=True):
                        st.session_state.current_page = p_num; new_instances = next((inst for pg, inst in multi_findings if pg == p_num), None)
                        st.session_state.last_search_result['instances'] = new_instances; st.session_state.last_search_result['page'] = p_num
                        term_d = st.session_state.last_search_result.get('term', 'text')
                        st.session_state.last_search_result['status'] = f"✅ Viewing match for '{term_d}' on page {p_num}."
                        st.session_state.last_search_result['all_findings'] = None # Clear multi-findings after jump
                        st.rerun()

            st.markdown("---")
            highlights_to_apply = None; render_status_override = None
            if st.session_state.last_search_result and st.session_state.last_search_result.get('page') == current_display_page:
                highlights_to_apply = st.session_state.last_search_result.get('instances')
                if not st.session_state.last_search_result.get('all_findings'): render_status_override = st.session_state.last_search_result.get('status')

            image_bytes, render_status = render_pdf_page_to_image(st.session_state.pdf_bytes, current_display_page, highlight_instances=highlights_to_apply, dpi=150)

            if image_bytes:
                st.image(image_bytes, caption=f"Page {current_display_page}", use_container_width=True)
                final_status = render_status_override if render_status_override else render_status
                if not (st.session_state.last_search_result and st.session_state.last_search_result.get('all_findings')):
                    if final_status:
                         if "✅" in final_status or "Found" in final_status : viewer_status_placeholder.success(final_status)
                         elif "⚠️" in final_status or "warning" in final_status.lower() or "multiple" in final_status.lower(): viewer_status_placeholder.warning(final_status)
                         elif "❌" in final_status or "error" in final_status.lower(): viewer_status_placeholder.error(final_status)
                         else: viewer_status_placeholder.caption(final_status)
                    else: viewer_status_placeholder.empty()
            else: viewer_status_placeholder.error(f"Failed to render page {current_display_page}. {render_status or ''}")
        else: st.info("Preparing viewer..."); viewer_status_placeholder.empty()
        st.markdown('</div>', unsafe_allow_html=True)

# --- Fallback message ---
elif not st.session_state.pdf_bytes and not st.session_state.viewing_history:
     st.info("⬆️ Upload a PDF or load from History.")

# # app.py
# # --- COMPLETE FILE (v5 - Added History Loading & GCS Integration) ---

# import streamlit as st
# import pandas as pd
# import os
# import io
# import base64
# import fitz   # PyMuPDF
# import google.generativeai as genai
# from google.generativeai import types
# import sys
# import json
# import re
# import time
# from datetime import datetime
# import traceback
# from collections import defaultdict # To group questions by category
# import plotly.express as px
# import google.cloud.firestore
# import google.cloud.storage # <-- IMPORT GCS
# import google.oauth2.service_account
# from google.api_core.exceptions import NotFound # For GCS blob check

# # --- 1. SET PAGE CONFIG (MUST BE FIRST st COMMAND) ---
# st.set_page_config(
#     layout="wide",
#     page_title="JASPER - Extraction and Review",
#     page_icon="📄" # Optional: Set an emoji icon
# )

# # --- Inject custom CSS for sticky column (AFTER set_page_config) ---
# # Make sure the top value aligns with Streamlit's header height (approx 50-60px)
# st.markdown("""
# <style>
#     /* Define a class for the sticky container */
#     .sticky-viewer-content {
#         position: sticky;
#         top: 55px; /* Adjust vertical offset from top */
#         z-index: 101; /* Ensure it's above other elements */
#         padding-bottom: 1rem; /* Add some space at the bottom */
#         /* Ensure background color matches theme for sticky element */
#         background-color: var(--streamlit-background-color);
#     }
# </style>
# """, unsafe_allow_html=True)

# # --- Load credentials and Initialize Clients ---
# try:
#     # Firestore
#     firestore_key_dict = st.secrets["firestore"]
#     firestore_creds = google.oauth2.service_account.Credentials.from_service_account_info(firestore_key_dict)
#     db = google.cloud.firestore.Client(credentials=firestore_creds)

#     # Google Cloud Storage
#     gcs_key_dict = st.secrets.get("gcs", firestore_key_dict) # Reuse Firestore creds if GCS specific aren't provided
#     gcs_creds = google.oauth2.service_account.Credentials.from_service_account_info(gcs_key_dict)
#     storage_client = google.cloud.storage.Client(credentials=gcs_creds)
#     GCS_BUCKET_NAME = st.secrets["gcs_config"]["bucket_name"]
#     GCS_PDF_FOLDER = "analysis_pdfs" # Define a folder within the bucket

# except KeyError as e:
#     st.error(f"❌ Configuration Error: Missing key '{e}' in Streamlit secrets (`secrets.toml`). Please check your configuration.")
#     st.stop()
# except Exception as e:
#     st.error(f"❌ Failed to initialize cloud clients: {e}")
#     print(traceback.format_exc())
#     st.stop()

# # --- 2. Configuration & Setup ---
# MODEL_NAME = "gemini-2.5-pro-preview-03-25"
# MAX_VALIDATION_RETRIES = 1
# RETRY_DELAY_SECONDS = 3
# PROMPT_FILE = "prompt.txt"
# LOGO_FILE = "jasper-logo-1.png" # Ensure this file exists if used later
# SEARCH_FALLBACK_MIN_LENGTH = 20
# SEARCH_PREFIX_MIN_WORDS = 4

# # --- Get the absolute path to the directory containing app.py ---
# APP_DIR = os.path.dirname(os.path.abspath(__file__))

# # --- Load Prompt Text from File ---
# try:
#     prompt_path = os.path.join(APP_DIR, PROMPT_FILE)
#     with open(prompt_path, 'r', encoding='utf-8') as f:
#         full_prompt_text = f.read()
# except FileNotFoundError:
#     st.error(f"Error: Prompt file '{PROMPT_FILE}' not found in the application directory ({APP_DIR}). Please ensure it exists.")
#     st.stop()
# except Exception as e:
#     st.error(f"Error reading prompt file '{PROMPT_FILE}': {e}")
#     st.stop()

# # --- Schema Definition (Version 3 - Search Text) ---
# ai_response_schema_dict = {
#   "type": "array", "description": "List of question analysis results.",
#   "items": {
#     "type": "object", "properties": {
#       "Question Number": {"type": "integer"}, "Question Category": {"type": "string"}, "Question": {"type": "string"}, "Answer": {"type": "string"}, "Answer Justification": {"type": "string"},
#       "Evidence": { "type": "array", "items": { "type": "object", "properties": { "Clause Reference": {"type": "string"}, "Clause Wording": {"type": "string"}, "Searchable Clause Text": {"type": "string"} }, "required": ["Clause Reference", "Clause Wording", "Searchable Clause Text"] } }
#     }, "required": ["Question Number", "Question Category", "Question", "Answer", "Answer Justification", "Evidence"]
#   }
# }
# AI_REQUIRED_KEYS = set(ai_response_schema_dict['items']['required'])
# AI_EVIDENCE_REQUIRED_KEYS = set(ai_response_schema_dict['items']['properties']['Evidence']['items']['required'])

# # --- Excel Column Order ---
# EXCEL_COLUMN_ORDER = [ "File Name", "Generation Time", "Question Number", "Question Category", "Question", "Answer", "Answer Justification", "Clause References (Concatenated)", "First Searchable Clause Text"]

# # --- Section Definitions ---
# # Define ALL possible sections here. Selection happens in the UI.
# ALL_SECTIONS = {
#     "agreement_details": (1, 4),
#     "eligibility_part_1": (5, 20),
#     "eligibility_part_2": (21, 34),
#     "eligibility_summary": (35, 36),
#     "confidentiality": (37, 63),
#     "additional_borrowers": (64, 66),
#     "interest_rate_provisions": (67, 71),
#     "prepayment_fee": (72, 78)
# }
# # This dictionary holds the master definition. Helper functions will use this.

# # --- System Instruction ---
# system_instruction_text = """You are an AI assistant specialized in analyzing legal facility agreements. Carefully read the provided document and answer the specific questions listed in the user prompt for the designated section only. Adhere strictly to the requested JSON output schema. Prioritize accuracy and extract evidence directly from the text."""

# # --- 3. Helper Function Definitions ---

# def filter_prompt_by_section(initial_full_prompt, section):
#     """Filters the full prompt to include only questions for the specified section."""
#     if section not in ALL_SECTIONS: raise ValueError(f"Invalid section specified: {section}")
#     start_q, end_q = ALL_SECTIONS[section]; questions_start_marker = "**Questions to Answer:**"; questions_end_marker = "**Final Instruction:**"
#     try: start_index = initial_full_prompt.index(questions_start_marker); end_index = initial_full_prompt.index(questions_end_marker)
#     except ValueError: raise ValueError(f"Could not find '{questions_start_marker}' or '{questions_end_marker}' markers in the prompt file.")
#     prompt_header = initial_full_prompt[:start_index]; full_questions_block = initial_full_prompt[start_index + len(questions_start_marker):end_index].strip(); prompt_footer = initial_full_prompt[end_index:]
#     question_entries = re.split(r'\n\s*(?=\d+\.\s*?\*\*Question Category:)', full_questions_block)
#     filtered_question_texts = []
#     for entry in question_entries:
#         entry = entry.strip()
#         if not entry: continue
#         # More robust regex for question number extraction
#         match = re.match(r'^\s*(\d+)\.', entry)
#         if match:
#             # This try-except block looks syntactically correct
#             try:
#                 q_num = int(match.group(1))
#                 # Check if the question number is within the desired range for the section
#                 if start_q <= q_num <= end_q:
#                     filtered_question_texts.append(entry)
#             except ValueError:
#                 # If int() fails (e.g., regex matched something unexpected),
#                 # simply skip this entry and continue to the next iteration
#                 # of the 'for' loop.
#                 continue
#     if not filtered_question_texts: raise ValueError(f"No questions found for section '{section}' (range {start_q}-{end_q}) in the parsed questions block. Check prompt formatting and section ranges in ALL_SECTIONS.")
#     filtered_questions_string = "\n\n".join(filtered_question_texts)
#     section_note = f"\n\n**Current Focus:** You MUST answer ONLY the questions listed below for the '{section.upper()}' section (Questions {start_q}-{end_q}). Ignore all other questions for this specific task.\n"
#     final_prompt_for_api = f"{prompt_header}{section_note}{questions_start_marker}\n\n{filtered_questions_string}\n\n{prompt_footer}"
#     return final_prompt_for_api

# def validate_ai_data(data, section_name):
#     """Validates AI response against the schema. Returns (validated_data, issues_list)."""
#     if not isinstance(data, list): return None, [f"CRITICAL VALIDATION ERROR: Response for section '{section_name}' is not a list."]
#     validated_data = []; issues_list = []
#     if section_name not in ALL_SECTIONS: issues_list.append(f"Warning: Section '{section_name}' not found in master section definitions (ALL_SECTIONS). Cannot validate question range."); expected_q_nums_in_section = set()
#     else: expected_q_nums_in_section = set(range(ALL_SECTIONS[section_name][0], ALL_SECTIONS[section_name][1] + 1))
#     found_q_nums = set()
#     for index, item in enumerate(data):
#         q_num = item.get('Question Number'); q_num_str = f"Q#{q_num}" if q_num is not None else f"Item Index {index}"; is_outer_valid = True
#         if not isinstance(item, dict): issues_list.append(f"{q_num_str}: Item is not a dictionary."); is_outer_valid = False; continue
#         if isinstance(q_num, int): found_q_nums.add(q_num)
#         else: issues_list.append(f"Item Index {index}: 'Question Number' is missing or not an integer."); is_outer_valid = False
#         missing_outer_keys = AI_REQUIRED_KEYS - set(item.keys())
#         if missing_outer_keys: issues_list.append(f"{q_num_str}: Missing required top-level keys: {missing_outer_keys}"); is_outer_valid = False
#         evidence_list = item.get("Evidence")
#         if not isinstance(evidence_list, list):
#             if "Evidence" in AI_REQUIRED_KEYS: issues_list.append(f"{q_num_str}: 'Evidence' field is not a list (found type: {type(evidence_list).__name__})."); is_outer_valid = False
#         else:
#             for ev_index, ev_item in enumerate(evidence_list):
#                 ev_id_str = f"Ev[{ev_index}]"
#                 if not isinstance(ev_item, dict): issues_list.append(f"{q_num_str} {ev_id_str}: Evidence item is not a dictionary."); is_outer_valid = False; continue
#                 missing_ev_keys = AI_EVIDENCE_REQUIRED_KEYS - set(ev_item.keys())
#                 if missing_ev_keys: issues_list.append(f"{q_num_str} {ev_id_str}: Missing required evidence keys: {missing_ev_keys}"); is_outer_valid = False
#                 for key, expected_type in [("Clause Reference", str), ("Clause Wording", str), ("Searchable Clause Text", str)]:
#                      if key in AI_EVIDENCE_REQUIRED_KEYS and key in ev_item and not isinstance(ev_item.get(key), expected_type): issues_list.append(f"{q_num_str} {ev_id_str}: Key '{key}' has incorrect type (expected {expected_type.__name__}, got {type(ev_item.get(key)).__name__})."); is_outer_valid = False
#         if is_outer_valid: validated_data.append(item)
#     if expected_q_nums_in_section:
#         missing_q_nums = expected_q_nums_in_section - found_q_nums
#         if missing_q_nums: issues_list.append(f"Section '{section_name}': Missing answers for expected Question Numbers: {sorted(list(missing_q_nums))}")
#         unexpected_q_nums = found_q_nums - expected_q_nums_in_section
#         if unexpected_q_nums: issues_list.append(f"Section '{section_name}': Received unexpected Question Numbers: {sorted(list(unexpected_q_nums))}")
#     if issues_list: issues_list.insert(0, f"Validation Issues Found [Section: {section_name}] ({len(validated_data)} items passed validation):")
#     if validated_data is None and isinstance(data, list): return [], issues_list
#     elif validated_data is None: return None, issues_list
#     else: return validated_data, issues_list

# def generate_section_analysis(section, uploaded_file_ref, status_placeholder, api_key_to_use):
#     """Generates analysis for a section using a specific API key."""
#     try: genai.configure(api_key=api_key_to_use)
#     except Exception as config_err: status_placeholder.error(f"❌ Invalid API Key provided or configuration failed: {config_err}"); return None, "Failed", [f"Invalid API Key or config error: {config_err}"]
#     status_placeholder.info(f"🔄 Starting Analysis: {section}..."); section_warnings = []
#     try:
#         model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction_text)
#         generation_config = types.GenerationConfig(response_mime_type="application/json", response_schema=ai_response_schema_dict, temperature=0.0,)
#         safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
#         final_validated_data = None
#         for attempt in range(1, MAX_VALIDATION_RETRIES + 2):
#             if attempt > 1: status_placeholder.info(f"⏳ Retrying generation/validation for '{section}' (Attempt {attempt}/{MAX_VALIDATION_RETRIES+1})..."); time.sleep(RETRY_DELAY_SECONDS)
#             try:
#                 prompt_for_api = filter_prompt_by_section(full_prompt_text, section)
#                 if not uploaded_file_ref or not hasattr(uploaded_file_ref, 'name'): raise ValueError("Invalid or missing uploaded file reference for GenAI call.")
#                 contents = [uploaded_file_ref, prompt_for_api]
#                 status_placeholder.info(f"🧠 Calling AI for '{section}' (Attempt {attempt})...")
#                 response = model.generate_content(contents=contents, generation_config=generation_config, safety_settings=safety_settings, request_options={'timeout': 600})
#                 parsed_ai_data = None; validated_ai_data = None; validation_issues = []
#                 status_placeholder.info(f"🔍 Processing response for '{section}'...")
#                 if response.parts:
#                     full_response_text = response.text
#                     try:
#                         match = re.search(r"```json\s*([\s\S]*?)\s*```", full_response_text, re.IGNORECASE); json_text = match.group(1).strip() if match else full_response_text.strip()
#                         if not json_text: raise json.JSONDecodeError("Extracted JSON content is empty.", json_text, 0)
#                         parsed_ai_data = json.loads(json_text)
#                         status_placeholder.info(f"✔️ Validating structure for '{section}'...")
#                         validated_ai_data, validation_issues = validate_ai_data(parsed_ai_data, section); section_warnings.extend(validation_issues)
#                         if validated_ai_data is not None and len(validated_ai_data) > 0: final_validated_data = validated_ai_data; status_placeholder.info(f"✅ Validation successful for '{section}'."); break
#                         elif validated_ai_data is not None and len(validated_ai_data) == 0 and not validation_issues: status_placeholder.warning(f"⚠️ AI returned an empty list for '{section}' (Attempt {attempt}). Check prompt/section definition."); section_warnings.append(f"AI returned empty list for '{section}'.")
#                         else: error_msg = f"Validation failed for '{section}'. Issues: {validation_issues}"; status_placeholder.warning(f"⚠️ {error_msg} (Attempt {attempt}).");
#                         if validated_ai_data is None: section_warnings.append(f"CRITICAL validation error for '{section}'.")
#                     except json.JSONDecodeError as json_err: error_msg = f"JSON Decode Error on attempt {attempt} for '{section}': {json_err}."; st.error(error_msg); st.code(full_response_text, language='text'); section_warnings.append(error_msg)
#                     except Exception as parse_validate_err: error_msg = f"Unexpected Error during parsing/validation on attempt {attempt} for '{section}': {type(parse_validate_err).__name__}: {parse_validate_err}"; st.error(error_msg); section_warnings.append(error_msg); print(traceback.format_exc())
#                 else:
#                     block_reason = "Unknown"; block_message = "N/A"; finish_reason = "Unknown"
#                     try:
#                         if response.prompt_feedback: block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown'); block_reason = block_reason.name if hasattr(block_reason, 'name') else str(block_reason); block_message = response.prompt_feedback.block_reason_message or "N/A"
#                         if response.candidates: finish_reason = getattr(response.candidates[0], 'finish_reason', 'Unknown'); finish_reason = finish_reason.name if hasattr(finish_reason, 'name') else str(finish_reason)
#                     except AttributeError: pass
#                     if finish_reason == "SAFETY": warn_msg = f"API Response Blocked (Attempt {attempt}, Section: {section}): Reason: SAFETY. Detail: {block_reason}. Message: {block_message}"; st.error(warn_msg)
#                     elif finish_reason == "RECITATION": warn_msg = f"API Response Potentially Blocked (Attempt {attempt}, Section: {section}): Finish Reason: RECITATION. Block Reason: {block_reason}."; st.warning(warn_msg)
#                     elif finish_reason == "STOP" and not final_validated_data: warn_msg = f"API Response Ended (Attempt {attempt}, Section: {section}): Finish Reason: STOP, but no valid data parsed yet."; st.info(warn_msg)
#                     else: warn_msg = f"API Issue (Attempt {attempt}, Section: {section}): Finish Reason: {finish_reason}. Block Reason: {block_reason}. Response may be incomplete or empty."; st.warning(warn_msg)
#                     section_warnings.append(warn_msg)
#             except types.StopCandidateException as sce: error_msg = f"Generation Stopped Error (Attempt {attempt}, Section: {section}): {sce}."; st.error(error_msg); section_warnings.append(error_msg); print(traceback.format_exc())
#             except Exception as e: error_msg = f"Processing Error during API call/prompt generation (Attempt {attempt}, Section: {section}): {type(e).__name__}: {e}"; st.error(error_msg); section_warnings.append(error_msg); section_warnings.append("Traceback logged to console."); print(traceback.format_exc())
#             if final_validated_data is not None: break
#         if final_validated_data is not None: status_placeholder.success(f"✅ Analysis completed successfully for: {section}."); return final_validated_data, "Success", section_warnings
#         else: status_placeholder.error(f"❌ Analysis failed for: {section} after {attempt} attempts."); section_warnings.append(f"Failed to get valid response for section '{section}' after {MAX_VALIDATION_RETRIES + 1} attempts."); return None, "Failed", section_warnings
#     except Exception as outer_err: error_msg = f"Critical Error during setup or execution for section '{section}': {type(outer_err).__name__}: {outer_err}"; st.error(error_msg); section_warnings.append(error_msg); section_warnings.append("Traceback logged to console."); print(traceback.format_exc()); status_placeholder.error(f"❌ Critical failure processing section: {section}."); return None, "Failed", section_warnings

# @st.cache_data(show_spinner=False)
# def find_text_in_pdf(_pdf_bytes, search_text):
#     """Searches PDF. Returns (first_page_found, instances_on_first_page, term_used, status_msg, all_findings)"""
#     if not _pdf_bytes or not search_text: return None, None, None, "Invalid input (PDF bytes or search text missing).", None
#     doc = None; search_text_cleaned = search_text.strip(); words = search_text_cleaned.split(); num_words = len(words)
#     search_attempts = []
#     # --- Build Search Terms List (Prioritized) ---
#     if num_words >= SEARCH_PREFIX_MIN_WORDS and num_words > 5: term_10 = ' '.join(words[:10]); search_attempts.append({'term': term_10, 'desc': "first 10 words"})
#     # Corrected logic for term_5
#     if num_words >= SEARCH_PREFIX_MIN_WORDS:
#         term_5 = ' '.join(words[:5])
#         if not search_attempts or term_5 != search_attempts[0]['term']: search_attempts.append({'term': term_5, 'desc': "first 5 words"})
#     term_full = search_text_cleaned;
#     if term_full and not any(term_full == a['term'] for a in search_attempts): search_attempts.append({'term': term_full, 'desc': "full text"})
#     sentences = re.split(r'(?<=[.?!])\s+', term_full); term_sentence = sentences[0].strip() if sentences else ""
#     if len(term_sentence) >= SEARCH_FALLBACK_MIN_LENGTH and not any(term_sentence == a['term'] for a in search_attempts): search_attempts.append({'term': term_sentence, 'desc': "first sentence fallback"})
#     # --- Execute Search Attempts ---
#     try:
#         doc = fitz.open(stream=_pdf_bytes, filetype="pdf"); search_flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES
#         for attempt in search_attempts:
#             term = attempt['term']; desc = attempt['desc']; findings_for_term = []
#             for page_index in range(doc.page_count):
#                 page = doc.load_page(page_index); instances = page.search_for(term, flags=search_flags, quads=False)
#                 if instances: findings_for_term.append((page_index + 1, instances))
#             if findings_for_term:
#                 doc.close(); first_page_found = findings_for_term[0][0]; instances_on_first_page = findings_for_term[0][1]
#                 if len(findings_for_term) == 1: status = f"✅ Found using '{desc}' on page {first_page_found}."; return first_page_found, instances_on_first_page, term, status, None
#                 else: pages_found = sorted([f[0] for f in findings_for_term]); status = f"⚠️ Found matches using '{desc}' on multiple pages: {pages_found}. Showing first match on page {first_page_found}."; return first_page_found, instances_on_first_page, term, status, findings_for_term
#         doc.close(); tried_descs = [a['desc'] for a in search_attempts]; return None, None, None, f"❌ Text not found (tried methods: {', '.join(tried_descs)}).", None
#     except Exception as e:
#         if doc: doc.close(); print(f"ERROR searching PDF: {e}\n{traceback.format_exc()}"); return None, None, None, f"❌ Error during PDF search: {e}", None

# # REMOVED @st.cache_data from this function due to UnhashableParamError with fitz.Rect
# def render_pdf_page_to_image(_pdf_bytes, page_number, highlight_instances=None, dpi=150):
#     """Renders PDF page to PNG image bytes, applying highlights. Returns (image_bytes, status_msg)."""
#     if not _pdf_bytes or page_number < 1: return None, "Invalid input for rendering (PDF bytes missing or invalid page number)."
#     doc = None; image_bytes = None; render_status_message = f"Rendered page {page_number}."
#     try:
#         doc = fitz.open(stream=_pdf_bytes, filetype="pdf"); page_index = page_number - 1
#         if page_index < 0 or page_index >= doc.page_count: doc.close(); return None, f"Page number {page_number} is out of range (Total pages: {doc.page_count})."
#         page = doc.load_page(page_index); highlight_applied_count = 0
#         if highlight_instances:
#             try:
#                 for inst in highlight_instances:
#                     if isinstance(inst, (fitz.Rect, fitz.Quad)):
#                         highlight = page.add_highlight_annot(inst)
#                         if highlight: highlight.set_colors(stroke=fitz.utils.getColor("yellow")); highlight.set_opacity(0.4); highlight.update(); highlight_applied_count += 1
#                         else: print(f"WARN: Failed to add highlight annotation for instance: {inst} on page {page_number}")
#                 if highlight_applied_count > 0: render_status_message = f"Rendered page {page_number} with {highlight_applied_count} highlight(s)."
#                 elif highlight_instances: render_status_message = f"Rendered page {page_number}, but no valid highlights applied from provided instances."
#             except Exception as highlight_err: print(f"ERROR applying highlights on page {page_number}: {highlight_err}\n{traceback.format_exc()}"); render_status_message = f"⚠️ Error applying highlights: {highlight_err}"
#         pix = page.get_pixmap(dpi=dpi, alpha=False); image_bytes = pix.tobytes("png")
#     except Exception as e: print(f"ERROR rendering page {page_number}: {e}\n{traceback.format_exc()}"); render_status_message = f"❌ Error rendering page {page_number}: {e}"; image_bytes = None
#     finally:
#         if doc: doc.close()
#     return image_bytes, render_status_message

# def upload_to_gcs(bucket_name, source_bytes, destination_blob_name, status_placeholder=None):
#     """Uploads bytes to GCS bucket."""
#     if status_placeholder: status_placeholder.info(f"☁️ Uploading PDF to Google Cloud Storage...")
#     try:
#         bucket = storage_client.bucket(bucket_name)
#         blob = bucket.blob(destination_blob_name)
#         blob.upload_from_string(source_bytes, content_type='application/pdf')
#         if status_placeholder: status_placeholder.info(f"☁️ PDF successfully uploaded to gs://{bucket_name}/{destination_blob_name}")
#         print(f"Uploaded to gs://{bucket_name}/{destination_blob_name}")
#         return f"gs://{bucket_name}/{destination_blob_name}"
#     except Exception as e:
#         if status_placeholder: status_placeholder.error(f"❌ GCS Upload Failed: {e}")
#         print(f"GCS Upload Error: {e}\n{traceback.format_exc()}")
#         raise # Re-raise the exception to be caught by the main analysis loop

# def download_from_gcs(bucket_name, source_blob_name):
#     """Downloads a blob from the bucket."""
#     try:
#         bucket = storage_client.bucket(bucket_name)
#         blob = bucket.blob(source_blob_name)
#         pdf_bytes = blob.download_as_bytes()
#         return pdf_bytes
#     except NotFound:
#         st.error(f"❌ Error: PDF file not found in Google Cloud Storage at gs://{bucket_name}/{source_blob_name}")
#         return None
#     except Exception as e:
#         st.error(f"❌ Failed to download PDF from GCS: {e}")
#         print(f"GCS Download Error: {e}\n{traceback.format_exc()}")
#         return None

# def reset_app_state():
#     """Resets the session state for a new analysis or clearing history."""
#     st.session_state.pdf_bytes = None
#     st.session_state.pdf_display_ready = False
#     st.session_state.analysis_results = None
#     st.session_state.analysis_complete = False
#     st.session_state.processing_in_progress = False
#     st.session_state.current_page = 1
#     st.session_state.run_status_summary = []
#     st.session_state.excel_data = None
#     st.session_state.search_trigger = None
#     st.session_state.last_search_result = None
#     st.session_state.show_wording_states = defaultdict(bool)
#     st.session_state.viewing_history = False # Ensure history view is exited
#     st.session_state.history_filename = None
#     st.session_state.history_timestamp = None
#     # Keep API key and selected sections if user wants to run again
#     # st.session_state.api_key = None # Optionally clear API key
#     # st.session_state.selected_sections_to_run = list(ALL_SECTIONS.keys())[:1] # Reset sections?

# # --- 4. Initialize Session State ---
# if 'show_wording_states' not in st.session_state: st.session_state.show_wording_states = defaultdict(bool)
# if 'current_page' not in st.session_state: st.session_state.current_page = 1
# if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
# if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None
# if 'pdf_display_ready' not in st.session_state: st.session_state.pdf_display_ready = False
# if 'processing_in_progress' not in st.session_state: st.session_state.processing_in_progress = False
# if 'analysis_complete' not in st.session_state: st.session_state.analysis_complete = False
# if 'run_key' not in st.session_state: st.session_state.run_key = 0
# if 'run_status_summary' not in st.session_state: st.session_state.run_status_summary = []
# if 'excel_data' not in st.session_state: st.session_state.excel_data = None
# if 'search_trigger' not in st.session_state: st.session_state.search_trigger = None
# if 'last_search_result' not in st.session_state: st.session_state.last_search_result = None
# if 'api_key' not in st.session_state: st.session_state.api_key = None
# if 'selected_sections_to_run' not in st.session_state: st.session_state.selected_sections_to_run = list(ALL_SECTIONS.keys())[:1] # Default to only first section for testing
# if 'load_history_id' not in st.session_state: st.session_state.load_history_id = None # For history loading trigger
# if 'viewing_history' not in st.session_state: st.session_state.viewing_history = False # Flag for history view mode
# if 'history_filename' not in st.session_state: st.session_state.history_filename = None # Store filename when viewing history
# if 'history_timestamp' not in st.session_state: st.session_state.history_timestamp = None # Store timestamp when viewing history


# # --- 5. Load History Data (if triggered) ---
# if st.session_state.load_history_id:
#     history_id = st.session_state.pop('load_history_id') # Get ID and clear trigger
#     st.info(f"📜 Loading historical analysis: {history_id}...")
#     try:
#         doc_ref = db.collection("analysis_runs").document(history_id)
#         doc_snapshot = doc_ref.get()

#         if doc_snapshot.exists:
#             run_data = doc_snapshot.to_dict()
#             gcs_pdf_path = run_data.get("gcs_pdf_path")
#             results = run_data.get("results")
#             filename = run_data.get("filename", "N/A")
#             timestamp = run_data.get("analysis_timestamp") # Firestore timestamp object
#             run_summary = run_data.get("run_status", []) # Load summary if saved

#             if not gcs_pdf_path:
#                  st.error("❌ History record is missing the PDF file path (gcs_pdf_path). Cannot load.")
#             elif not results:
#                  st.error("❌ History record is missing analysis results. Cannot load.")
#             else:
#                 # Download PDF from GCS
#                 if gcs_pdf_path.startswith("gs://"):
#                     path_parts = gcs_pdf_path[5:].split("/", 1)
#                     hist_bucket_name = path_parts[0]
#                     hist_blob_name = path_parts[1]
#                 else: # Assume path format is FOLDER/FILENAME.PDF
#                      hist_bucket_name = GCS_BUCKET_NAME # Use configured bucket
#                      hist_blob_name = gcs_pdf_path

#                 with st.spinner(f"Downloading PDF from gs://{hist_bucket_name}/{hist_blob_name}..."):
#                     pdf_bytes_from_hist = download_from_gcs(hist_bucket_name, hist_blob_name)

#                 if pdf_bytes_from_hist:
#                     # Reset state before loading historical data
#                     reset_app_state()

#                     # Load historical data into session state
#                     st.session_state.pdf_bytes = pdf_bytes_from_hist
#                     st.session_state.analysis_results = results
#                     st.session_state.run_status_summary = run_summary
#                     st.session_state.analysis_complete = True
#                     st.session_state.pdf_display_ready = True
#                     st.session_state.viewing_history = True
#                     st.session_state.history_filename = filename
#                     st.session_state.history_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp)
#                     st.session_state.current_page = 1 # Reset to first page

#                     st.success(f"✅ Successfully loaded history for '{filename}' ({st.session_state.history_timestamp}).")
#                     time.sleep(1) # Give user time to see message
#                     st.rerun() # Rerun to update UI immediately
#                 else:
#                     # Download failed, error shown by download_from_gcs
#                     st.session_state.viewing_history = False # Ensure we are not stuck in history mode
#         else:
#             st.error(f"❌ History record with ID '{history_id}' not found in database.")
#             st.session_state.viewing_history = False

#     except Exception as e:
#         st.error(f"❌ Error loading historical data: {e}")
#         print(f"History Load Error: {e}\n{traceback.format_exc()}")
#         st.session_state.viewing_history = False # Ensure we are not stuck in history mode


# # --- 6. Streamlit UI Logic ---
# st.title("JASPER - Just A Smart Platform for Extraction and Review")

# # --- Display History Mode Banner ---
# if st.session_state.viewing_history:
#     hist_ts_str = st.session_state.history_timestamp or "N/A"
#     st.info(f"📜 **Viewing Historical Analysis:** File: **{st.session_state.history_filename}** (Generated: {hist_ts_str})")
#     if st.button("⬅️ Exit History View / Start New Analysis", key="clear_history_view"):
#         reset_app_state()
#         st.rerun()
#     st.markdown("---") # Separator

# elif not st.session_state.analysis_complete: # Only show initial message if not viewing history and not complete
#     st.markdown("Upload a PDF agreement, **enter your Gemini API Key**, **select sections**, click 'Analyse'. Results grouped below. Click clause references to view & highlight.")

# # --- Sidebar Setup ---
# st.sidebar.markdown("## Controls")

# # --- Conditional Controls (Only show if NOT viewing history) ---
# if not st.session_state.viewing_history:
#     st.sidebar.markdown("### 1. API Key")
#     # --- API Key Input (in Sidebar) ---
#     api_key_input = st.sidebar.text_input("Enter your Google AI Gemini API Key", type="password", key="api_key_input", help="Your API key is used only for this session and is not stored.", value=st.session_state.get("api_key", ""))
#     if api_key_input != st.session_state.api_key: st.session_state.api_key = api_key_input; st.rerun()
#     if not st.session_state.api_key and not st.session_state.analysis_complete and not st.session_state.processing_in_progress : st.sidebar.warning("API Key required to run analysis.", icon="🔑")

#     st.sidebar.markdown("### 2. Upload PDF")
#     # --- File Upload (in Sidebar) ---
#     uploaded_file_obj = st.sidebar.file_uploader("Upload Facility Agreement PDF", type="pdf", key=f"pdf_uploader_{st.session_state.run_key}")
#     if uploaded_file_obj is not None:
#         new_file_bytes = uploaded_file_obj.getvalue()
#         if new_file_bytes != st.session_state.get('pdf_bytes'):
#              # Reset state completely when a new file is uploaded
#             reset_app_state()
#             st.session_state.pdf_bytes = new_file_bytes
#             st.session_state.pdf_display_ready = True
#             # Keep API key and section selection
#             st.session_state.api_key = api_key_input # Re-capture current key just in case
#             # st.session_state.selected_sections_to_run = selected_sections # Re-capture selected sections

#             st.toast("✅ New PDF file loaded. Viewer ready.", icon="📄")
#             st.rerun()
#     elif not st.session_state.pdf_bytes:
#          st.sidebar.info("Upload a PDF to enable analysis.")


#     st.sidebar.markdown("### 3. Select Sections")
#     # --- Section Selection for Analysis (NEW) ---
#     selected_sections = st.sidebar.multiselect("Choose sections:", options=list(ALL_SECTIONS.keys()), default=st.session_state.selected_sections_to_run, key="section_selector", help="Select which parts of the document you want to analyse in this run.")
#     if selected_sections != st.session_state.selected_sections_to_run: st.session_state.selected_sections_to_run = selected_sections; # No rerun needed just for selection change

#     st.sidebar.markdown("### 4. Run Analysis")
#     # --- Analysis Trigger (in Sidebar) ---
#     can_analyse = (st.session_state.pdf_bytes is not None and st.session_state.api_key is not None and not st.session_state.processing_in_progress and not st.session_state.analysis_complete and st.session_state.selected_sections_to_run)
#     analyse_button_tooltip = "Analysis complete for the current file." if st.session_state.analysis_complete else "Analysis is currently running." if st.session_state.processing_in_progress else "Upload a PDF file first." if not st.session_state.pdf_bytes else "Enter your Gemini API key first." if not st.session_state.api_key else "Select at least one section to analyse." if not st.session_state.selected_sections_to_run else "Start analyzing the selected sections"
#     if st.sidebar.button("✨ Analyse Document", key="analyse_button", disabled=not can_analyse, help=analyse_button_tooltip, use_container_width=True, type="primary"):
#         if not st.session_state.api_key: st.error("API Key is missing. Please enter it in the sidebar.")
#         elif not st.session_state.pdf_bytes: st.error("No PDF file uploaded. Please upload a file.")
#         elif not st.session_state.selected_sections_to_run: st.error("No sections selected for analysis. Please select sections in the sidebar.")
#         else:
#             st.session_state.processing_in_progress = True; st.session_state.analysis_complete = False; st.session_state.analysis_results = []
#             st.session_state.run_key += 1; st.session_state.run_status_summary = []; st.session_state.excel_data = None; st.session_state.search_trigger = None
#             st.session_state.last_search_result = None; st.session_state.show_wording_states = defaultdict(bool)
#             current_api_key = st.session_state.api_key; run_start_time = datetime.now(); run_timestamp_str = run_start_time.strftime("%Y-%m-%d %H:%M:%S")
#             # Use the actual UploadedFile object name if available, otherwise generate one
#             # Note: uploaded_file_obj might be None if analysis is somehow triggered without a new upload (shouldn't happen with guards)
#             # We need to access the name from the object *before* analysis starts, as it might be lost on rerun
#             if uploaded_file_obj:
#                  st.session_state.current_filename = uploaded_file_obj.name
#             elif not hasattr(st.session_state, 'current_filename') or not st.session_state.current_filename:
#                 # Fallback if name isn't captured (e.g. state loss?)
#                 st.session_state.current_filename = f"upload_{run_start_time.strftime('%Y%m%d%H%M%S')}.pdf"

#             base_file_name = st.session_state.current_filename # Use the stored filename

#             try: genai.configure(api_key=current_api_key); st.toast("API Key validated and configured.", icon="🔑")
#             except Exception as config_err: st.error(f"❌ Failed to configure Gemini API with provided key: {config_err}"); st.session_state.processing_in_progress = False; st.stop()
#             status_container = st.container(); progress_bar = status_container.progress(0, text="Initializing analysis..."); status_text = status_container.empty()
#             temp_dir = "temp_uploads"; safe_base_name = re.sub(r'[^\w\-.]', '_', base_file_name); temp_file_path = os.path.join(APP_DIR, temp_dir, f"{run_start_time.strftime('%Y%m%d%H%M%S')}_{safe_base_name}")
#             os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
#             gemini_uploaded_file_ref = None; all_validated_data = []; overall_success = True; gcs_file_path = None # Initialize GCS path

#             try:
#                 status_text.info("💾 Saving file temporarily..."); progress_bar.progress(5, text="Saving temp file...");
#                 with open(temp_file_path, "wb") as f: f.write(st.session_state.pdf_bytes)
#                 status_text.info("☁️ Uploading file to Google Cloud AI..."); progress_bar.progress(10, text="Uploading to cloud...")
#                 for upload_attempt in range(3):
#                     try: gemini_uploaded_file_ref = genai.upload_file(path=temp_file_path); st.toast(f"File '{gemini_uploaded_file_ref.display_name}' uploaded to cloud.", icon="☁️"); break
#                     except Exception as upload_err:
#                         err_str = str(upload_err).lower()
#                         if "api key" in err_str or "authenticat" in err_str or "permission" in err_str: st.error(f"❌ File upload failed due to API key/permission issue: {upload_err}"); st.error("Please verify the API key has File API permissions enabled."); raise
#                         elif upload_attempt < 2: st.warning(f"Upload attempt {upload_attempt+1} failed: {upload_err}. Retrying..."); time.sleep(2 + upload_attempt)
#                         else: st.error(f"Upload failed after multiple attempts: {upload_err}"); raise
#                 if not gemini_uploaded_file_ref: raise Exception("Failed to upload file to Google Cloud AI after retries.")
#                 progress_bar.progress(15, text="File uploaded. Starting section analysis...")
#                 sections_to_process = st.session_state.selected_sections_to_run; num_sections = len(sections_to_process); progress_per_section = (95 - 15) / num_sections if num_sections > 0 else 0
#                 for i, section_name in enumerate(sections_to_process):
#                     if section_name not in ALL_SECTIONS: st.warning(f"Skipping invalid section '{section_name}' found in selection."); continue
#                     current_progress = int(15 + (i * progress_per_section)); progress_bar.progress(current_progress, text=f"Analysing Section: {section_name}...")
#                     section_data, section_status, section_warnings = generate_section_analysis(section_name, gemini_uploaded_file_ref, status_text, current_api_key)
#                     st.session_state.run_status_summary.append({"section": section_name, "status": section_status, "warnings": section_warnings})
#                     if section_status == "Success" and section_data:
#                         for item in section_data: item["File Name"] = base_file_name; item["Generation Time"] = run_timestamp_str
#                         all_validated_data.extend(section_data)
#                     else: overall_success = False
#                     progress_bar.progress(int(15 + ((i + 1) * progress_per_section)), text=f"Completed: {section_name}")
#                 st.session_state.analysis_results = all_validated_data; progress_bar.progress(100, text="Analysis process finished!")
#                 if overall_success: status_text.success("🏁 Analysis finished successfully!")
#                 else: status_text.warning("🏁 Analysis finished, but some sections encountered issues (see summary below).")
#                 st.session_state.analysis_complete = True

#                 # --- Save to GCS and Firestore (after successful analysis, even if partial) ---
#                 if overall_success and all_validated_data and st.session_state.pdf_bytes:
#                     try:
#                         timestamp = datetime.now()
#                         # Use timestamp in blob name for uniqueness if needed, but Firestore ID is better
#                         firestore_doc_id = f"{safe_base_name}_{timestamp.isoformat()}"
#                         gcs_blob_name = f"{GCS_PDF_FOLDER}/{firestore_doc_id}.pdf"

#                         # Upload PDF to GCS
#                         gcs_file_path = upload_to_gcs(GCS_BUCKET_NAME, st.session_state.pdf_bytes, gcs_blob_name, status_text)

#                         # Save results and GCS path to Firestore
#                         status_text.info("💾 Saving results and PDF reference to database...")
#                         doc_ref = db.collection("analysis_runs").document(firestore_doc_id)
#                         doc_ref.set({
#                             "filename": base_file_name,
#                             "analysis_timestamp": timestamp, # Use Firestore server timestamp if preferred: firestore.SERVER_TIMESTAMP
#                             "results": all_validated_data,
#                             "run_status": st.session_state.run_status_summary,
#                             "gcs_pdf_path": gcs_file_path # Store the GCS path
#                         })
#                         status_text.success("💾 Results and PDF link saved successfully.")
#                         time.sleep(1)
#                     except Exception as db_gcs_err:
#                         st.error(f"❌ Failed to save results/PDF to cloud: {db_gcs_err}")
#                         print(f"DB/GCS Save Error: {db_gcs_err}\n{traceback.format_exc()}")
#                         # Decide if this should mark the overall run as failed? Maybe just warn.
#                         st.session_state.run_status_summary.append({
#                             "section": "Cloud Save",
#                             "status": "Failed",
#                             "warnings": [f"Error saving to GCS/Firestore: {db_gcs_err}"]
#                         })

#             except Exception as main_err:
#                 st.error(f"❌ CRITICAL ERROR during analysis workflow: {main_err}"); print(traceback.format_exc())
#                 overall_success = False; st.session_state.analysis_complete = False
#                 st.session_state.run_status_summary.append({"section": "Overall Process Control", "status": "Critical Error", "warnings": [str(main_err), "Analysis halted. See logs."]})
#                 status_text.error(f"Analysis stopped due to critical error: {main_err}")
#             finally:
#                 st.session_state.processing_in_progress = False; time.sleep(4); status_text.empty(); progress_bar.empty()
#                 if gemini_uploaded_file_ref and hasattr(gemini_uploaded_file_ref, 'name'):
#                     try: status_text.info(f"☁️ Deleting temporary Gemini cloud file: {gemini_uploaded_file_ref.name}..."); genai.delete_file(name=gemini_uploaded_file_ref.name); st.toast("Gemini cloud file deleted.", icon="🗑️"); time.sleep(1); status_text.empty()
#                     except Exception as del_err: st.sidebar.warning(f"Gemini cloud cleanup issue: {del_err}", icon="⚠️"); status_text.warning(f"Could not delete Gemini cloud file: {del_err}"); print(f"WARN: Failed to delete cloud file {gemini_uploaded_file_ref.name}: {del_err}")
#                 if os.path.exists(temp_file_path):
#                     try: os.remove(temp_file_path)
#                     except Exception as local_del_err: st.sidebar.warning(f"Local temp file cleanup issue: {local_del_err}", icon="⚠️"); print(f"WARN: Failed to delete local temp file {temp_file_path}: {local_del_err}")
#             st.rerun()
# else:
#     # If viewing history, show minimal sidebar info or placeholders
#     st.sidebar.info("📜 Viewing historical data.")
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("Click 'Exit History View' above to start a new analysis.")


# # --- 7. Display Area (Results and PDF Viewer) ---
# if st.session_state.pdf_bytes is not None:
#     col1, col2 = st.columns([6, 4]) # Results | PDF Viewer

#     # --- Column 1: Analysis Results ---
#     with col1:
#         # --- Run Status Summary ---
#         if st.session_state.run_status_summary:
#             final_status = "✅ Success"; has_failures = any(s['status'] == "Failed" or "Error" in s['status'] for s in st.session_state.run_status_summary); has_warnings = any(s['status'] != "Success" and not has_failures for s in st.session_state.run_status_summary)
#             # Corrected if/elif logic
#             if has_failures:
#                 final_status = "❌ Failed"
#             elif has_warnings:
#                 final_status = "⚠️ Issues"
#             # Expand summary if viewing history OR if there were issues
#             expand_summary = st.session_state.viewing_history or (final_status != "✅ Success")
#             with st.expander(f"📊 Analysis Run Summary ({final_status})", expanded=expand_summary):
#                 for item in st.session_state.run_status_summary:
#                     icon = "✅" if item['status'] == "Success" else "❌" if item['status'] == "Failed" or "Error" in item['status'] else "⚠️"
#                     st.markdown(f"**{item['section']}**: {icon} {item['status']}")
#                     if item['warnings']:
#                         filtered_warnings = [msg for msg in item['warnings'] if not (isinstance(msg, str) and msg.startswith("Validation Issues Found"))]
#                         if filtered_warnings:
#                             with st.container(): st.caption("Details:")
#                             for msg in filtered_warnings:
#                                 msg_str = str(msg);
#                                 if "CRITICAL" in msg_str or "Error" in msg_str or "Block" in msg_str or "Fail" in msg_str: st.error(f" L> {msg_str}")
#                                 elif "Warn" in msg_str or "Missing" in msg_str or "Unexpected" in msg_str or "Empty list" in msg_str or "validation issues" in msg_str.lower(): st.warning(f" L> {msg_str}")
#                                 else: st.caption(f" L> {msg_str}")

#         st.subheader("Analysis Results")

#         # --- Display Results (if analysis complete/history loaded and results exist) ---
#         if (st.session_state.analysis_complete or st.session_state.viewing_history) and st.session_state.analysis_results:
#             results_list = st.session_state.analysis_results

#             # --- NEW: Scatter Plot Expander ---
#             try:
#                 plot_data = []
#                 for item in results_list:
#                     plot_data.append({
#                         'Question Number': item.get('Question Number', 0), # Default to 0 if missing
#                         'Number of Evidence Items': len(item.get('Evidence', [])),
#                         'Question Category': item.get('Question Category', 'Uncategorized'),
#                         'Question': item.get('Question', 'N/A') # For hover
#                     })

#                 if plot_data:
#                     df_plot = pd.DataFrame(plot_data)
#                     with st.expander("📊 Evidence Count Analysis (Scatter Plot)", expanded=False):
#                         fig = px.scatter(
#                             df_plot,
#                             x='Question Number',
#                             y='Number of Evidence Items',
#                             color='Question Category',
#                             title="Number of Evidence Clauses Found per Question",
#                             labels={'Number of Evidence Items': 'Evidence Count'},
#                             hover_data=['Question'] # Show question text on hover
#                         )
#                         fig.update_traces(marker=dict(size=10, opacity=0.7), selector=dict(mode='markers'))
#                         fig.update_layout(xaxis_title="Question Number", yaxis_title="Number of Evidence Clauses")
#                         st.plotly_chart(fig, use_container_width=True)
#                         st.caption("This plot shows how many separate evidence clauses the AI referenced for each question. Hover over points for question details.")
#                 # else: # Optional: message if no data for plot
#                     # st.info("No data available to generate the evidence count plot.")

#             except Exception as plot_err:
#                  st.warning(f"Could not generate scatter plot: {plot_err}")
#                  print(f"Plotting Error: {plot_err}\n{traceback.format_exc()}")
#             # --- END: Scatter Plot Expander ---


#             # --- Tabbed Results Display ---
#             grouped_results = defaultdict(list); categories_ordered = []
#             for item in results_list:
#                 category = item.get("Question Category", "Uncategorized")
#                 if category not in grouped_results: categories_ordered.append(category)
#                 grouped_results[category].append(item)

#             if categories_ordered:
#                 # Try to maintain a consistent tab order if possible (e.g., based on ALL_SECTIONS)
#                 # This is complex if results don't cover all sections. Simple sort for now.
#                 categories_ordered.sort()
#                 category_tabs = st.tabs(categories_ordered)

#                 for i, category in enumerate(categories_ordered):
#                     with category_tabs[i]:
#                         category_items = sorted(grouped_results[category], key=lambda x: x.get('Question Number', float('inf')))
#                         for index, result_item in enumerate(category_items):
#                             q_num = result_item.get('Question Number', 'N/A'); question_text = result_item.get('Question', 'N/A')
#                             # Use a more unique key including category to avoid clashes if same Q# appears unexpectedly elsewhere
#                             expander_key = f"exp_{category}_{q_num}_{index}"
#                             with st.expander(f"**Q{q_num}:** {question_text}"):
#                                 st.markdown(f"**Answer:**"); st.markdown(f"> {result_item.get('Answer', 'N/A')}"); st.markdown("---")
#                                 evidence_list = result_item.get('Evidence', [])
#                                 if evidence_list:
#                                     st.markdown("**Evidence:**")
#                                     for ev_index, evidence_item in enumerate(evidence_list):
#                                         clause_ref = evidence_item.get('Clause Reference', 'N/A'); search_text = evidence_item.get('Searchable Clause Text', None)
#                                         clause_wording = evidence_item.get('Clause Wording', 'N/A'); base_key = f"ev_{category}_{q_num}_{index}_{ev_index}"
#                                         ev_cols = st.columns([3, 1])
#                                         with ev_cols[0]:
#                                             if search_text:
#                                                 button_key = f"search_btn_{base_key}"; button_label = f"Clause: **{clause_ref or 'Link'}** (Find & View)"
#                                                 if st.button(button_label, key=button_key, help=f"Search for text related to '{clause_ref or 'this evidence'}' and view the page."):
#                                                     st.session_state.search_trigger = {'text': search_text, 'ref': clause_ref}; st.session_state.last_search_result = None; st.rerun()
#                                             elif clause_ref != 'N/A': st.markdown(f"- Clause: **{clause_ref}** (No searchable text provided by AI)")
#                                             else: st.caption("No clause reference provided.")
#                                         with ev_cols[1]:
#                                             if clause_wording != 'N/A':
#                                                 toggle_key = f"toggle_wording_{base_key}"; show_wording = st.toggle("Show Wording", key=toggle_key, value=st.session_state.show_wording_states.get(toggle_key, False), help="Show/hide the exact clause wording extracted by the AI.")
#                                                 # Check if toggle state changed
#                                                 if show_wording != st.session_state.show_wording_states.get(toggle_key, False):
#                                                     st.session_state.show_wording_states[toggle_key] = show_wording
#                                                     st.rerun() # Rerun needed to show/hide text area
#                                         if st.session_state.show_wording_states.get(f"toggle_wording_{base_key}", False): st.text_area(f"AI Extracted Wording for '{clause_ref}':", value=clause_wording, height=150, disabled=True, key=f"wording_area_{base_key}")
#                                         st.markdown("---") # Separator after each evidence item
#                                 else: st.markdown("**Evidence:** None provided.")
#                                 st.markdown("---"); st.markdown("**Answer Justification:**")
#                                 justification_text = result_item.get('Answer Justification', ''); just_key = f"justification_{category}_{q_num}_{index}"
#                                 st.text_area(label="Justification Text Area", value=justification_text, height=100, disabled=True, label_visibility="collapsed", key=just_key)
#             else: st.warning("Analysis generated results, but they could not be grouped by category. Displaying raw list."); st.json(results_list)

#             # --- Excel Download (Now always in sidebar, but button logic here) ---
#             st.sidebar.markdown("---"); st.sidebar.markdown("## Export Results")
#             # Button to trigger preparation
#             if st.sidebar.button("Prepare Data for Excel Download", key="prep_excel", use_container_width=True):
#                 st.session_state.excel_data = None; excel_prep_status = st.sidebar.empty(); excel_prep_status.info("Preparing Excel data...")
#                 try:
#                     excel_rows = [];
#                     for item in results_list:
#                         references = []; first_search_text = "N/A"; evidence = item.get("Evidence")
#                         # Determine File Name and Generation Time for this row
#                         # If viewing history, use the stored values, otherwise use the values added during analysis
#                         file_name_for_excel = st.session_state.history_filename if st.session_state.viewing_history else item.get("File Name", "")
#                         gen_time_for_excel = st.session_state.history_timestamp if st.session_state.viewing_history else item.get("Generation Time", "")

#                         if evidence:
#                             for i, ev in enumerate(evidence):
#                                 if isinstance(ev, dict): references.append(str(ev.get("Clause Reference", "N/A")));
#                                 if i == 0 and isinstance(ev, dict): first_search_text = ev.get("Searchable Clause Text", "N/A")
#                                 # Removed else clause which added "[Invalid Evidence Item]" - validation should prevent this

#                         excel_row = {
#                             "File Name": file_name_for_excel,
#                             "Generation Time": gen_time_for_excel,
#                             "Question Number": item.get("Question Number"),
#                             "Question Category": item.get("Question Category", "Uncategorized"),
#                             "Question": item.get("Question", "N/A"),
#                             "Answer": item.get("Answer", "N/A"),
#                             "Answer Justification": item.get("Answer Justification", "N/A"),
#                             "Clause References (Concatenated)": "; ".join(references) if references else "N/A",
#                             "First Searchable Clause Text": first_search_text
#                         }
#                         excel_rows.append(excel_row)

#                     if not excel_rows: excel_prep_status.warning("No data available to export."); st.session_state.excel_data = None
#                     else:
#                         df_excel = pd.DataFrame(excel_rows); final_columns = [col for col in EXCEL_COLUMN_ORDER if col in df_excel.columns]; extra_cols = [col for col in df_excel.columns if col not in final_columns]; df_excel = df_excel[final_columns + extra_cols]
#                         output = io.BytesIO();
#                         with pd.ExcelWriter(output, engine='openpyxl') as writer: df_excel.to_excel(writer, index=False, sheet_name='Analysis Results')
#                         st.session_state.excel_data = output.getvalue(); excel_prep_status.success("✅ Excel file ready for download!"); time.sleep(2); excel_prep_status.empty()
#                 except Exception as excel_err: excel_prep_status.error(f"Excel Prep Error: {excel_err}"); print(traceback.format_exc())

#             # Download button (only appears if data is ready)
#             if st.session_state.excel_data:
#                  # Use history filename if available, otherwise fallback to current run's name
#                 current_filename = st.session_state.history_filename if st.session_state.viewing_history else st.session_state.get("current_filename", "analysis_results")
#                 safe_base_name = re.sub(r'[^\w\s-]', '', os.path.splitext(current_filename)[0]).strip().replace(' ', '_'); download_filename = f"Analysis_{safe_base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
#                 st.sidebar.download_button(label="📥 Download Results as Excel", data=st.session_state.excel_data, file_name=download_filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_excel_final", use_container_width=True)

#         # --- Fallback messages within Col 1 ---
#         elif st.session_state.analysis_complete and not st.session_state.analysis_results: st.info("Analysis process completed, but no valid results were generated. Check the run summary above for potential issues.")
#         elif st.session_state.processing_in_progress: st.info("Analysis is currently in progress...")
#         elif not st.session_state.analysis_complete and st.session_state.pdf_bytes is not None and not st.session_state.viewing_history: st.info("PDF loaded. Select sections and click 'Analyse Document' in the sidebar to start.")
#         elif not st.session_state.pdf_bytes and not st.session_state.viewing_history: st.info("⬆️ Upload a PDF file or load from history.") # Initial state message


#     # --- Column 2: PDF Viewer (Sticky) ---
#     with col2:
#         st.markdown('<div class="sticky-viewer-content">', unsafe_allow_html=True) # Sticky wrapper start
#         st.subheader("📄 PDF Viewer"); viewer_status_placeholder = st.empty()

#         if st.session_state.search_trigger:
#             search_info = st.session_state.search_trigger; st.session_state.search_trigger = None
#             with st.spinner(f"🔎 Searching for text related to: '{search_info['ref']}'..."):
#                 found_page, instances, term_used, search_status, all_findings = find_text_in_pdf(st.session_state.pdf_bytes, search_info['text'])
#             if found_page:
#                 st.session_state.last_search_result = {'page': found_page, 'instances': instances, 'term': term_used, 'status': search_status, 'ref': search_info['ref'], 'all_findings': all_findings}
#                 st.session_state.current_page = found_page; viewer_status_placeholder.empty(); st.rerun()
#             else: st.session_state.last_search_result = None; viewer_status_placeholder.error(search_status)

#         if st.session_state.pdf_display_ready:
#             try:
#                 # Use tempfile for fitz if bytes cause issues, but bytes usually work
#                 with fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf") as doc: total_pages = doc.page_count
#             except Exception as pdf_load_err: st.error(f"Error loading PDF for page count: {pdf_load_err}"); total_pages = 1; st.session_state.current_page = 1

#             # Ensure current page is valid after potential changes (like loading history)
#             current_display_page = max(1, min(int(st.session_state.get('current_page', 1)), total_pages))
#             if current_display_page != st.session_state.get('current_page'):
#                  st.session_state.current_page = current_display_page
#                  # No rerun here, just update the state for rendering

#             nav_cols = st.columns([1, 3, 1])
#             with nav_cols[0]:
#                 if st.button("⬅️ Prev", key="prev_page", disabled=(current_display_page <= 1), use_container_width=True): st.session_state.current_page -= 1; st.session_state.last_search_result = None; st.rerun()
#             with nav_cols[2]:
#                 if st.button("Next ➡️", key="next_page", disabled=(current_display_page >= total_pages), use_container_width=True): st.session_state.current_page += 1; st.session_state.last_search_result = None; st.rerun()

#             page_info_text = f"Page {current_display_page} of {total_pages}"; search_context_ref = None
#             if st.session_state.last_search_result and st.session_state.last_search_result.get('all_findings'): search_context_ref = st.session_state.last_search_result['ref']; page_info_text += f" (🎯 Multi-match: '{search_context_ref}')"
#             elif st.session_state.last_search_result and st.session_state.last_search_result['page'] == current_display_page: search_context_ref = st.session_state.last_search_result['ref']; page_info_text += f" (🎯 Ref: '{search_context_ref}')"
#             nav_cols[1].markdown(f"<div style='text-align: center; padding-top: 0.5rem;'>{page_info_text}</div>", unsafe_allow_html=True)

#             if st.session_state.last_search_result and st.session_state.last_search_result.get('all_findings'):
#                 multi_findings = st.session_state.last_search_result['all_findings']; found_pages = sorted([f[0] for f in multi_findings])
#                 status_msg = st.session_state.last_search_result['status']; viewer_status_placeholder.info(status_msg); st.write("Jump to other matches for this reference:")
#                 num_buttons = len(found_pages); btn_cols = st.columns(min(num_buttons, 5)) # Limit columns to avoid crowding
#                 for idx, p_num in enumerate(found_pages):
#                     col_idx = idx % len(btn_cols) # Distribute buttons across columns
#                     is_current = (p_num == current_display_page)
#                     if btn_cols[col_idx].button(f"Page {p_num}", key=f"jump_{p_num}_{search_context_ref}", disabled=is_current, use_container_width=True):
#                         st.session_state.current_page = p_num; new_instances = next((inst for pg, inst in multi_findings if pg == p_num), None)
#                         # Update last_search_result correctly for the new page
#                         st.session_state.last_search_result['instances'] = new_instances
#                         st.session_state.last_search_result['page'] = p_num
#                         # Ensure status message reflects the jump, not the multi-match warning anymore
#                         term_desc = st.session_state.last_search_result.get('term', 'text')
#                         st.session_state.last_search_result['status'] = f"✅ Viewing match for '{term_desc}' on page {p_num}."
#                         st.session_state.last_search_result['all_findings'] = None # Clear multi-findings after jump
#                         st.rerun()

#             st.markdown("---")
#             highlights_to_apply = None; render_status_override = None
#             if st.session_state.last_search_result and st.session_state.last_search_result.get('page') == current_display_page:
#                 highlights_to_apply = st.session_state.last_search_result.get('instances')
#                 # Don't override status if we just jumped from multi-match
#                 if not st.session_state.last_search_result.get('all_findings'):
#                     render_status_override = st.session_state.last_search_result.get('status')

#             image_bytes, render_status = render_pdf_page_to_image(st.session_state.pdf_bytes, current_display_page, highlight_instances=highlights_to_apply, dpi=150)

#             if image_bytes:
#                 st.image(image_bytes, caption=f"Page {current_display_page} - View", use_container_width=True) # Use container width
#                 final_status = render_status_override if render_status_override else render_status
#                 # Only show status if not in multi-match jump list mode OR if override exists
#                 if not (st.session_state.last_search_result and st.session_state.last_search_result.get('all_findings') and not render_status_override):
#                     if final_status:
#                         if "✅" in final_status or "✨" in final_status or "Found" in final_status : viewer_status_placeholder.success(final_status)
#                         elif "⚠️" in final_status or "warning" in final_status.lower() or "multiple pages" in final_status.lower(): viewer_status_placeholder.warning(final_status)
#                         elif "❌" in final_status or "error" in final_status.lower(): viewer_status_placeholder.error(final_status)
#                         else: viewer_status_placeholder.caption(final_status)
#                     else: viewer_status_placeholder.empty() # Clear status if none applies
#             else: viewer_status_placeholder.error(f"Failed to render page {current_display_page}. {render_status or ''}")
#         else: st.info("PDF loaded, preparing viewer..."); viewer_status_placeholder.empty()
#         st.markdown('</div>', unsafe_allow_html=True) # Sticky wrapper end

# # --- Fallback message if no PDF loaded (and not viewing history) ---
# elif not st.session_state.pdf_bytes and not st.session_state.viewing_history:
#      st.info("⬆️ Upload a PDF file using the sidebar to begin, or load an analysis from the History page.")