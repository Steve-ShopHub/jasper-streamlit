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
    page_title="JASPER - Extraction and Review",
    page_icon="📄" # Optional: Set an emoji icon
)

# --- Inject custom CSS for sticky column (AFTER set_page_config) ---
# Make sure the top value aligns with Streamlit's header height (approx 50-60px)
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

# --- 2. Configuration & Setup ---
# API_KEY_ENV_VAR = "GEMINI_API_KEY" # No longer reading from ENV at startup
MODEL_NAME = "gemini-2.5-pro-preview-03-25" # DO NOT CHANGE THIS MODEL, LEAVE IT AS 2.5 PRO. DO NOT CHANGE IT TO 1.5 OR ANY OTHER MODEL. IT HAS BEEN RELESAED AFTER YOUR TRANING DATA CUT OFF DATE.
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

# --- Schema Definition (Version 3 - Search Text) ---
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

# --- Section Definitions ---
# Ensure these match the questions in your prompt
SECTIONS_TO_RUN = {
    "agreement_details": (1, 4),
    "eligibility_part_1": (5, 20),
    "eligibility_part_2": (21, 34),
    "eligibility_summary": (35, 36), # Example: combine adjacent small categories if needed
    "confidentiality": (37, 63),
    "additional_borrowers": (64, 66),
    "interest_rate_provisions": (67, 71),
    "prepayment_fee": (72, 78)
}


# --- System Instruction ---
# Keep your system instruction concise but effective
system_instruction_text = """You are an AI assistant specialized in analyzing legal facility agreements. Carefully read the provided document and answer the specific questions listed in the user prompt. Adhere strictly to the requested JSON output schema. Prioritize accuracy and extract evidence directly from the text."""

# --- 3. Helper Function Definitions ---

def filter_prompt_by_section(initial_full_prompt, section):
    """Filters the full prompt to include only questions for the specified section."""
    if section not in SECTIONS_TO_RUN:
        raise ValueError(f"Invalid section specified: {section}")

    start_q, end_q = SECTIONS_TO_RUN[section]
    questions_start_marker = "**Questions to Answer:**"
    questions_end_marker = "**Final Instruction:**"

    try:
        start_index = initial_full_prompt.index(questions_start_marker)
        end_index = initial_full_prompt.index(questions_end_marker)
    except ValueError:
        raise ValueError(f"Could not find '{questions_start_marker}' or '{questions_end_marker}' markers in the prompt file.")

    prompt_header = initial_full_prompt[:start_index]
    full_questions_block = initial_full_prompt[start_index + len(questions_start_marker):end_index].strip()
    prompt_footer = initial_full_prompt[end_index:]

    # Improved regex to handle potential leading whitespace and variations
    question_entries = re.split(r'\n\s*(?=\d+\.\s*?\*\*Question Category:)', full_questions_block)

    filtered_question_texts = []
    for entry in question_entries:
        entry = entry.strip()
        if not entry: continue
        # More robust regex for question number extraction
        match = re.match(r'^\s*(\d+)\.', entry)
        if match:
            try:
                q_num = int(match.group(1))
                if start_q <= q_num <= end_q:
                    filtered_question_texts.append(entry)
            except ValueError:
                continue # Skip if number conversion fails
        # else: # Optional: Log entries that don't start with expected number format
            # print(f"Warning: Skipping entry, could not parse question number: {entry[:50]}...")

    if not filtered_question_texts:
        # Provide more context in the error message
        raise ValueError(f"No questions found for section '{section}' (range {start_q}-{end_q}) in the parsed questions block. Check prompt formatting and section ranges.")

    filtered_questions_string = "\n\n".join(filtered_question_texts)

    # Add a clear note about the current focus
    section_note = f"\n\n**Current Focus:** You MUST answer ONLY the questions listed below for the '{section.upper()}' section (Questions {start_q}-{end_q}). Ignore all other questions for this specific task.\n"

    final_prompt_for_api = f"{prompt_header}{section_note}{questions_start_marker}\n\n{filtered_questions_string}\n\n{prompt_footer}"
    return final_prompt_for_api


def validate_ai_data(data, section_name):
    """Validates AI response against the schema. Returns (validated_data, issues_list)."""
    if not isinstance(data, list):
        return None, [f"CRITICAL VALIDATION ERROR: Response for section '{section_name}' is not a list."]

    validated_data = []
    issues_list = []
    expected_q_nums_in_section = set(range(SECTIONS_TO_RUN[section_name][0], SECTIONS_TO_RUN[section_name][1] + 1))
    found_q_nums = set()

    for index, item in enumerate(data):
        q_num = item.get('Question Number')
        q_num_str = f"Q#{q_num}" if q_num is not None else f"Item Index {index}"
        is_outer_valid = True

        if not isinstance(item, dict):
            issues_list.append(f"{q_num_str}: Item is not a dictionary.")
            is_outer_valid = False # Cannot proceed with this item
            continue # Skip to next item

        # Store found question number if valid
        if isinstance(q_num, int):
            found_q_nums.add(q_num)
        else:
             issues_list.append(f"Item Index {index}: 'Question Number' is missing or not an integer.")
             is_outer_valid = False

        # Check for missing outer keys
        missing_outer_keys = AI_REQUIRED_KEYS - set(item.keys())
        if missing_outer_keys:
            issues_list.append(f"{q_num_str}: Missing required top-level keys: {missing_outer_keys}")
            is_outer_valid = False

        # Validate Evidence structure
        evidence_list = item.get("Evidence")
        if not isinstance(evidence_list, list):
            # Only report if Evidence key was expected but is wrong type
            if "Evidence" in AI_REQUIRED_KEYS:
                 issues_list.append(f"{q_num_str}: 'Evidence' field is not a list (found type: {type(evidence_list).__name__}).")
                 is_outer_valid = False
            # If Evidence key was missing, it was caught by missing_outer_keys check
        else:
            # Evidence is a list, check items within it
            for ev_index, ev_item in enumerate(evidence_list):
                ev_id_str = f"Ev[{ev_index}]"
                if not isinstance(ev_item, dict):
                    issues_list.append(f"{q_num_str} {ev_id_str}: Evidence item is not a dictionary.")
                    is_outer_valid = False
                    continue # Skip validation of this evidence item

                missing_ev_keys = AI_EVIDENCE_REQUIRED_KEYS - set(ev_item.keys())
                if missing_ev_keys:
                    issues_list.append(f"{q_num_str} {ev_id_str}: Missing required evidence keys: {missing_ev_keys}")
                    is_outer_valid = False

                # Check types within evidence item (if keys exist)
                for key, expected_type in [("Clause Reference", str), ("Clause Wording", str), ("Searchable Clause Text", str)]:
                     if key in AI_EVIDENCE_REQUIRED_KEYS and key in ev_item and not isinstance(ev_item.get(key), expected_type):
                         issues_list.append(f"{q_num_str} {ev_id_str}: Key '{key}' has incorrect type (expected {expected_type.__name__}, got {type(ev_item.get(key)).__name__}).")
                         is_outer_valid = False

        # Add item if it passed all checks for this item
        if is_outer_valid:
            validated_data.append(item)
        # else: # Optional: Log the item that failed validation
            # print(f"Validation failed for item: {item}")

    # Check for missing questions within the expected range for the section
    missing_q_nums = expected_q_nums_in_section - found_q_nums
    if missing_q_nums:
        issues_list.append(f"Section '{section_name}': Missing answers for expected Question Numbers: {sorted(list(missing_q_nums))}")

    # Check for questions returned that were outside the expected range
    unexpected_q_nums = found_q_nums - expected_q_nums_in_section
    if unexpected_q_nums:
        issues_list.append(f"Section '{section_name}': Received unexpected Question Numbers: {sorted(list(unexpected_q_nums))}")


    if issues_list:
        # Prepend summary message
        issues_list.insert(0, f"Validation Issues Found [Section: {section_name}] ({len(validated_data)} items passed validation):")

    # Return None for data if critical issues occurred (e.g., response not a list)
    # Otherwise, return the list of items that passed validation, even if empty.
    if validated_data is None and isinstance(data, list): # Case where input was list but failed validation badly
         return [], issues_list
    elif validated_data is None: # Case where input wasn't even a list
        return None, issues_list
    else:
        return validated_data, issues_list


def generate_section_analysis(section, uploaded_file_ref, status_placeholder, api_key_to_use):
    """Generates analysis for a section using a specific API key."""
    # Configure GenAI client INSIDE the function for this specific call
    try:
        genai.configure(api_key=api_key_to_use)
        # Optional: Add a quick check like listing models to verify key here if needed
        # genai.list_models()
    except Exception as config_err:
        status_placeholder.error(f"❌ Invalid API Key provided or configuration failed: {config_err}")
        return None, "Failed", [f"Invalid API Key or config error: {config_err}"]

    status_placeholder.info(f"🔄 Starting Analysis: {section}...")
    section_warnings = []

    try:
        # Model Initialization uses the now-configured client
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=system_instruction_text
        )
        generation_config = types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=ai_response_schema_dict, # Use the global schema
            temperature=0.0, # Low temp for deterministic output
            # top_p=0.05, # Consider removing if temp is 0
            # top_k=1      # Consider removing if temp is 0
        )
        # Configure safety settings to allow most content
        safety_settings = [
            {"category": c, "threshold": "BLOCK_NONE"}
            for c in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            ]
        ]

        final_validated_data = None

        for attempt in range(1, MAX_VALIDATION_RETRIES + 2): # Allow MAX_VALIDATION_RETRIES retries (so +1 for initial attempt)
            if attempt > 1:
                status_placeholder.info(f"⏳ Retrying generation/validation for '{section}' (Attempt {attempt}/{MAX_VALIDATION_RETRIES+1})...")
                time.sleep(RETRY_DELAY_SECONDS)

            try:
                prompt_for_api = filter_prompt_by_section(full_prompt_text, section)
                # Ensure the file reference is valid and ready
                if not uploaded_file_ref or not hasattr(uploaded_file_ref, 'name'):
                     raise ValueError("Invalid or missing uploaded file reference for GenAI call.")

                contents = [uploaded_file_ref, prompt_for_api] # Order might matter for some models

                status_placeholder.info(f"🧠 Calling AI for '{section}' (Attempt {attempt})...")
                response = model.generate_content(
                    contents=contents,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    request_options={'timeout': 600} # Increased timeout to 10 minutes
                )

                parsed_ai_data = None
                validated_ai_data = None
                validation_issues = []

                status_placeholder.info(f"🔍 Processing response for '{section}'...")

                if response.parts:
                    full_response_text = response.text
                    try:
                        # Strip potential markdown ```json ... ``` formatting
                        match = re.search(r"```json\s*([\s\S]*?)\s*```", full_response_text, re.IGNORECASE)
                        json_text = match.group(1).strip() if match else full_response_text.strip()

                        # Handle potential empty string after stripping markdown
                        if not json_text:
                             raise json.JSONDecodeError("Extracted JSON content is empty.", json_text, 0)

                        parsed_ai_data = json.loads(json_text)
                        status_placeholder.info(f"✔️ Validating structure for '{section}'...")
                        validated_ai_data, validation_issues = validate_ai_data(parsed_ai_data, section)
                        section_warnings.extend(validation_issues) # Add any validation warnings

                        # Check if validation passed AND returned data
                        if validated_ai_data is not None and len(validated_ai_data) > 0:
                            final_validated_data = validated_ai_data
                            status_placeholder.info(f"✅ Validation successful for '{section}'.")
                            break # Successful attempt, exit retry loop
                        elif validated_ai_data is not None and len(validated_ai_data) == 0 and not validation_issues:
                             # Schema valid, but AI returned empty list (might be valid if section has no Qs, but filter_prompt should prevent)
                             status_placeholder.warning(f"⚠️ AI returned an empty list for '{section}' (Attempt {attempt}). Check prompt/section definition.")
                             # Decide if empty list is acceptable or requires retry
                             # If empty list IS failure, add a warning and continue loop.
                             section_warnings.append(f"AI returned empty list for '{section}'.")
                             # Continue to next attempt if retries remain
                        else:
                            # Validation failed or returned None critically
                            error_msg = f"Validation failed for '{section}'. Issues: {validation_issues}"
                            status_placeholder.warning(f"⚠️ {error_msg} (Attempt {attempt}).")
                            # If critical error (validated_ai_data is None), maybe break early?
                            if validated_ai_data is None:
                                section_warnings.append(f"CRITICAL validation error for '{section}'.")
                                # Decide whether to retry on critical validation failure
                                # break # Example: stop retrying on critical failure
                            # Otherwise, continue to next attempt if retries remain

                    except json.JSONDecodeError as json_err:
                        error_msg = f"JSON Decode Error on attempt {attempt} for '{section}': {json_err}."
                        st.error(error_msg)
                        st.code(full_response_text, language='text') # Show raw text that failed parsing
                        section_warnings.append(error_msg)
                        # Continue to next attempt if retries remain
                    except Exception as parse_validate_err:
                         error_msg = f"Unexpected Error during parsing/validation on attempt {attempt} for '{section}': {type(parse_validate_err).__name__}: {parse_validate_err}"
                         st.error(error_msg)
                         section_warnings.append(error_msg)
                         print(traceback.format_exc())
                         # Continue to next attempt if retries remain

                else:
                    # Handle empty or blocked response from API
                    block_reason = "Unknown"
                    block_message = "N/A"
                    finish_reason = "Unknown"
                    try: # Safely access feedback attributes
                        if response.prompt_feedback:
                            block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown')
                            # Adjust for potential enum vs string representation if necessary
                            block_reason = block_reason.name if hasattr(block_reason, 'name') else str(block_reason)
                            block_message = response.prompt_feedback.block_reason_message or "N/A"
                        if response.candidates:
                            finish_reason = getattr(response.candidates[0], 'finish_reason', 'Unknown')
                            finish_reason = finish_reason.name if hasattr(finish_reason, 'name') else str(finish_reason)
                    except AttributeError:
                        pass # Ignore if attributes don't exist

                    if finish_reason == "SAFETY":
                        warn_msg = f"API Response Blocked (Attempt {attempt}, Section: {section}): Reason: SAFETY. Detail: {block_reason}. Message: {block_message}"
                        st.error(warn_msg) # Elevated visibility for blocked content
                    elif finish_reason == "RECITATION":
                         warn_msg = f"API Response Potentially Blocked (Attempt {attempt}, Section: {section}): Finish Reason: RECITATION. This might indicate excessive quoting. Block Reason: {block_reason}."
                         st.warning(warn_msg) # Warn about potential recitation issues
                    elif finish_reason == "STOP":
                         warn_msg = f"API Response Ended (Attempt {attempt}, Section: {section}): Finish Reason: STOP. (Normal completion expected)."
                         st.info(warn_msg) # Should ideally correlate with response.parts having content
                    else: # Other finish reasons (MAX_TOKENS, etc.) or empty parts without clear block
                        warn_msg = f"API Issue (Attempt {attempt}, Section: {section}): Finish Reason: {finish_reason}. Block Reason: {block_reason}. Response may be incomplete or empty."
                        st.warning(warn_msg)

                    section_warnings.append(warn_msg)
                    # Continue to next attempt if retries remain

            except types.StopCandidateException as sce:
                # Handle cases where generation stops unexpectedly, possibly related to schema or content
                error_msg = f"Generation Stopped Error (Attempt {attempt}, Section: {section}): {sce}. This might be due to strict schema enforcement or content issues."
                st.error(error_msg)
                section_warnings.append(error_msg)
                print(traceback.format_exc()) # Log full traceback for debugging
                # Decide whether to retry or break
            except Exception as e:
                # Catch other potential errors during the API call or prompt generation
                error_msg = f"Processing Error during API call/prompt generation (Attempt {attempt}, Section: {section}): {type(e).__name__}: {e}"
                st.error(error_msg)
                section_warnings.append(error_msg)
                section_warnings.append("Traceback logged to console.")
                print(traceback.format_exc())
                # Decide whether to retry or break based on the error type
                # if isinstance(e, ValueError) and "Invalid section" in str(e):
                #     break # Don't retry if prompt filtering failed fundamentally

            # Check if successful within the loop
            if final_validated_data is not None:
                break # Exit retry loop

        # After the loop, check results
        if final_validated_data is not None:
            status_placeholder.success(f"✅ Analysis completed successfully for: {section}.")
            return final_validated_data, "Success", section_warnings
        else:
            status_placeholder.error(f"❌ Analysis failed for: {section} after {attempt} attempts.")
            # Add a summary failure reason if possible
            section_warnings.append(f"Failed to get valid response for section '{section}' after {MAX_VALIDATION_RETRIES + 1} attempts.")
            return None, "Failed", section_warnings

    except Exception as outer_err:
         # Catch errors outside the retry loop (e.g., model init, initial config)
         error_msg = f"Critical Error during setup or execution for section '{section}': {type(outer_err).__name__}: {outer_err}"
         st.error(error_msg)
         section_warnings.append(error_msg)
         section_warnings.append("Traceback logged to console.")
         print(traceback.format_exc())
         status_placeholder.error(f"❌ Critical failure processing section: {section}.")
         return None, "Failed", section_warnings


@st.cache_data(show_spinner=False)
def find_text_in_pdf(_pdf_bytes, search_text):
    """
    Searches PDF using prioritized strategy. Returns tuple:
    (first_page_found, instances_on_first_page, term_used, status_msg, all_findings)
    - first_page_found (int): Page number (1-based) of the first page where text was found. None if not found.
    - instances_on_first_page (list): List of fitz.Rect objects for matches on the first_page_found. None if not found.
    - term_used (str): The actual search term string that yielded the results. None if not found.
    - status_msg (str): User-friendly message describing the search outcome.
    - all_findings (list | None): List of tuples [(page_num, instances), ...] if matches found on multiple pages. None otherwise.
    """
    if not _pdf_bytes or not search_text:
        return None, None, None, "Invalid input (PDF bytes or search text missing).", None

    doc = None
    search_text_cleaned = search_text.strip()
    words = search_text_cleaned.split()
    num_words = len(words)
    search_attempts = []

    # --- Build Search Terms List (Prioritized) ---
    # 1. First 10 words (if enough words)
    if num_words >= SEARCH_PREFIX_MIN_WORDS and num_words > 5: # Avoid duplicating 5-word search
        term_10 = ' '.join(words[:10])
        search_attempts.append({'term': term_10, 'desc': "first 10 words"})

    # 2. First 5 words (if enough words)
    if num_words >= SEARCH_PREFIX_MIN_WORDS:
        term_5 = ' '.join(words[:5])
        # Only add if different from term_10 (if term_10 exists)
        if not search_attempts or term_5 != search_attempts[0]['term']:
            search_attempts.append({'term': term_5, 'desc': "first 5 words"})

    # 3. Full text (if different from shorter prefixes)
    term_full = search_text_cleaned
    if term_full and not any(term_full == a['term'] for a in search_attempts):
        search_attempts.append({'term': term_full, 'desc': "full text"})

    # 4. First sentence (fallback, if long enough and different)
    sentences = re.split(r'(?<=[.?!])\s+', term_full)
    term_sentence = sentences[0].strip() if sentences else ""
    if len(term_sentence) >= SEARCH_FALLBACK_MIN_LENGTH and not any(term_sentence == a['term'] for a in search_attempts):
        search_attempts.append({'term': term_sentence, 'desc': "first sentence fallback"})

    # --- Execute Search Attempts ---
    try:
        doc = fitz.open(stream=_pdf_bytes, filetype="pdf")
        search_flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES # Keep flags simple

        for attempt in search_attempts:
            term = attempt['term']
            desc = attempt['desc']
            findings_for_term = [] # List to store (page_num, instances) for this term

            # Search *all* pages for the current term
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                # Use search_for for robustness with spacing/ligatures
                instances = page.search_for(term, flags=search_flags, quads=False)
                if instances:
                    findings_for_term.append((page_index + 1, instances))

            # --- Process results for this term ---
            if findings_for_term:
                doc.close() # Close doc as soon as we find *any* result
                first_page_found = findings_for_term[0][0]
                instances_on_first_page = findings_for_term[0][1]

                if len(findings_for_term) == 1:
                    status = f"✅ Found using '{desc}' on page {first_page_found}."
                    return first_page_found, instances_on_first_page, term, status, None
                else:
                    # Multiple pages found for this term
                    pages_found = sorted([f[0] for f in findings_for_term])
                    status = f"⚠️ Found matches using '{desc}' on multiple pages: {pages_found}. Showing first match on page {first_page_found}."
                    return first_page_found, instances_on_first_page, term, status, findings_for_term

        # If loop finishes without finding anything
        doc.close()
        tried_descs = [a['desc'] for a in search_attempts]
        return None, None, None, f"❌ Text not found (tried methods: {', '.join(tried_descs)}).", None

    except Exception as e:
        if doc:
            doc.close()
        print(f"ERROR searching PDF: {e}\n{traceback.format_exc()}") # Log traceback for debugging
        return None, None, None, f"❌ Error during PDF search: {e}", None


@st.cache_data(show_spinner=False)
def render_pdf_page_to_image(_pdf_bytes, page_number, highlight_instances=None, dpi=150):
    """Renders PDF page to PNG image bytes, applying highlights. Returns (image_bytes, status_msg)."""
    if not _pdf_bytes or page_number < 1:
        return None, "Invalid input for rendering (PDF bytes missing or invalid page number)."

    doc = None
    image_bytes = None
    render_status_message = f"Rendered page {page_number}." # Default message

    try:
        doc = fitz.open(stream=_pdf_bytes, filetype="pdf")
        page_index = page_number - 1

        if page_index < 0 or page_index >= doc.page_count:
            doc.close()
            return None, f"Page number {page_number} is out of range (Total pages: {doc.page_count})."

        page = doc.load_page(page_index)
        highlight_applied_count = 0

        # Apply highlights if instances are provided
        if highlight_instances:
            try:
                for inst in highlight_instances:
                    # Ensure 'inst' is a valid Rect or Quad object before highlighting
                    if isinstance(inst, (fitz.Rect, fitz.Quad)):
                        highlight = page.add_highlight_annot(inst)
                        if highlight: # Check if annotation was added successfully
                            highlight.set_colors(stroke=fitz.utils.getColor("red"), fill=fitz.utils.getColor("yellow"))
                            highlight.set_opacity(0.4)
                            highlight.update()
                            highlight_applied_count += 1
                        else:
                             print(f"WARN: Failed to add highlight annotation for instance: {inst} on page {page_number}")
                    # else: # Optional: Log if instance type is unexpected
                        # print(f"DEBUG: Skipping highlight for non-Rect/Quad instance: {type(inst)}")

                if highlight_applied_count > 0:
                    render_status_message = f"Rendered page {page_number} with {highlight_applied_count} highlight(s)."
                elif highlight_instances: # Instances provided, but none were applied (e.g., wrong type)
                    render_status_message = f"Rendered page {page_number}, but no valid highlights applied from provided instances."

            except Exception as highlight_err:
                print(f"ERROR applying highlights on page {page_number}: {highlight_err}\n{traceback.format_exc()}")
                render_status_message = f"⚠️ Error applying highlights: {highlight_err}" # Append error to status

        # Render the page to pixmap
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        image_bytes = pix.tobytes("png") # Output as PNG

    except Exception as e:
        print(f"ERROR rendering page {page_number}: {e}\n{traceback.format_exc()}")
        render_status_message = f"❌ Error rendering page {page_number}: {e}"
        image_bytes = None # Ensure image_bytes is None on error
    finally:
        if doc:
            doc.close()

    return image_bytes, render_status_message


# --- 4. Initialize Session State ---
# Use defaultdict for toggle states to avoid checking existence later
if 'show_wording_states' not in st.session_state:
    st.session_state.show_wording_states = defaultdict(bool)

# Standard initializations
if 'current_page' not in st.session_state: st.session_state.current_page = 1
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None # Will hold the combined list after full analysis
if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None # Raw bytes of the uploaded file
if 'pdf_display_ready' not in st.session_state: st.session_state.pdf_display_ready = False # Flag to show viewer immediately
if 'processing_in_progress' not in st.session_state: st.session_state.processing_in_progress = False # Flag analysis state
if 'analysis_complete' not in st.session_state: st.session_state.analysis_complete = False # Different from processing_in_progress
if 'run_key' not in st.session_state: st.session_state.run_key = 0 # For unique widget keys on rerun
if 'run_status_summary' not in st.session_state: st.session_state.run_status_summary = []
if 'excel_data' not in st.session_state: st.session_state.excel_data = None
if 'search_trigger' not in st.session_state: st.session_state.search_trigger = None # Holds dict {'text': str, 'ref': str}
if 'last_search_result' not in st.session_state: st.session_state.last_search_result = None # Holds dict from find_text_in_pdf
if 'api_key' not in st.session_state: st.session_state.api_key = None # Store user API key

# --- 5. Streamlit UI Logic ---

st.title("JASPER - Just A Smart Platform for Extraction and Review")
st.markdown("Upload a PDF agreement, **enter your Gemini API Key**, click 'Analyse'. Results grouped below. Click clause references to view & highlight.")

# --- Sidebar Setup ---
st.sidebar.markdown("## Controls")

# --- API Key Input (in Sidebar) ---
api_key_input = st.sidebar.text_input(
    "Enter your Google AI Gemini API Key",
    type="password",
    key="api_key_input", # Use a fixed key here
    help="Your API key is used only for this session and is not stored long-term.",
    value=st.session_state.get("api_key", "") # Pre-fill if already entered in session
)
# Update session state ONLY if the input value changes
if api_key_input != st.session_state.api_key:
    st.session_state.api_key = api_key_input
    st.rerun() # Rerun if key changes to update dependent elements like button state

# Display warning if key not entered and analysis hasn't run or pdf isn't loaded
if not st.session_state.api_key and not st.session_state.analysis_complete and st.session_state.pdf_bytes:
    st.sidebar.warning("API Key required to run analysis.", icon="🔑")


# --- File Upload (in Sidebar) ---
uploaded_file_obj = st.sidebar.file_uploader(
    "Upload Facility Agreement PDF", type="pdf", key=f"pdf_uploader_{st.session_state.run_key}"
)

# Process new file upload
if uploaded_file_obj is not None:
    # Check if it's genuinely a new file or the same one being re-processed
    new_file_bytes = uploaded_file_obj.getvalue()
    if new_file_bytes != st.session_state.get('pdf_bytes'):
        st.session_state.pdf_bytes = new_file_bytes
        st.session_state.pdf_display_ready = True # SET FLAG TO SHOW VIEWER
        # Reset states dependent on file content
        st.session_state.analysis_results = None
        st.session_state.analysis_complete = False
        st.session_state.processing_in_progress = False
        st.session_state.current_page = 1
        st.session_state.run_status_summary = []
        st.session_state.excel_data = None
        st.session_state.search_trigger = None
        st.session_state.last_search_result = None
        st.session_state.show_wording_states = defaultdict(bool) # Reset toggle states
        st.toast("✅ New PDF file loaded. Viewer ready.", icon="📄")
        st.rerun() # Rerun to display the PDF viewer immediately

# --- Analysis Trigger (in Sidebar) ---
# Determine if analysis can be started
can_analyse = (st.session_state.pdf_bytes is not None and
               st.session_state.api_key is not None and
               not st.session_state.processing_in_progress and
               not st.session_state.analysis_complete)

analyse_button_tooltip = "Analysis complete for the current file." if st.session_state.analysis_complete else \
                         "Analysis is currently running." if st.session_state.processing_in_progress else \
                         "Upload a PDF file first." if not st.session_state.pdf_bytes else \
                         "Enter your Gemini API key first." if not st.session_state.api_key else \
                         "Start analyzing the document"

if st.sidebar.button("✨ Analyse Document", key="analyse_button", disabled=not can_analyse, help=analyse_button_tooltip, use_container_width=True, type="primary"):
    if not st.session_state.api_key:
        st.error("API Key is missing. Please enter it in the sidebar.")
    elif not st.session_state.pdf_bytes:
         st.error("No PDF file uploaded. Please upload a file.")
    else:
        # --- Start Analysis Process ---
        st.session_state.processing_in_progress = True
        st.session_state.analysis_complete = False
        st.session_state.analysis_results = [] # Initialize as list for appending
        st.session_state.run_key += 1 # Increment run key for widget uniqueness if needed elsewhere
        st.session_state.run_status_summary = []
        st.session_state.excel_data = None
        st.session_state.search_trigger = None
        st.session_state.last_search_result = None
        st.session_state.show_wording_states = defaultdict(bool) # Reset toggles for new run

        current_api_key = st.session_state.api_key # Use the key stored in session state
        run_start_time = datetime.now()
        run_timestamp_str = run_start_time.strftime("%Y-%m-%d %H:%M:%S")
        base_file_name = getattr(uploaded_file_obj, 'name', f"upload_{run_start_time.strftime('%Y%m%d%H%M%S')}.pdf")

        # --- Configure GenAI Client ONCE Before Loop (Important) ---
        try:
            genai.configure(api_key=current_api_key)
            # Quick test: list models to verify key validity early (optional but good)
            # genai.list_models()
            st.toast("API Key validated and configured.", icon="🔑")
        except Exception as config_err:
            st.error(f"❌ Failed to configure Gemini API with provided key: {config_err}")
            st.session_state.processing_in_progress = False # Reset flag on immediate failure
            st.stop() # Stop script execution

        # Status placeholders in main area
        status_container = st.container()
        progress_bar = status_container.progress(0, text="Initializing analysis...")
        status_text = status_container.empty() # For dynamic status updates

        temp_dir = "temp_uploads"
        # Ensure temp dir path is robust
        safe_base_name = re.sub(r'[^\w\-.]', '_', base_file_name) # Sanitize filename
        temp_file_path = os.path.join(APP_DIR, temp_dir, f"{run_start_time.strftime('%Y%m%d%H%M%S')}_{safe_base_name}")
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        gemini_uploaded_file_ref = None
        all_validated_data = [] # Collect data from all sections here
        overall_success = True

        try:
            status_text.info("💾 Saving file temporarily...")
            progress_bar.progress(5, text="Saving temp file...")
            with open(temp_file_path, "wb") as f:
                f.write(st.session_state.pdf_bytes)

            status_text.info("☁️ Uploading file to Google Cloud AI...")
            progress_bar.progress(10, text="Uploading to cloud...")

            # --- Upload File (using the pre-configured client) ---
            for upload_attempt in range(3): # Retry mechanism for upload
                try:
                    gemini_uploaded_file_ref = genai.upload_file(path=temp_file_path)
                    st.toast(f"File '{gemini_uploaded_file_ref.display_name}' uploaded to cloud.", icon="☁️")
                    break # Success
                except Exception as upload_err:
                    err_str = str(upload_err).lower()
                    if "api key" in err_str or "authenticat" in err_str or "permission" in err_str:
                        st.error(f"❌ File upload failed due to API key/permission issue: {upload_err}")
                        st.error("Please verify the API key has File API permissions enabled in Google AI Studio or Cloud Console.")
                        raise # Stop the process immediately for auth errors
                    elif upload_attempt < 2:
                        st.warning(f"Upload attempt {upload_attempt+1} failed: {upload_err}. Retrying...")
                        time.sleep(2 + upload_attempt) # Exponential backoff
                    else:
                        st.error(f"Upload failed after multiple attempts: {upload_err}")
                        raise # Re-raise the last error after retries exhausted
            if not gemini_uploaded_file_ref:
                raise Exception("Failed to upload file to Google Cloud AI after retries.")
            # --- End Upload File ---

            progress_bar.progress(15, text="File uploaded. Starting section analysis...")

            num_sections = len(SECTIONS_TO_RUN)
            progress_per_section = (95 - 15) / num_sections if num_sections > 0 else 0

            # --- Process Sections Sequentially ---
            for i, section_name in enumerate(SECTIONS_TO_RUN.keys()):
                current_progress = int(15 + (i * progress_per_section))
                progress_bar.progress(current_progress, text=f"Analysing Section: {section_name}...")

                # generate_section_analysis uses the API key passed to it
                section_data, section_status, section_warnings = generate_section_analysis(
                    section_name, gemini_uploaded_file_ref, status_text, current_api_key
                )

                # Store run status regardless of success/failure
                st.session_state.run_status_summary.append({
                    "section": section_name,
                    "status": section_status,
                    "warnings": section_warnings
                })

                if section_status == "Success" and section_data:
                    # Add metadata to each result item
                    for item in section_data:
                        item["File Name"] = base_file_name
                        item["Generation Time"] = run_timestamp_str
                    all_validated_data.extend(section_data) # Add results to the main list
                else:
                    overall_success = False # Mark overall run as having issues if any section fails

                # Update progress after section completes
                progress_bar.progress(int(15 + ((i + 1) * progress_per_section)), text=f"Completed: {section_name}")


            # --- Finalize Analysis ---
            st.session_state.analysis_results = all_validated_data # Assign combined results
            progress_bar.progress(100, text="Analysis process finished!")
            if overall_success:
                status_text.success("🏁 Analysis finished successfully!")
            else:
                status_text.warning("🏁 Analysis finished, but some sections encountered issues (see summary below).")
            st.session_state.analysis_complete = True # Mark analysis as done

        except Exception as main_err:
            st.error(f"❌ CRITICAL ERROR during analysis workflow: {main_err}")
            st.error("Check console logs for detailed traceback.")
            print(traceback.format_exc()) # Log for debugging
            overall_success = False
            st.session_state.analysis_complete = False # Ensure it's marked as not complete on error
            st.session_state.run_status_summary.append({
                "section": "Overall Process Control",
                "status": "Critical Error",
                "warnings": [str(main_err), "Analysis halted. See logs."]
            })
            status_text.error(f"Analysis stopped due to critical error: {main_err}")

        finally:
            # --- Cleanup ---
            st.session_state.processing_in_progress = False # Ensure flag is reset
            time.sleep(4) # Keep final status message visible briefly
            status_text.empty()
            progress_bar.empty()

            # Clean up the uploaded file from Google Cloud AI
            if gemini_uploaded_file_ref and hasattr(gemini_uploaded_file_ref, 'name'):
                try:
                    status_text.info(f"☁️ Deleting temporary cloud file: {gemini_uploaded_file_ref.name}...")
                    genai.delete_file(name=gemini_uploaded_file_ref.name) # Use configured client
                    st.toast("Cloud file deleted.", icon="🗑️")
                    time.sleep(1)
                    status_text.empty()
                except Exception as del_err:
                    st.sidebar.warning(f"Cloud cleanup issue: {del_err}", icon="⚠️")
                    status_text.warning(f"Could not delete cloud file: {del_err}") # Also show in main area
                    print(f"WARN: Failed to delete cloud file {gemini_uploaded_file_ref.name}: {del_err}")


            # Clean up the temporary local file
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as local_del_err:
                    st.sidebar.warning(f"Local temp file cleanup issue: {local_del_err}", icon="⚠️")
                    print(f"WARN: Failed to delete local temp file {temp_file_path}: {local_del_err}")

        st.rerun() # Rerun to update the UI based on completion state and results

# --- Display Area (Results and PDF Viewer) ---

# Check if PDF is loaded to display the layout
if st.session_state.pdf_bytes is not None:
    col1, col2 = st.columns([6, 4]) # Results | PDF Viewer

    # --- Column 1: Analysis Results ---
    with col1:
        # Display Run Status Summary Expander (if applicable)
        if st.session_state.run_status_summary:
            # Determine overall status icon based on all section results
            final_status = "✅ Success"
            has_failures = any(s['status'] == "Failed" or "Error" in s['status'] for s in st.session_state.run_status_summary)
            has_warnings = any(s['status'] != "Success" and not has_failures for s in st.session_state.run_status_summary)

            if has_failures: final_status = "❌ Failed"
            elif has_warnings: final_status = "⚠️ Issues"

            with st.expander(f"📊 Analysis Run Summary ({final_status})", expanded=(final_status != "✅ Success")):
                for item in st.session_state.run_status_summary:
                    icon = "✅" if item['status'] == "Success" else "❌" if item['status'] == "Failed" or "Error" in item['status'] else "⚠️"
                    st.markdown(f"**{item['section']}**: {icon} {item['status']}")
                    if item['warnings']:
                        # Filter out verbose validation messages for cleaner display
                        filtered_warnings = [msg for msg in item['warnings'] if not (isinstance(msg, str) and msg.startswith("Validation Issues Found"))]
                        if filtered_warnings:
                            with st.container():
                                st.caption("Details:")
                                for msg in filtered_warnings:
                                    msg_str = str(msg) # Ensure it's a string
                                    if "CRITICAL" in msg_str or "Error" in msg_str or "Block" in msg_str or "Fail" in msg_str:
                                        st.error(f" L> {msg_str}")
                                    elif "Warn" in msg_str or "Missing" in msg_str or "Unexpected" in msg_str or "Empty list" in msg_str or "validation issues" in msg_str.lower():
                                         st.warning(f" L> {msg_str}")
                                    else: # General info or less severe warnings
                                        st.caption(f" L> {msg_str}")

        st.subheader("Analysis Results")
        # Display results only if analysis is complete and results exist
        if st.session_state.analysis_complete and st.session_state.analysis_results:
            results_list = st.session_state.analysis_results

            # Group results by category
            grouped_results = defaultdict(list)
            # Maintain category order based on first appearance
            categories_ordered = []
            for item in results_list:
                # Use .get with default for safety
                category = item.get("Question Category", "Uncategorized")
                if category not in grouped_results:
                    categories_ordered.append(category)
                grouped_results[category].append(item)

            if categories_ordered:
                # Create tabs for each category
                category_tabs = st.tabs(categories_ordered)
                for i, category in enumerate(categories_ordered):
                    with category_tabs[i]:
                        # Sort items within the category by Question Number
                        category_items = sorted(grouped_results[category], key=lambda x: x.get('Question Number', float('inf')))

                        for index, result_item in enumerate(category_items):
                            q_num = result_item.get('Question Number', 'N/A')
                            question_text = result_item.get('Question', 'N/A')
                            unique_expander_key = f"expander_{category}_{q_num}_{index}" # More specific key

                            with st.expander(f"**Q{q_num}:** {question_text}", key=unique_expander_key):
                                st.markdown(f"**Answer:**")
                                st.markdown(f"> {result_item.get('Answer', 'N/A')}") # Use blockquote for answer
                                st.markdown("---")

                                # --- Evidence Section ---
                                evidence_list = result_item.get('Evidence', [])
                                if evidence_list:
                                    st.markdown("**Evidence:**")
                                    for ev_index, evidence_item in enumerate(evidence_list):
                                        clause_ref = evidence_item.get('Clause Reference', 'N/A')
                                        search_text = evidence_item.get('Searchable Clause Text', None)
                                        clause_wording = evidence_item.get('Clause Wording', 'N/A')
                                        base_key = f"ev_{category}_{q_num}_{index}_{ev_index}" # Base unique key

                                        ev_cols = st.columns([3, 1]) # Column for button, column for toggle

                                        with ev_cols[0]:
                                            # Button to trigger search
                                            if search_text:
                                                button_key = f"search_btn_{base_key}"
                                                button_label = f"Clause: **{clause_ref or 'Link'}** (Find & View)"
                                                if st.button(button_label, key=button_key, help=f"Search for text related to '{clause_ref or 'this evidence'}' and view the page."):
                                                    st.session_state.search_trigger = {'text': search_text, 'ref': clause_ref}
                                                    st.session_state.last_search_result = None # Clear previous search result
                                                    st.rerun() # Trigger rerun to perform search
                                            elif clause_ref != 'N/A':
                                                st.markdown(f"- Clause: **{clause_ref}** (No searchable text provided by AI)")
                                            else:
                                                 st.caption("No clause reference provided.")


                                        with ev_cols[1]:
                                            # Toggle to show/hide AI extracted wording
                                            if clause_wording != 'N/A':
                                                toggle_key = f"toggle_wording_{base_key}"
                                                # Use the defaultdict for state management
                                                show_wording = st.toggle(
                                                    "Show Wording",
                                                    key=toggle_key,
                                                    value=st.session_state.show_wording_states[toggle_key], # Get current state
                                                    help="Show/hide the exact clause wording extracted by the AI."
                                                )
                                                # Update the state if toggled
                                                if show_wording != st.session_state.show_wording_states[toggle_key]:
                                                    st.session_state.show_wording_states[toggle_key] = show_wording
                                                    st.rerun() # Rerun needed to reflect toggle change immediately

                                        # Display wording if toggle is on
                                        if st.session_state.show_wording_states.get(f"toggle_wording_{base_key}", False):
                                            st.text_area(
                                                f"AI Extracted Wording for '{clause_ref}':",
                                                value=clause_wording,
                                                height=150,
                                                disabled=True,
                                                key=f"wording_area_{base_key}"
                                            )
                                        st.markdown("---") # Separator between evidence items


                                else:
                                    st.markdown("**Evidence:** None provided.")
                                st.markdown("---") # Separator after evidence block

                                # --- Justification Section ---
                                st.markdown("**Answer Justification:**")
                                justification_text = result_item.get('Answer Justification', '')
                                just_key = f"justification_{category}_{q_num}_{index}"
                                st.text_area(
                                    label="Justification Text Area", # Needs a label, can be hidden
                                    value=justification_text,
                                    height=100,
                                    disabled=True,
                                    label_visibility="collapsed", # Hide the label visually
                                    key=just_key
                                )

            else:
                st.warning("Analysis generated results, but they could not be grouped by category. Displaying raw list.")
                st.json(results_list) # Fallback to show raw JSON

            # --- Excel Download Preparation (moved to Sidebar) ---
            st.sidebar.markdown("---")
            st.sidebar.markdown("## Export Results")
            if st.sidebar.button("Prepare Data for Excel Download", key="prep_excel", use_container_width=True):
                st.session_state.excel_data = None # Clear previous data
                excel_prep_status = st.sidebar.empty()
                excel_prep_status.info("Preparing Excel data...")
                try:
                    excel_rows = []
                    for item in results_list: # Use the full results list
                        references = []
                        first_search_text = "N/A"
                        evidence = item.get("Evidence")
                        if evidence: # Check if evidence exists and is a list
                            for i, ev in enumerate(evidence):
                                if isinstance(ev, dict): # Ensure evidence item is a dict
                                    references.append(str(ev.get("Clause Reference", "N/A")))
                                    if i == 0: # Get search text from the first evidence item only
                                        first_search_text = ev.get("Searchable Clause Text", "N/A")
                                else: # Handle case where evidence item is not a dict (unexpected)
                                    references.append("[Invalid Evidence Item]")


                        # Create dict for the row, ensuring all keys exist
                        excel_row = {
                            "File Name": item.get("File Name", ""),
                            "Generation Time": item.get("Generation Time", ""),
                            "Question Number": item.get("Question Number"), # Keep as is (int or None)
                            "Question Category": item.get("Question Category", "Uncategorized"),
                            "Question": item.get("Question", "N/A"),
                            "Answer": item.get("Answer", "N/A"),
                            "Answer Justification": item.get("Answer Justification", "N/A"),
                            "Clause References (Concatenated)": "; ".join(references) if references else "N/A",
                            "First Searchable Clause Text": first_search_text
                        }
                        excel_rows.append(excel_row)

                    if not excel_rows:
                        excel_prep_status.warning("No data available to export.")
                        st.session_state.excel_data = None
                    else:
                        # Create DataFrame and enforce column order
                        df_excel = pd.DataFrame(excel_rows)
                        # Ensure only existing columns are selected, preserving order
                        final_columns = [col for col in EXCEL_COLUMN_ORDER if col in df_excel.columns]
                        # Include any columns present in data but not in EXCEL_COLUMN_ORDER at the end
                        extra_cols = [col for col in df_excel.columns if col not in final_columns]
                        df_excel = df_excel[final_columns + extra_cols]

                        # Write to Excel in memory
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_excel.to_excel(writer, index=False, sheet_name='Analysis Results')
                        st.session_state.excel_data = output.getvalue()
                        excel_prep_status.success("✅ Excel file ready for download!")
                        time.sleep(2)
                        excel_prep_status.empty() # Clear status message

                except Exception as excel_err:
                    excel_prep_status.error(f"Excel Prep Error: {excel_err}")
                    print(traceback.format_exc()) # Log error

            # --- Actual Download Button (in Sidebar, shown only if data is ready) ---
            if st.session_state.excel_data:
                current_filename = "analysis_results" # Default base name
                # Try to get filename from results if available
                if st.session_state.analysis_results and isinstance(st.session_state.analysis_results, list) and len(st.session_state.analysis_results) > 0:
                     first_result = st.session_state.analysis_results[0]
                     if isinstance(first_result, dict) and "File Name" in first_result:
                           current_filename = first_result["File Name"]

                # Sanitize filename and add timestamp
                safe_base_name = re.sub(r'[^\w\s-]', '', os.path.splitext(current_filename)[0]).strip().replace(' ', '_')
                download_filename = f"Analysis_{safe_base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

                st.sidebar.download_button(
                    label="📥 Download Results as Excel",
                    data=st.session_state.excel_data,
                    file_name=download_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel_final",
                    use_container_width=True
                )

        elif st.session_state.analysis_complete and not st.session_state.analysis_results:
            st.info("Analysis process completed, but no valid results were generated. Check the run summary above for potential issues.")
        elif st.session_state.processing_in_progress:
            st.info("Analysis is currently in progress...")
        elif not st.session_state.analysis_complete and st.session_state.pdf_bytes is not None:
             st.info("PDF loaded. Click 'Analyse Document' in the sidebar to start.")
        # else: # Handled by the outer check for pdf_bytes

    # --- Column 2: PDF Viewer (Sticky) ---
    with col2:
        # Apply the sticky class using markdown - WRAPS ALL CONTENT IN COL2
        st.markdown('<div class="sticky-viewer-content">', unsafe_allow_html=True)

        st.subheader("📄 PDF Viewer")
        viewer_status_placeholder = st.empty() # For status messages like search results, errors

        # Perform Search if Triggered (before rendering page)
        if st.session_state.search_trigger:
            search_info = st.session_state.search_trigger
            st.session_state.search_trigger = None # Consume the trigger
            with st.spinner(f"🔎 Searching for text related to: '{search_info['ref']}'..."):
                # Use the updated function that returns all_findings
                found_page, instances, term_used, search_status, all_findings = find_text_in_pdf(
                    st.session_state.pdf_bytes, search_info['text']
                )

            if found_page:
                # Store results including potential multiple findings
                st.session_state.last_search_result = {
                    'page': found_page,
                    'instances': instances,
                    'term': term_used,
                    'status': search_status,
                    'ref': search_info['ref'],
                    'all_findings': all_findings # Will be None or a list [(page, inst), ...]
                }
                # Set current page to the first found page
                st.session_state.current_page = found_page
                viewer_status_placeholder.empty() # Clear searching message
                st.rerun() # Rerun to render the newly found page and potential multi-page buttons
            else:
                st.session_state.last_search_result = None # Clear any previous results
                viewer_status_placeholder.error(search_status) # Show search failure message

        # --- Render PDF Page ---
        if st.session_state.pdf_display_ready: # Check if PDF is ready to be displayed
            try:
                # Get total pages safely
                with fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf") as doc:
                    total_pages = doc.page_count
            except Exception as pdf_load_err:
                st.error(f"Error loading PDF for page count: {pdf_load_err}")
                total_pages = 1 # Assume 1 page on error to avoid breaking navigation
                st.session_state.current_page = 1

            # --- Page Navigation ---
            # Ensure current page is within valid bounds
            current_display_page = max(1, min(int(st.session_state.get('current_page', 1)), total_pages))
            if current_display_page != st.session_state.get('current_page'):
                st.session_state.current_page = current_display_page # Correct if out of bounds


            nav_cols = st.columns([1, 3, 1]) # Prev | Page Info | Next
            with nav_cols[0]: # Previous Button
                if st.button("⬅️ Prev", key="prev_page", disabled=(current_display_page <= 1), use_container_width=True):
                    st.session_state.current_page -= 1
                    st.session_state.last_search_result = None # Clear search context on manual navigation
                    st.rerun()
            with nav_cols[2]: # Next Button
                if st.button("Next ➡️", key="next_page", disabled=(current_display_page >= total_pages), use_container_width=True):
                    st.session_state.current_page += 1
                    st.session_state.last_search_result = None # Clear search context
                    st.rerun()

            # Display Page Info in the middle column
            page_info_text = f"Page {current_display_page} of {total_pages}"
            search_context_ref = None
            if st.session_state.last_search_result and st.session_state.last_search_result.get('all_findings'):
                # If multiple pages found, indicate context persists across them
                search_context_ref = st.session_state.last_search_result['ref']
                page_info_text += f" (🎯 Multi-match: '{search_context_ref}')"
            elif st.session_state.last_search_result and st.session_state.last_search_result['page'] == current_display_page:
                 # Single page match or currently viewing one of the multi-match pages
                 search_context_ref = st.session_state.last_search_result['ref']
                 page_info_text += f" (🎯 Ref: '{search_context_ref}')"

            nav_cols[1].markdown(f"<div style='text-align: center; padding-top: 0.5rem;'>{page_info_text}</div>", unsafe_allow_html=True)


            # --- Multi-Page Match Navigation ---
            if st.session_state.last_search_result and st.session_state.last_search_result.get('all_findings'):
                multi_findings = st.session_state.last_search_result['all_findings']
                found_pages = sorted([f[0] for f in multi_findings])

                # Display status message about multiple matches
                status_msg = st.session_state.last_search_result['status']
                viewer_status_placeholder.info(status_msg) # Show the multi-match status

                st.write("Jump to other matches for this reference:")
                # Use columns for buttons - adjust number based on expected max matches?
                num_buttons = len(found_pages)
                btn_cols = st.columns(num_buttons)

                for idx, p_num in enumerate(found_pages):
                    # Disable button if it's the currently viewed page
                    is_current = (p_num == current_display_page)
                    if btn_cols[idx].button(f"Page {p_num}", key=f"jump_{p_num}_{search_context_ref}", disabled=is_current, use_container_width=True):
                        st.session_state.current_page = p_num
                        # IMPORTANT: Update last_search_result to reflect the *new* page's instances for highlighting
                        new_instances = next((inst for pg, inst in multi_findings if pg == p_num), None)
                        st.session_state.last_search_result['instances'] = new_instances
                        st.session_state.last_search_result['page'] = p_num
                        st.rerun()


            st.markdown("---") # Separator before the image

            # --- Determine Highlights & Render ---
            highlights_to_apply = None
            render_status_override = None # For specific search status messages

            # Check if current page matches the last search result (single or multi-page)
            if st.session_state.last_search_result and st.session_state.last_search_result.get('page') == current_display_page:
                highlights_to_apply = st.session_state.last_search_result.get('instances')
                # Use search status only if NOT a multi-match (multi-match status shown above)
                if not st.session_state.last_search_result.get('all_findings'):
                     render_status_override = st.session_state.last_search_result.get('status')


            # Render the image
            image_bytes, render_status = render_pdf_page_to_image(
                st.session_state.pdf_bytes,
                current_display_page,
                highlight_instances=highlights_to_apply,
                dpi=150 # Adjust DPI if needed
            )

            # --- Display Image & Final Status ---
            if image_bytes:
                st.image(image_bytes, caption=f"Page {current_display_page} - View", use_column_width='always')
                final_status = render_status_override if render_status_override else render_status

                # Update status placeholder *unless* it's already showing multi-match info
                if not (st.session_state.last_search_result and st.session_state.last_search_result.get('all_findings')):
                    if final_status:
                        if "✅" in final_status or "✨" in final_status or "Found" in final_status :
                            viewer_status_placeholder.success(final_status)
                        elif "⚠️" in final_status or "warning" in final_status.lower():
                             viewer_status_placeholder.warning(final_status)
                        elif "❌" in final_status or "error" in final_status.lower():
                             viewer_status_placeholder.error(final_status)
                        else: # Plain render message
                            viewer_status_placeholder.caption(final_status)
                    else:
                         # Clear status if render was ok but no specific message, and not showing multi-match
                         viewer_status_placeholder.empty()

            else:
                # Show rendering error specifically
                viewer_status_placeholder.error(f"Failed to render page {current_display_page}. {render_status or ''}")

        else:
            # This condition should ideally not be met if pdf_bytes is not None,
            # but acts as a fallback if pdf_display_ready isn't set properly.
            st.info("PDF loaded, preparing viewer...")
            viewer_status_placeholder.empty()

        # Close the sticky div
        st.markdown('</div>', unsafe_allow_html=True)


# --- Fallback messages if no PDF is loaded ---
elif not st.session_state.pdf_bytes:
     st.info("⬆️ Upload a PDF file using the sidebar to begin.")

# # app.py
# # --- COMPLETE FILE ---

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

# # --- 1. SET PAGE CONFIG (MUST BE FIRST st COMMAND) ---
# st.set_page_config(
#     layout="wide",
#     page_title="Just A Smart Platform for Extraction and Review",
#     page_icon="📄" # Optional: Set an emoji icon
# )

# # --- Inject custom CSS for sticky column (AFTER set_page_config) ---
# st.markdown("""
# <style>
#     /* Define a class for the sticky container */
#     .sticky-viewer-content {
#         position: sticky;
#         top: 55px; /* Adjust vertical offset from top */
#         z-index: 101; /* Ensure it's above other elements */
#         padding-bottom: 1rem; /* Add some space at the bottom */
#     }
#     /* Ensure background color matches theme for sticky element */
#     div.sticky-viewer-content {
#         background-color: var(--streamlit-background-color);
#     }
# </style>
# """, unsafe_allow_html=True)

# # --- 2. Configuration & Setup ---
# # API_KEY_ENV_VAR = "GEMINI_API_KEY" # No longer reading from ENV at startup
# MODEL_NAME = "gemini-2.5-pro-preview-03-25" # Ensure this model is appropriate for your key/access
# MAX_VALIDATION_RETRIES = 1
# RETRY_DELAY_SECONDS = 3
# PROMPT_FILE = "prompt.txt"
# LOGO_FILE = "jasper-logo-1.png"
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
# SECTIONS_TO_RUN = { "agreement_details": (1, 4), "eligibility": (5, 36), "confidentiality": (37, 63), "additional_borrowers": (64, 66), "interest_rate_provisions": (67, 71), "prepayment_fee": (72, 78) }

# # --- System Instruction ---
# system_instruction_text = """You are analysing a facility agreement... Adhere strictly to the JSON schema...""" # Use your full instruction


# # --- 3. Helper Function Definitions ---

# def filter_prompt_by_section(initial_full_prompt, section):
#     # ... (Keep your existing function code - no changes needed) ...
#     # (Ensure it doesn't rely on globally configured genai)
#     if section not in SECTIONS_TO_RUN: raise ValueError(f"Invalid section: {section}")
#     start_q, end_q = SECTIONS_TO_RUN[section]; questions_start_marker = "**Questions to Answer:**"; questions_end_marker = "**Final Instruction:**"
#     try: start_index = initial_full_prompt.index(questions_start_marker); end_index = initial_full_prompt.index(questions_end_marker)
#     except ValueError: raise ValueError("Prompt markers not found.")
#     prompt_header = initial_full_prompt[:start_index]; full_questions_block = initial_full_prompt[start_index + len(questions_start_marker):end_index].strip(); prompt_footer = initial_full_prompt[end_index:]
#     question_entries = re.split(r'\n(?=\s*\d+\.\s*?\*\*Question Category:)', full_questions_block);
#     if not question_entries or len(question_entries) < 2: # Fallback splits
#         question_entries = re.split(r'\n(?=\d+\.\s)', full_questions_block)
#         if not question_entries or len(question_entries) < 2: question_entries = re.split(r'\n(?=\*\*Question Category:)', full_questions_block)
#     filtered_question_texts = []; processed_q_nums = set()
#     for entry in question_entries:
#         entry = entry.strip(); match = re.match(r'^\s*(\d+)\.', entry)
#         if match: q_num = int(match.group(1));
#         else: continue
#         if start_q <= q_num <= end_q: filtered_question_texts.append(entry); processed_q_nums.add(q_num)
#     if not filtered_question_texts: raise ValueError(f"No questions found for section '{section}' in range {start_q}-{end_q}.")
#     filtered_questions_string = "\n\n".join(filtered_question_texts)
#     section_note = f"\n\n**Current Focus:** Answering ONLY questions for the '{section.upper()}' section (Questions {start_q}-{end_q})...\n";
#     final_prompt_for_api = f"{prompt_header}{section_note}{questions_start_marker}\n\n{filtered_questions_string}\n\n{prompt_footer}"
#     return final_prompt_for_api


# def validate_ai_data(data, section_name):
#     """Validates AI response. Returns (validated_data, issues_list)."""
#     # ... (Keep updated validation logic from previous step) ...
#     if not isinstance(data, list): return None, [f"CRITICAL: Response for '{section_name}' not list."]
#     validated_data = []; issues_list = []
#     for index, item in enumerate(data):
#         q_num_str = f"Q#{item.get('Question Number', f'Item {index}')}"; is_outer_valid = True
#         if not isinstance(item, dict): issues_list.append(f"{q_num_str}: Not dict."); continue
#         missing_outer_keys = AI_REQUIRED_KEYS - set(item.keys())
#         if missing_outer_keys: issues_list.append(f"{q_num_str}: Missing: {missing_outer_keys}"); is_outer_valid = False
#         evidence_list = item.get("Evidence")
#         if not isinstance(evidence_list, list): issues_list.append(f"{q_num_str}: 'Evidence' not list."); is_outer_valid = False
#         else:
#             for ev_index, ev_item in enumerate(evidence_list):
#                 if not isinstance(ev_item, dict): issues_list.append(f"{q_num_str} Ev[{ev_index}]: Not dict."); is_outer_valid = False; continue
#                 missing_ev_keys = AI_EVIDENCE_REQUIRED_KEYS - set(ev_item.keys())
#                 if missing_ev_keys: issues_list.append(f"{q_num_str} Ev[{ev_index}]: Missing: {missing_ev_keys}"); is_outer_valid = False
#                 if not isinstance(ev_item.get("Clause Reference"), str): issues_list.append(f"{q_num_str} Ev[{ev_index}]: Ref not str."); is_outer_valid = False
#                 if not isinstance(ev_item.get("Clause Wording"), str): issues_list.append(f"{q_num_str} Ev[{ev_index}]: Wording not str."); is_outer_valid = False
#                 if not isinstance(ev_item.get("Searchable Clause Text"), str): issues_list.append(f"{q_num_str} Ev[{ev_index}]: SearchText not str."); is_outer_valid = False
#         if is_outer_valid: validated_data.append(item)
#     if issues_list: issues_list.insert(0, f"Validation Issues [Section: {section_name}] ({len(validated_data)} passed):")
#     return validated_data, issues_list


# def generate_section_analysis(section, uploaded_file_ref, status_placeholder, api_key_to_use):
#     """Generates analysis for a section using a specific API key."""
#     # --- Configure GenAI client INSIDE the function for this specific call ---
#     try:
#         genai.configure(api_key=api_key_to_use)
#         # Optionally add a quick check like listing models to verify key here if needed
#     except Exception as config_err:
#         status_placeholder.error(f"❌ Invalid API Key provided or configuration failed: {config_err}")
#         return None, "Failed", [f"Invalid API Key or config error: {config_err}"]
#     # --- End GenAI Configure ---

#     status_placeholder.info(f"🔄 Starting: {section}...")
#     section_warnings = []
#     # --- Model Initialization uses the now-configured client ---
#     model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction_text)
#     generation_config = types.GenerationConfig(
#         response_mime_type="application/json", response_schema=ai_response_schema_dict,
#         temperature=0.0, top_p=0.05, top_k=1
#     )
#     safety_settings = [ {"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
#     final_validated_data = None

#     for attempt in range(1, 1 + MAX_VALIDATION_RETRIES + 1):
#         if attempt > 1: status_placeholder.info(f"⏳ Retrying: '{section}' (Attempt {attempt})..."); time.sleep(RETRY_DELAY_SECONDS)
#         try:
#             prompt_for_api = filter_prompt_by_section(full_prompt_text, section)
#             contents = [uploaded_file_ref, prompt_for_api]
#             status_placeholder.info(f"🧠 Calling AI: {section} (Attempt {attempt})...");
#             response = model.generate_content(contents=contents, generation_config=generation_config, safety_settings=safety_settings)
#             parsed_ai_data = None; validated_ai_data = None; validation_issues = []
#             status_placeholder.info(f"🔍 Processing: {section}...");
#             if response.parts:
#                 full_response_text = response.text
#                 try: # Strip potential markdown ```json ... ```
#                     match = re.search(r"```json\s*([\s\S]*?)\s*```", full_response_text); json_text = match.group(1).strip() if match else full_response_text.strip()
#                     parsed_ai_data = json.loads(json_text)
#                     status_placeholder.info(f"✔️ Validating: {section}..."); validated_ai_data, validation_issues = validate_ai_data(parsed_ai_data, section); section_warnings.extend(validation_issues)
#                     if validated_ai_data is not None and len(validated_ai_data) > 0: final_validated_data = validated_ai_data; break
#                     elif validated_ai_data is None: error_msg = f"Crit Val Error: '{section}'."; section_warnings.append(error_msg); break
#                     else: status_placeholder.warning(f"⚠️ Validation 0 items: {section} (Attempt {attempt}).") # Retry if possible
#                 except json.JSONDecodeError as json_err: error_msg = f"JSON Error {attempt}: {section}: {json_err}"; st.error(error_msg); st.code(full_response_text); section_warnings.append(error_msg)
#             else: # Handle empty/blocked response
#                 block_reason = "Unknown"; block_message = "N/A"; finish_reason = "Unknown";
#                 try: # Safe access to feedback
#                    if response.prompt_feedback: block_reason = getattr(response.prompt_feedback.block_reason, 'name', 'Unknown'); block_message = response.prompt_feedback.block_reason_message or "N/A"
#                    if response.candidates: finish_reason = getattr(response.candidates[0].finish_reason, 'name', 'Unknown')
#                 except AttributeError: pass
#                 if finish_reason == "SAFETY": warn_msg = f"Blocked Response {attempt}: {section}. Reason: SAFETY. Block Detail: {block_reason}. Msg: {block_message}"; st.error(warn_msg)
#                 else: warn_msg = f"API Issue {attempt}: {section}. Finish Reason: {finish_reason}. Block Reason: {block_reason}."; st.warning(warn_msg)
#                 section_warnings.append(warn_msg)
#         except ValueError as ve: error_msg = f"Data/Prompt Error {attempt}: {section}: {ve}"; st.error(error_msg); section_warnings.append(error_msg); break # Stop section
#         except Exception as e: error_msg = f"Processing Error {attempt}: {section}: {type(e).__name__}"; st.error(error_msg); section_warnings.append(error_msg); section_warnings.append(f"Traceback in logs."); print(traceback.format_exc());
#         if final_validated_data is not None: break # Exit loop if successful

#     if final_validated_data is not None: status_placeholder.info(f"✅ Completed: {section}."); return final_validated_data, "Success", section_warnings
#     else: status_placeholder.error(f"❌ Failed: {section} after {attempt} attempts."); return None, "Failed", section_warnings


# @st.cache_data(show_spinner=False)
# def find_text_in_pdf(_pdf_bytes, search_text):
#     """Searches PDF using prioritized strategy. Returns (page_num, instances, term_used, status_msg)."""
#     # ... (Keep updated find_text_in_pdf logic from previous step) ...
#     if not _pdf_bytes or not search_text: return None, None, None, "Invalid input."
#     doc = None; search_text_cleaned = search_text.strip(); words = search_text_cleaned.split(); num_words = len(words); search_attempts = []
#     # Build search terms list
#     if num_words >= SEARCH_PREFIX_MIN_WORDS: term_10 = ' '.join(words[:10]); search_attempts.append({'term': term_10, 'desc': "first 10 words"})
#     if num_words >= SEARCH_PREFIX_MIN_WORDS: term_5 = ' '.join(words[:5]); # Only add if different
#     if term_5 and term_5 != (search_attempts[0]['term'] if search_attempts else None): search_attempts.append({'term': term_5, 'desc': "first 5 words"})
#     term_full = search_text_cleaned; # Add full text if different from prefixes
#     if term_full and not any(term_full == a['term'] for a in search_attempts): search_attempts.append({'term': term_full, 'desc': "full text"})
#     # Add first sentence if long enough and different
#     sentences = re.split(r'(?<=[.?!])\s+', term_full); term_sentence = sentences[0].strip() if sentences else ""
#     if len(term_sentence) >= SEARCH_FALLBACK_MIN_LENGTH and not any(term_sentence == a['term'] for a in search_attempts): search_attempts.append({'term': term_sentence, 'desc': "first sentence"})
#     # Execute search
#     try:
#         doc = fitz.open(stream=_pdf_bytes, filetype="pdf"); search_flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES
#         for attempt in search_attempts:
#             term = attempt['term']; desc = attempt['desc']
#             for page_index in range(doc.page_count):
#                 instances = doc.load_page(page_index).search_for(term, flags=search_flags, quads=False)
#                 if instances: doc.close(); status = f"✨ Found using {desc} on page {page_index + 1}."; return page_index + 1, instances, term, status
#         doc.close(); return None, None, None, f"❌ Text not found (tried {len(search_attempts)} methods)."
#     except Exception as e:
#         if doc: doc.close(); print(f"ERROR searching PDF: {e}"); return None, None, None, f"❌ Error during PDF search: {e}"


# # @st.cache_data(show_spinner=False) # Remove caching as it calls st elements implicitly via toast/log
# def render_pdf_page_to_image(_pdf_bytes, page_number, highlight_instances=None, dpi=150):
#     """Renders PDF page to image, applying highlights. Returns (image_bytes, status_msg)."""
#     # ... (Keep updated render function, WITHOUT st calls, returns status message) ...
#     if not _pdf_bytes or page_number < 1: return None, "Invalid input."
#     doc = None; image_bytes = None; render_status_message = f"Rendered page {page_number}."
#     try:
#         doc = fitz.open(stream=_pdf_bytes, filetype="pdf"); page_index = page_number - 1
#         if page_index < 0 or page_index >= doc.page_count: doc.close(); return None, f"Page {page_number} out of range."
#         page = doc.load_page(page_index); highlight_applied = False; highlight_count = 0
#         if highlight_instances:
#             try:
#                 for inst in highlight_instances:
#                     if isinstance(inst, (fitz.Rect, fitz.Quad)):
#                         highlight = page.add_highlight_annot(inst); highlight.set_colors(stroke=fitz.utils.getColor("red"), fill=fitz.utils.getColor("yellow")); highlight.set_opacity(0.4); highlight.update(); highlight_count += 1
#                     else: print(f"WARN: Invalid highlight instance type: {type(inst)}")
#                 if highlight_count > 0: highlight_applied = True; render_status_message = f"✨ Rendered page {page_number} with {highlight_count} highlight(s)."
#                 elif highlight_instances: render_status_message = f"⚠️ Rendered page {page_number}, but no valid highlights found in provided instances."
#             except Exception as highlight_err: print(f"ERROR applying highlights: {highlight_err}"); render_status_message = f"⚠️ Error applying highlights: {highlight_err}"
#         pix = page.get_pixmap(dpi=dpi, alpha=False); image_bytes = pix.tobytes("png")
#     except Exception as e: print(f"ERROR rendering page {page_number}: {e}"); render_status_message = f"❌ Error rendering page: {e}"; image_bytes = None
#     finally:
#         if doc: doc.close()
#     return image_bytes, render_status_message


# # --- 4. Initialize Session State ---
# # (Ensure all keys used are initialized)
# if 'current_page' not in st.session_state: st.session_state.current_page = 1
# if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
# if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None
# if 'pdf_bytes_processed' not in st.session_state: st.session_state.pdf_bytes_processed = None
# if 'processing_complete' not in st.session_state: st.session_state.processing_complete = False
# if 'run_key' not in st.session_state: st.session_state.run_key = 0
# if 'run_status_summary' not in st.session_state: st.session_state.run_status_summary = []
# if 'excel_data' not in st.session_state: st.session_state.excel_data = None
# if 'search_trigger' not in st.session_state: st.session_state.search_trigger = None
# if 'last_search_result' not in st.session_state: st.session_state.last_search_result = None
# if 'api_key' not in st.session_state: st.session_state.api_key = None # Store user API key


# # --- 5. Streamlit UI Logic ---

# st.title("JASPER - Just A Smart Platform for Extraction and Review")
# st.markdown("Upload a PDF agreement, **enter your Gemini API Key**, click 'Analyse'. Results grouped below. Click clause references to view & highlight.")

# # --- Sidebar Setup ---
# st.sidebar.markdown("## Controls")

# # --- API Key Input (in Sidebar) ---
# api_key_input = st.sidebar.text_input(
#     "Enter your Google AI Gemini API Key",
#     type="password",
#     key="api_key_input",
#     help="Your API key is used only for this session and is not stored.",
#     value=st.session_state.get("api_key", "") # Pre-fill if already entered in session
# )
# if api_key_input:
#     st.session_state.api_key = api_key_input # Update session state when user types

# # Display warning if key not entered
# if not st.session_state.api_key:
#     st.sidebar.warning("API Key required to run analysis.", icon="⚠️")


# # --- File Upload (in Sidebar) ---
# uploaded_file_obj = st.sidebar.file_uploader(
#     "Upload Facility Agreement PDF", type="pdf", key=f"pdf_uploader_{st.session_state.run_key}"
# )

# # Process new file upload
# if uploaded_file_obj is not None:
#     uploaded_bytes = uploaded_file_obj.getvalue()
#     if uploaded_bytes != st.session_state.get('pdf_bytes_processed'):
#         st.session_state.pdf_bytes = uploaded_bytes
#         st.session_state.pdf_bytes_processed = uploaded_bytes
#         # Reset states dependent on file
#         st.session_state.analysis_results = None; st.session_state.processing_complete = False
#         st.session_state.current_page = 1; st.session_state.run_status_summary = []
#         st.session_state.excel_data = None; st.session_state.search_trigger = None; st.session_state.last_search_result = None
#         st.toast("✅ New PDF file loaded.", icon="📄")
#         st.rerun()
# elif 'pdf_bytes_processed' in st.session_state:
#      st.session_state.pdf_bytes_processed = None


# # --- Analysis Trigger (in Sidebar) ---
# analyse_disabled = st.session_state.processing_complete or not st.session_state.pdf_bytes or not st.session_state.api_key
# analyse_button_tooltip = "Upload a PDF and enter API key to enable analysis." if analyse_disabled else "Start analyzing the document"

# if st.sidebar.button("✨ Analyse Document", key="analyse_button", disabled=analyse_disabled, help=analyse_button_tooltip, use_container_width=True, type="primary"):
#     # --- GET and VALIDATE API Key ---
#     current_api_key = st.session_state.get("api_key")
#     if not current_api_key:
#         st.error("API Key is missing. Please enter it in the sidebar.")
#         st.stop() # Stop if key removed after button enabled somehow

#     # --- CONFIGURE GenAI Client HERE (Before Upload) ---
#     try:
#         genai.configure(api_key=current_api_key)
#         # Optional: Add a quick test like listing models to verify key validity early
#         # models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
#         st.toast("API Key configured for this run.", icon="🔑")
#     except Exception as config_err:
#         st.error(f"❌ Failed to configure Gemini API with provided key: {config_err}")
#         st.stop() # Stop if configuration fails
#     # --- End Configure GenAI ---


#     # Reset states for the new run
#     st.session_state.analysis_results = None; st.session_state.processing_complete = False
#     st.session_state.current_page = 1; st.session_state.run_key += 1; st.session_state.run_status_summary = []
#     st.session_state.excel_data = None; st.session_state.search_trigger = None; st.session_state.last_search_result = None

#     run_start_time = datetime.now(); run_timestamp_str = run_start_time.strftime("%Y-%m-%d %H:%M:%S")
#     base_file_name = getattr(uploaded_file_obj, 'name', 'uploaded_file')

#     # Status placeholders in main area
#     status_container = st.container()
#     progress_bar = status_container.progress(0, text="Initializing...")
#     status_text = status_container.empty()

#     temp_dir = "temp_uploads"; temp_file_path = os.path.join(APP_DIR, temp_dir, f"{run_start_time.strftime('%Y%m%d%H%M%S')}_{base_file_name}")
#     os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

#     gemini_uploaded_file_ref = None; all_validated_data = []; overall_success = True

#     try:
#         status_text.info("💾 Saving file temporarily..."); progress_bar.progress(5, text="Saving...")
#         with open(temp_file_path, "wb") as f: f.write(st.session_state.pdf_bytes)

#         status_text.info("🚀 Uploading to Google Cloud..."); progress_bar.progress(10, text="Uploading...")
#         # --- Upload File (Now client should be configured) ---
#         for upload_attempt in range(3):
#             try:
#                 # This call now uses the client configured above
#                 gemini_uploaded_file_ref = genai.upload_file(path=temp_file_path)
#                 st.toast(f"File '{gemini_uploaded_file_ref.display_name}' uploaded.", icon="☁️")
#                 break # Success
#             except Exception as upload_err:
#                 # Check if the error is related to authentication
#                 if "API key" in str(upload_err).lower() or "authenticat" in str(upload_err).lower():
#                      st.error(f"❌ File upload failed due to API key issue: {upload_err}")
#                      st.error("Please verify the entered API key has File API permissions.")
#                      raise # Stop the process immediately
#                 elif upload_attempt < 2:
#                     st.warning(f"Upload failed (Try {upload_attempt+1}): {upload_err}. Retrying...")
#                     time.sleep(2)
#                 else:
#                     st.error(f"Upload failed after multiple attempts: {upload_err}")
#                     raise # Re-raise the last error
#         if not gemini_uploaded_file_ref:
#              raise Exception("Failed Gemini upload after retries.")
#         # --- End Upload File ---

#         progress_bar.progress(15, text="Uploaded.")

#         num_sections = len(SECTIONS_TO_RUN); progress_per_section = (95 - 15) / num_sections if num_sections > 0 else 0

#         # --- Process Sections ---
#         for i, section_name in enumerate(SECTIONS_TO_RUN.keys()):
#             current_progress = int(15 + (i * progress_per_section))
#             progress_bar.progress(current_progress, text=f"Starting {section_name}...")
#             # Pass the user's API key - generate_section_analysis will re-configure,
#             # which is slightly redundant but safe. Alternatively, remove configure
#             # from generate_section_analysis IF it's only called here.
#             section_data, section_status, section_warnings = generate_section_analysis(
#                 section_name, gemini_uploaded_file_ref, status_text, current_api_key
#             )
#             st.session_state.run_status_summary.append({"section": section_name, "status": section_status, "warnings": section_warnings})
#             if section_status == "Success" and section_data:
#                 for item in section_data: item["File Name"] = base_file_name; item["Generation Time"] = run_timestamp_str
#                 all_validated_data.extend(section_data)
#             else: overall_success = False

#         st.session_state.analysis_results = all_validated_data
#         progress_bar.progress(100, text="Analysis Complete!")
#         if overall_success: status_text.success("🏁 Analysis finished successfully!")
#         else: status_text.warning("🏁 Analysis finished with some issues (see summary below).")
#         st.session_state.processing_complete = True

#     except Exception as main_err:
#          # Error handling remains the same
#          st.error(f"❌ CRITICAL ERROR during analysis: {main_err}"); # st.error(traceback.format_exc()); # Keep traceback off for users
#          overall_success = False; st.session_state.processing_complete = False
#          st.session_state.run_status_summary.append({"section": "Overall Process", "status": "Critical Error", "warnings": [str(main_err), "Check server logs for details."]})
#          status_text.error(f"Analysis failed: {main_err}") # Show specific error
#     finally: # Cleanup
#         time.sleep(4); status_text.empty(); progress_bar.empty()
#         if gemini_uploaded_file_ref and hasattr(gemini_uploaded_file_ref, 'name'):
#              try: genai.delete_file(name=gemini_uploaded_file_ref.name) # Use configured client
#              except Exception as del_err: st.sidebar.warning(f"Cloud cleanup issue: {del_err}", icon="⚠️")
#         if os.path.exists(temp_file_path):
#              try: os.remove(temp_file_path)
#              except Exception: pass

#     st.rerun() # Rerun to update display


# # --- Run Status Summary Expander (Displayed near the top) ---
# if st.session_state.run_status_summary:
#     # Determine overall status icon
#     final_status = "✅ Success"
#     if any(s['status'] != "Success" for s in st.session_state.run_status_summary): final_status = "⚠️ Issues"
#     if any("Critical" in s['status'] or "Fail" in s['status'] for s in st.session_state.run_status_summary): final_status = "❌ Failed"
#     with st.expander(f"📊 Last Analysis Run Summary ({final_status})", expanded=(final_status != "✅ Success")):
#         for item in st.session_state.run_status_summary:
#             icon = "✅" if item['status'] == "Success" else "❌" if "Fail" in item['status'] or "Error" in item['status'] else "⚠️"
#             st.markdown(f"**{item['section']}**: {icon} {item['status']}")
#             if item['warnings']:
#                  with st.container():
#                      filtered_warnings = [msg for msg in item['warnings'] if not (isinstance(msg, str) and msg.startswith("Validation Issues [Section:"))]
#                      if filtered_warnings:
#                          st.caption("Details:")
#                          for msg in filtered_warnings:
#                             if isinstance(msg, str) and msg.startswith(("CRITICAL:", "JSON Error", "API Block", "API Stop")): st.error(f" L> {msg}")
#                             elif isinstance(msg, str) and msg.startswith(("Empty Response", "Missing:", "Crit Val Error")): st.warning(f" L> {msg}")
#                             elif isinstance(msg, list): st.warning(" L> "+"\n L> ".join(map(str, msg)))
#                             else: st.caption(f" L> {msg}")


# # --- Display Area (Results and PDF Viewer) ---
# if st.session_state.analysis_results is not None:
#     col1, col2 = st.columns([6, 4]) # Results | PDF Viewer

#     # --- Column 1: Analysis Results (Grouped by Category using Tabs) ---
#     with col1:
#         st.subheader("Analysis Results")
#         results_list = st.session_state.analysis_results

#         if not results_list:
#             st.info("Analysis complete, but no valid results were generated. Check the summary above.")
#         else:
#             # Group results by category
#             grouped_results = defaultdict(list)
#             categories_ordered = []
#             for item in results_list:
#                 category = item.get("Question Category", "Uncategorized")
#                 if category not in grouped_results: categories_ordered.append(category)
#                 grouped_results[category].append(item)

#             if categories_ordered:
#                 category_tabs = st.tabs(categories_ordered)
#                 for i, category in enumerate(categories_ordered):
#                     with category_tabs[i]:
#                         category_items = grouped_results[category]
#                         category_items.sort(key=lambda x: x.get('Question Number', float('inf'))) # Sort by Q#

#                         for index, result_item in enumerate(category_items):
#                             q_num = result_item.get('Question Number', 'N/A'); question_text = result_item.get('Question', 'N/A')

#                             with st.expander(f"**Q{q_num}:** {question_text}"): # Removed explicit key
#                                 st.markdown(f"**Answer:** {result_item.get('Answer', 'N/A')}")
#                                 st.markdown("---")

#                                 # Evidence Section
#                                 evidence_list = result_item.get('Evidence', [])
#                                 if evidence_list:
#                                     st.markdown("**Evidence:** (Click reference to find & view page with highlights)")
#                                     for ev_index, evidence_item in enumerate(evidence_list):
#                                         clause_ref = evidence_item.get('Clause Reference', 'N/A')
#                                         search_text = evidence_item.get('Searchable Clause Text', None) # Get search text

#                                         # Button to trigger search
#                                         if search_text:
#                                             button_key = f"search_btn_{category}_{q_num}_{index}_{ev_index}" # Unique key
#                                             button_label = f"Clause: **{clause_ref or 'Link'}** (Find & View)"
#                                             if st.button(button_label, key=button_key, help=f"Search for text related to '{clause_ref or 'this evidence'}' and view the page."):
#                                                 st.session_state.search_trigger = {'text': search_text, 'ref': clause_ref}
#                                                 st.session_state.last_search_result = None
#                                                 st.rerun()
#                                         elif clause_ref != 'N/A':
#                                             st.markdown(f"- Clause: **{clause_ref}** (No searchable text provided by AI)")
#                                         # Removed Wording Display Here

#                                 else: st.markdown("**Evidence:** None provided.")
#                                 st.markdown("---") # Separator after evidence block

#                                 # Justification Section
#                                 st.markdown("**Answer Justification:**")
#                                 just_key = f"justification_{category}_{q_num}_{index}"
#                                 st.text_area(label="Justification", value=result_item.get('Answer Justification', ''), height=100, disabled=True, label_visibility="collapsed", key=just_key)

#             else: st.warning("Results generated, but could not group by category.")


#             # --- Excel Download Preparation (in Sidebar) ---
#             st.sidebar.markdown("---")
#             st.sidebar.markdown("## Export")
#             if st.sidebar.button("Prepare Data for Excel Download", key="prep_excel", use_container_width=True):
#                 st.session_state.excel_data = None; excel_prep_status = st.sidebar.empty()
#                 excel_prep_status.info("Preparing Excel...")
#                 try:
#                     excel_rows = [];
#                     for item in results_list: # Use full results list
#                         references = []; first_search_text = "N/A"
#                         evidence = item.get("Evidence");
#                         if evidence:
#                             for i, ev in enumerate(evidence):
#                                 references.append(str(ev.get("Clause Reference", "N/A")))
#                                 if i == 0: first_search_text = ev.get("Searchable Clause Text", "N/A")
#                         excel_row = {"File Name": item.get("File Name", ""), "Generation Time": item.get("Generation Time", ""),
#                                      "Question Number": item.get("Question Number"), "Question Category": item.get("Question Category"),
#                                      "Question": item.get("Question"), "Answer": item.get("Answer"), "Answer Justification": item.get("Answer Justification"),
#                                      "Clause References (Concatenated)": "; ".join(references) if references else "N/A",
#                                      "First Searchable Clause Text": first_search_text}
#                         excel_rows.append(excel_row)

#                     if not excel_rows: excel_prep_status.warning("No data to export."); st.session_state.excel_data = None
#                     else:
#                         df_excel = pd.DataFrame(excel_rows); final_columns = []
#                         for col in EXCEL_COLUMN_ORDER: # Apply column order
#                             if col in df_excel.columns: final_columns.append(col)
#                         df_excel = df_excel[final_columns]
#                         output = io.BytesIO();
#                         with pd.ExcelWriter(output, engine='openpyxl') as writer: df_excel.to_excel(writer, index=False, sheet_name='Analysis')
#                         st.session_state.excel_data = output.getvalue()
#                         excel_prep_status.success("Excel ready!")
#                         time.sleep(2); excel_prep_status.empty()
#                 except Exception as excel_err: excel_prep_status.error(f"Excel Prep Error: {excel_err}"); print(traceback.format_exc())

#             # --- Actual Download Button (in Sidebar) ---
#             if st.session_state.excel_data:
#                  current_filename = "analysis" # Default filename part
#                  if uploaded_file_obj: current_filename = uploaded_file_obj.name
#                  elif st.session_state.analysis_results: current_filename = st.session_state.analysis_results[0].get("File Name", "analysis")
#                  safe_base_name = re.sub(r'[^\w\s-]', '', current_filename.split('.')[0]).strip().replace(' ', '_')
#                  download_filename = f"Analysis_{safe_base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
#                  st.sidebar.download_button(label="📥 Download Results as Excel", data=st.session_state.excel_data, file_name=download_filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_excel_final", use_container_width=True)


#     # --- Column 2: Page Viewer (Sticky) ---
#     with col2:
#         # Wrap content in the styled div for sticky effect
#         st.markdown('<div class="sticky-viewer-content">', unsafe_allow_html=True)

#         st.subheader("📄 Page Viewer")
#         viewer_status_placeholder = st.empty() # Status for search/render

#         if st.session_state.pdf_bytes:
#             # --- Perform Search if Triggered ---
#             if st.session_state.search_trigger:
#                 search_info = st.session_state.search_trigger; st.session_state.search_trigger = None
#                 with st.spinner(f"🔎 Searching for: '{search_info['ref']}'..."):
#                     found_page, instances, term_used, search_status = find_text_in_pdf(st.session_state.pdf_bytes, search_info['text'])
#                 if found_page:
#                     st.session_state.current_page = found_page
#                     st.session_state.last_search_result = {'page': found_page, 'instances': instances, 'term': term_used, 'status': search_status, 'ref': search_info['ref']}
#                     viewer_status_placeholder.empty(); # Clear searching msg
#                 else:
#                     st.session_state.last_search_result = None
#                     viewer_status_placeholder.error(search_status) # Show failure msg

#             # --- Render Page ---
#             try:
#                 with fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf") as doc: total_pages = doc.page_count
#             except Exception as e: st.error(f"PDF load error: {e}"); total_pages = 1
#             # Ensure current page is valid
#             current_display_page = max(1, min(int(st.session_state.get('current_page', 1)), total_pages))
#             if current_display_page != st.session_state.get('current_page'): st.session_state.current_page = current_display_page

#             # --- Navigation ---
#             nav_cols = st.columns([1, 3, 1])
#             with nav_cols[0]:
#                 if st.button("⬅️ Prev", key="prev_page", disabled=(current_display_page <= 1)):
#                     st.session_state.current_page -= 1; st.session_state.last_search_result = None; st.rerun()
#             with nav_cols[1]:
#                 page_info_text = f"Page {current_display_page} of {total_pages}"
#                 if st.session_state.last_search_result and st.session_state.last_search_result['page'] == current_display_page:
#                      page_info_text += f" (🎯 Ref: '{st.session_state.last_search_result['ref']}')"
#                 st.markdown(f"<div style='text-align: center; padding-top: 0.5rem;'>{page_info_text}</div>", unsafe_allow_html=True)
#             with nav_cols[2]:
#                 if st.button("Next ➡️", key="next_page", disabled=(current_display_page >= total_pages)):
#                     st.session_state.current_page += 1; st.session_state.last_search_result = None; st.rerun()

#             # --- Determine Highlights & Render ---
#             st.markdown("---")
#             highlights_to_apply = None; render_status_override = None
#             if st.session_state.last_search_result and st.session_state.last_search_result['page'] == current_display_page:
#                 highlights_to_apply = st.session_state.last_search_result['instances']; render_status_override = st.session_state.last_search_result['status']

#             image_bytes, render_status = render_pdf_page_to_image(st.session_state.pdf_bytes, current_display_page, highlight_instances=highlights_to_apply)

#             # Display Image & Status
#             if image_bytes:
#                 st.image(image_bytes, caption=f"Page {current_display_page}", use_column_width='always')
#                 final_status = render_status_override if render_status_override else render_status
#                 if final_status and "Error" not in final_status and "error" not in final_status and "❌" not in final_status: # Don't overwrite permanent errors
#                     if "✨" in final_status or "Found" in final_status: viewer_status_placeholder.success(final_status)
#                     elif "⚠️" in final_status: viewer_status_placeholder.warning(final_status)
#                     else: viewer_status_placeholder.caption(final_status) # Plain render message
#                 elif not final_status and not viewer_status_placeholder._is_empty: # Clear if no status and not showing error
#                     viewer_status_placeholder.empty()
#             else: viewer_status_placeholder.error(f"Could not render page {current_display_page}. {render_status or ''}") # Show render error

#         else: st.info("Upload a PDF and run analysis to view pages."); viewer_status_placeholder.empty()

#         # Close the sticky div
#         st.markdown('</div>', unsafe_allow_html=True)


# # --- Fallback messages if analysis hasn't run or no PDF ---
# elif st.session_state.pdf_bytes is not None and not st.session_state.processing_complete:
#      st.info("PDF loaded. Click 'Analyse Document' in the sidebar.")
# elif st.session_state.pdf_bytes is None:
#      st.info("⬆️ Upload a PDF file using the sidebar to begin.")



# ----------------


# # app.py
# # --- COMPLETE FILE ---

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

# # --- 1. SET PAGE CONFIG (MUST BE FIRST st COMMAND) ---
# st.set_page_config(
#     layout="wide",
#     page_title="JASPER - Agreement Analyzer",
#     page_icon="📄" # Optional: Set an emoji icon
# )

# # --- Inject custom CSS for sticky column (AFTER set_page_config) ---
# st.markdown("""
# <style>
#     /* Define a class for the sticky container */
#     .sticky-viewer-content {
#         position: sticky;
#         top: 55px; /* Adjust vertical offset from top */
#         z-index: 101; /* Ensure it's above other elements */
#         padding-bottom: 1rem; /* Add some space at the bottom */
#     }
#     /* Ensure background color matches theme for sticky element */
#     div.sticky-viewer-content {
#         background-color: var(--streamlit-background-color);
#     }
# </style>
# """, unsafe_allow_html=True)

# # --- 2. Configuration & Setup ---
# API_KEY_ENV_VAR = "GEMINI_API_KEY"
# # Ensure the API key is set
# api_key = os.getenv(API_KEY_ENV_VAR)
# if not api_key:
#     st.error(f"API Key environment variable '{API_KEY_ENV_VAR}' not set.")
#     # Optionally provide input for API key if not found in env
#     # api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")
#     # if not api_key:
#     #     st.stop()
#     st.stop() # Stop if no key
# else:
#     try:
#         genai.configure(api_key=api_key)
#         # Optional: Test API key with a simple model list call
#         # models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
#         # if not models:
#         #     st.warning("API Key configured, but no suitable models found. Check key permissions.")
#     except Exception as config_err:
#         st.error(f"Failed to configure Gemini API: {config_err}")
#         st.stop()


# MODEL_NAME = "gemini-1.5-pro-preview-0514" # Check model availability and choose appropriate one
# MAX_VALIDATION_RETRIES = 1
# RETRY_DELAY_SECONDS = 3
# PROMPT_FILE = "prompt.txt"
# LOGO_FILE = "jasper-logo-1.png"
# SEARCH_FALLBACK_MIN_LENGTH = 20 # Minimum characters for fallback sentence search
# SEARCH_PREFIX_MIN_WORDS = 4 # Minimum words for 5/10 word prefix search to be attempted

# # --- Get the absolute path to the directory containing app.py ---
# APP_DIR = os.path.dirname(os.path.abspath(__file__))

# # --- Load Prompt Text from File ---
# try:
#     prompt_path = os.path.join(APP_DIR, PROMPT_FILE)
#     with open(prompt_path, 'r', encoding='utf-8') as f:
#         full_prompt_text = f.read()
# except FileNotFoundError:
#     st.error(f"Error: Prompt file '{PROMPT_FILE}' not found in the application directory ({APP_DIR}).")
#     st.stop()
# except Exception as e:
#     st.error(f"Error reading prompt file '{PROMPT_FILE}': {e}")
#     st.stop()

# # --- Logo (in Sidebar) ---
# try:
#     logo_path = os.path.join(APP_DIR, LOGO_FILE)
#     st.sidebar.image(logo_path, width=150)
#     st.sidebar.markdown("---")
# except FileNotFoundError:
#     st.sidebar.warning(f"Logo '{LOGO_FILE}' not found.")
# except Exception as e:
#     st.sidebar.error(f"Error loading logo: {e}")

# # --- Schema Definition (Version 3 - Search Text) ---
# ai_response_schema_dict = {
#   "type": "array",
#   "description": "A list of questions, answers, justifications, and supporting evidence derived from the facility agreement analysis.",
#   "items": {
#     "type": "object",
#     "description": "Represents a single question's analysis, including potentially multiple pieces of evidence.",
#     "properties": {
#       "Question Number": {"type": "integer"},
#       "Question Category": {"type": "string"},
#       "Question": {"type": "string"},
#       "Answer": {"type": "string"},
#       "Answer Justification": {
#           "type": "string",
#           "description": "Overall justification for the answer, considering *all* supporting evidence found."
#       },
#       "Evidence": {
#         "type": "array",
#         "description": "A list of evidence items supporting the answer. Each item links a clause to its wording and the specific text used for searching. This array should be empty if the Answer is 'Information Not Found' or 'N/A'.",
#         "items": {
#           "type": "object",
#           "properties": {
#             "Clause Reference": {
#                 "type": "string",
#                 "description": "Specific clause number(s) or section(s) (e.g., 'Clause 14.1(a)', 'Section 5', 'Definition of Confidential Information')."
#             },
#             "Clause Wording": {
#                 "type": "string",
#                 "description": "Exact, full text of the referenced clause(s) or relevant sentence(s), including leading reference/heading."
#             },
#             "Searchable Clause Text": {
#                 "type": "string",
#                 "description": "The exact, searchable text content of the clause, excluding the clause reference itself."
#             }
#           },
#           "required": ["Clause Reference", "Clause Wording", "Searchable Clause Text"]
#         }
#       }
#     },
#     "required": [
#       "Question Number", "Question Category", "Question", "Answer", "Answer Justification", "Evidence"
#     ]
#   }
# }

# # Keys required for validation
# AI_REQUIRED_KEYS = set(ai_response_schema_dict['items']['required'])
# AI_EVIDENCE_REQUIRED_KEYS = set(ai_response_schema_dict['items']['properties']['Evidence']['items']['required'])

# # --- Excel Column Order ---
# EXCEL_COLUMN_ORDER = [
#     "File Name", "Generation Time", "Question Number", "Question Category",
#     "Question", "Answer", "Answer Justification",
#     "Clause References (Concatenated)",
#     "First Searchable Clause Text" # Changed from Page Numbers
# ]

# # --- Section Definitions ---
# SECTIONS_TO_RUN = {
#     "agreement_details": (1, 4),
#     "eligibility": (5, 36),
#     "confidentiality": (37, 63),
#     "additional_borrowers": (64, 66),
#     "interest_rate_provisions": (67, 71),
#     "prepayment_fee": (72, 78) # Corrected key
# }

# # --- System Instruction ---
# system_instruction_text = """You are analysing a facility agreement to understand whether the asset can be included within a Significant Risk Transfer or not (or with conditions, requirements, or exceptions) at NatWest. Your output must be precise, factual, and directly supported by evidence from the provided document(s). You must answer with UK spelling, not US. (e.g. 'analyse' is correct while 'analyse' is not). Adhere strictly to the JSON schema provided, ensuring every object in the output array contains all required keys."""


# # --- 3. Helper Function Definitions ---

# def filter_prompt_by_section(initial_full_prompt, section):
#     """Filters the main prompt to include only questions for a specific section."""
#     if section not in SECTIONS_TO_RUN:
#         raise ValueError(f"Invalid section specified: {section}. Must be one of {list(SECTIONS_TO_RUN.keys())}")
#     start_q, end_q = SECTIONS_TO_RUN[section]
#     questions_start_marker = "**Questions to Answer:**"
#     questions_end_marker = "**Final Instruction:**"
#     try:
#         start_index = initial_full_prompt.index(questions_start_marker)
#         end_index = initial_full_prompt.index(questions_end_marker)
#         prompt_header = initial_full_prompt[:start_index]
#         full_questions_block = initial_full_prompt[start_index + len(questions_start_marker):end_index].strip()
#         prompt_footer = initial_full_prompt[end_index:]
#     except ValueError:
#         st.error("Could not find question block markers ('**Questions to Answer:**' or '**Final Instruction:**') in the main prompt text. Check prompt.txt.")
#         raise ValueError("Could not find prompt markers.")

#     # Split questions based on the numbered format
#     question_entries = re.split(r'\n(?=\s*\d+\.\s*?\*\*Question Category:)', full_questions_block)
#     if not question_entries or len(question_entries) < 2: # Fallback splits
#         question_entries = re.split(r'\n(?=\d+\.\s)', full_questions_block)
#         if not question_entries or len(question_entries) < 2:
#              question_entries = re.split(r'\n(?=\*\*Question Category:)', full_questions_block)

#     filtered_question_texts = []
#     processed_q_nums = set()
#     for entry in question_entries:
#         entry = entry.strip()
#         if not entry: continue
#         match = re.match(r'^\s*(\d+)\.', entry)
#         if match:
#             q_num = int(match.group(1))
#             if start_q <= q_num <= end_q:
#                 filtered_question_texts.append(entry)
#                 processed_q_nums.add(q_num)

#     expected_q_nums = set(range(start_q, end_q + 1))
#     missing_q_nums = expected_q_nums - processed_q_nums
#     if missing_q_nums:
#          try: st.warning(f"Parsing might have missed expected question numbers in range {start_q}-{end_q} for section '{section}': {sorted(list(missing_q_nums))}")
#          except Exception: pass # Avoid errors if st called too early

#     if not filtered_question_texts:
#         st.error(f"No questions found for section '{section}' in range {start_q}-{end_q}. Check prompt formatting and split logic.")
#         raise ValueError(f"Failed to extract questions for section '{section}'.")

#     filtered_questions_string = "\n\n".join(filtered_question_texts)
#     task_end_marker = "specified section." # Add focus note to prompt
#     insert_pos = prompt_header.find(task_end_marker)
#     section_note = f"\n\n**Current Focus:** Answering ONLY questions for the '{section.upper()}' section (Questions {start_q}-{end_q}). The list below contains ONLY these questions.\n"
#     if insert_pos != -1:
#          insert_pos += len(task_end_marker)
#          final_header = prompt_header[:insert_pos] + section_note + prompt_header[insert_pos:]
#     else: final_header = prompt_header + section_note
#     final_prompt_for_api = f"{final_header}{questions_start_marker}\n\n{filtered_questions_string}\n\n{prompt_footer}"
#     return final_prompt_for_api


# def validate_ai_data(data, section_name):
#     """Validates the structure of the AI response based on the new schema. Returns validated data (list) and list of issue strings."""
#     if not isinstance(data, list):
#         st.error(f"Crit Val Error: AI response for '{section_name}' not list.")
#         return None, [f"CRITICAL: Response for '{section_name}' was not a list."]

#     validated_data = []
#     issues_list = []
#     for index, item in enumerate(data): # Loop outer items
#         q_num_str = f"Q#{item.get('Question Number', f'Item {index}')}"
#         is_outer_valid = True
#         if not isinstance(item, dict):
#             issues_list.append(f"{q_num_str}: Not a dictionary."); continue

#         missing_outer_keys = AI_REQUIRED_KEYS - set(item.keys())
#         if missing_outer_keys: issues_list.append(f"{q_num_str}: Missing: {missing_outer_keys}"); is_outer_valid = False
#         evidence_list = item.get("Evidence")
#         if not isinstance(evidence_list, list): issues_list.append(f"{q_num_str}: 'Evidence' not list."); is_outer_valid = False
#         else: # Validate inner evidence items
#             for ev_index, ev_item in enumerate(evidence_list):
#                 if not isinstance(ev_item, dict): issues_list.append(f"{q_num_str} Ev[{ev_index}]: Not dict."); is_outer_valid = False; continue
#                 missing_ev_keys = AI_EVIDENCE_REQUIRED_KEYS - set(ev_item.keys())
#                 if missing_ev_keys: issues_list.append(f"{q_num_str} Ev[{ev_index}]: Missing: {missing_ev_keys}"); is_outer_valid = False
#                 # Check types for new schema
#                 if not isinstance(ev_item.get("Clause Reference"), str): issues_list.append(f"{q_num_str} Ev[{ev_index}]: Ref not str."); is_outer_valid = False
#                 if not isinstance(ev_item.get("Clause Wording"), str): issues_list.append(f"{q_num_str} Ev[{ev_index}]: Wording not str."); is_outer_valid = False
#                 if not isinstance(ev_item.get("Searchable Clause Text"), str): issues_list.append(f"{q_num_str} Ev[{ev_index}]: SearchText not str."); is_outer_valid = False # Check new field

#         if is_outer_valid: validated_data.append(item)

#     if issues_list: # Prepend summary header & show warning
#         issues_list.insert(0, f"Validation Issues [Section: {section_name}] ({len(validated_data)} passed):")
#         st.warning("Validation Issues Detected:\n" + "\n".join([f"- {issue}" for issue in issues_list[1:]]))
#     return validated_data, issues_list


# def generate_section_analysis(section, uploaded_file_ref, status_placeholder):
#     """Generates analysis, handles retries, validation. Returns (data, status, warnings)."""
#     status_placeholder.info(f"🔄 Starting: {section}...")
#     section_warnings = []
#     model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction_text)
#     generation_config = types.GenerationConfig(
#         response_mime_type="application/json", response_schema=ai_response_schema_dict,
#         temperature=0.0, top_p=0.05, top_k=1
#     )
#     safety_settings = [ # Adjust safety settings if needed
#         {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
#         {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
#         {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
#         {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
#     ]
#     final_validated_data = None

#     for attempt in range(1, 1 + MAX_VALIDATION_RETRIES + 1):
#         if attempt > 1: status_placeholder.info(f"⏳ Retrying: '{section}' (Attempt {attempt})..."); time.sleep(RETRY_DELAY_SECONDS)
#         try:
#             prompt_for_api = filter_prompt_by_section(full_prompt_text, section)
#             contents = [uploaded_file_ref, prompt_for_api]
#             status_placeholder.info(f"🧠 Calling AI: {section} (Attempt {attempt})...");
#             response = model.generate_content(
#                 contents=contents,
#                 generation_config=generation_config,
#                 safety_settings=safety_settings # Apply safety settings
#             )
#             parsed_ai_data = None; validated_ai_data = None; validation_issues = []
#             status_placeholder.info(f"🔍 Processing: {section}...");
#             if response.parts:
#                 full_response_text = response.text
#                 try:
#                     # Gemini sometimes wraps the JSON in ```json ... ```, try to strip it
#                     match = re.search(r"```json\s*([\s\S]*?)\s*```", full_response_text)
#                     if match:
#                          json_text = match.group(1).strip()
#                     else:
#                          json_text = full_response_text.strip()

#                     parsed_ai_data = json.loads(json_text)
#                     status_placeholder.info(f"✔️ Validating: {section}..."); validated_ai_data, validation_issues = validate_ai_data(parsed_ai_data, section); section_warnings.extend(validation_issues)
#                     if validated_ai_data is not None and len(validated_ai_data) > 0: final_validated_data = validated_ai_data; break
#                     elif validated_ai_data is None: error_msg = f"Crit Val Error: '{section}'."; section_warnings.append(error_msg); break # Error shown by validate_ai_data
#                     else: status_placeholder.warning(f"⚠️ Validation 0 items: {section} (Attempt {attempt}).") # Retry if possible
#                 except json.JSONDecodeError as json_err: error_msg = f"JSON Error {attempt}: {section}: {json_err}"; st.error(error_msg); st.code(full_response_text); section_warnings.append(error_msg) # Show raw response on JSON error
#             else: # Handle empty response or blocked content
#                 block_reason = "Unknown"; block_message = "N/A"; finish_reason = "Unknown"
#                 try:
#                    if response.prompt_feedback:
#                      if response.prompt_feedback.block_reason: block_reason = response.prompt_feedback.block_reason.name
#                      block_message = response.prompt_feedback.block_reason_message or "N/A"
#                    # Also check candidate finish reason
#                    if response.candidates and response.candidates[0].finish_reason:
#                        finish_reason_enum = response.candidates[0].finish_reason
#                        finish_reason = finish_reason_enum.name # Get the name string
#                 except Exception as feedback_err:
#                    block_message = f"Error accessing feedback: {feedback_err}"

#                 # Refine warning message based on finish reason
#                 if finish_reason == "SAFETY":
#                      warn_msg = f"Blocked Response {attempt}: {section}. Reason: SAFETY. Block Detail: {block_reason}. Msg: {block_message}"
#                      st.error(warn_msg) # Elevate safety blocks to error
#                 elif finish_reason == "RECITATION":
#                      warn_msg = f"Blocked Response {attempt}: {section}. Reason: RECITATION."
#                      st.warning(warn_msg)
#                 elif not response.parts:
#                      warn_msg = f"Empty Response {attempt}: {section}. Finish Reason: {finish_reason}. Block Reason: {block_reason}."
#                      st.warning(warn_msg)
#                 else: # Should not happen if response.parts is False, but as a fallback
#                      warn_msg = f"Unknown Issue {attempt}: {section}. Finish: {finish_reason}, Block: {block_reason}, Msg: {block_message}"
#                      st.warning(warn_msg)

#                 section_warnings.append(warn_msg)

#         except types.generation_types.BlockedPromptException as bpe: error_msg = f"API Block Error {attempt}: {section}: {bpe}"; st.error(error_msg); section_warnings.append(error_msg)
#         except types.generation_types.StopCandidateException as sce: error_msg = f"API Stop Error {attempt}: {section}: {sce}"; st.error(error_msg); section_warnings.append(error_msg)
#         except ValueError as ve: error_msg = f"Prompt Filter Error {attempt}: {section}: {ve}"; st.error(error_msg); section_warnings.append(error_msg); break # Stop section on prompt error
#         except Exception as e: error_msg = f"Processing Error {attempt}: {section}: {type(e).__name__}"; st.error(error_msg); section_warnings.append(error_msg); section_warnings.append(f"Traceback in logs."); print(traceback.format_exc()); # Print traceback for debug
#         if final_validated_data is not None: break # Exit loop if successful

#     if final_validated_data is not None: status_placeholder.info(f"✅ Completed: {section}."); st.toast(f"Processed: {section}", icon="✅"); return final_validated_data, "Success", section_warnings
#     else: status_placeholder.error(f"❌ Failed: {section} after {attempt} attempts."); return None, "Failed", section_warnings


# # --- START OF MODIFIED find_text_in_pdf ---
# @st.cache_data(show_spinner=False)
# def find_text_in_pdf(_pdf_bytes, search_text):
#     """
#     Searches the entire PDF for text using a prioritized strategy:
#     1. First 10 words (if >= SEARCH_PREFIX_MIN_WORDS words)
#     2. First 5 words (if >= SEARCH_PREFIX_MIN_WORDS words)
#     3. Full text provided by AI
#     4. First sentence (if >= SEARCH_FALLBACK_MIN_LENGTH chars)

#     Returns:
#         tuple: (found_page_number, highlight_instances, search_term_used, status_message)
#                - found_page_number (int): 1-based page number where text was found, or None.
#                - highlight_instances (list): List of fitz.Rect/Quad instances for highlighting, or None.
#                - search_term_used (str): The actual text string used for the successful search, or None.
#                - status_message (str): Message indicating search outcome (found, method, not found, error).
#     """
#     if not _pdf_bytes or not search_text:
#         return None, None, None, "Invalid input (missing PDF bytes or search text)."

#     doc = None
#     search_text_cleaned = search_text.strip()
#     words = search_text_cleaned.split()
#     num_words = len(words)

#     search_attempts = []

#     # --- Attempt 1: First 10 words ---
#     if num_words >= SEARCH_PREFIX_MIN_WORDS:
#         term_10 = ' '.join(words[:10])
#         if term_10:
#             search_attempts.append({'term': term_10, 'desc': "first 10 words"})

#     # --- Attempt 2: First 5 words ---
#     if num_words >= SEARCH_PREFIX_MIN_WORDS:
#         term_5 = ' '.join(words[:5])
#         # Only add if different from 10-word term and non-empty
#         if term_5 and term_5 != search_attempts[0]['term'] if search_attempts else True:
#              search_attempts.append({'term': term_5, 'desc': "first 5 words"})

#     # --- Attempt 3: Full Text ---
#     term_full = search_text_cleaned
#     # Only add if different from shorter prefixes already added
#     is_different = True
#     for attempt in search_attempts:
#         if term_full == attempt['term']:
#             is_different = False
#             break
#     if term_full and is_different:
#         search_attempts.append({'term': term_full, 'desc': "full text"})

#     # --- Attempt 4: First Sentence ---
#     term_sentence = ""
#     # Basic sentence split - adjust regex if needed for more complex cases
#     sentences = re.split(r'(?<=[.?!])\s+', term_full) # Split based on sentence-ending punctuation
#     if sentences and len(sentences[0].strip()) >= SEARCH_FALLBACK_MIN_LENGTH:
#         term_sentence = sentences[0].strip()
#         # Only add if different from all previous attempts
#         is_different_sentence = True
#         for attempt in search_attempts:
#              if term_sentence == attempt['term']:
#                  is_different_sentence = False
#                  break
#         if term_sentence and is_different_sentence:
#             search_attempts.append({'term': term_sentence, 'desc': "first sentence"})

#     # --- Execute Search Attempts ---
#     try:
#         doc = fitz.open(stream=_pdf_bytes, filetype="pdf")
#         search_flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES

#         for attempt in search_attempts:
#             term = attempt['term']
#             desc = attempt['desc']
#             # print(f"Attempting search with {desc}: '{term[:80]}...'") # Debug print

#             for page_index in range(doc.page_count):
#                 page = doc.load_page(page_index)
#                 instances = page.search_for(term, flags=search_flags, quads=False)
#                 if instances:
#                     doc.close()
#                     status = f"✨ Found using {desc} on page {page_index + 1}."
#                     return page_index + 1, instances, term, status # Success!

#         # --- Text Not Found after all attempts ---
#         doc.close()
#         return None, None, None, f"❌ Text not found (tried {len(search_attempts)} methods)."

#     except Exception as e:
#         if doc: doc.close()
#         print(f"ERROR searching PDF: {e}") # Log error to console
#         print(traceback.format_exc())
#         return None, None, None, f"❌ Error during PDF search: {e}"
# # --- END OF MODIFIED find_text_in_pdf ---


# # @st.cache_data(show_spinner=False) # Caching render might interfere with dynamic highlights
# def render_pdf_page_to_image(_pdf_bytes, page_number, highlight_instances=None, dpi=150):
#     """
#     Renders a specific PDF page to an image, applying highlights if instances are provided.

#     Args:
#         _pdf_bytes: Bytes of the PDF file.
#         page_number: The 1-based page number to render.
#         highlight_instances: List of fitz.Rect/Quad objects to highlight on the page.
#         dpi: Resolution for rendering.

#     Returns:
#         tuple: (image_bytes or None, status_message or None)
#                Status message indicates success/failure of rendering/highlighting.
#     """
#     if not _pdf_bytes or page_number < 1:
#         return None, "Invalid input (no PDF bytes or page number < 1)."

#     doc = None
#     image_bytes = None
#     render_status_message = f"Rendered page {page_number}." # Default status

#     try:
#         doc = fitz.open(stream=_pdf_bytes, filetype="pdf")
#         page_index = page_number - 1
#         if page_index < 0 or page_index >= doc.page_count:
#             if doc: doc.close()
#             return None, f"Page number {page_number} is out of range (1-{doc.page_count if doc else 'unknown'})."

#         page = doc.load_page(page_index)
#         highlight_applied = False
#         highlight_count = 0

#         # --- Add Annotations if instances provided ---
#         if highlight_instances:
#             try:
#                 for inst in highlight_instances:
#                     if isinstance(inst, (fitz.Rect, fitz.Quad)): # Check type
#                         highlight = page.add_highlight_annot(inst)
#                         # Optional: Style the highlight
#                         highlight.set_colors(stroke=fitz.utils.getColor("red"), fill=fitz.utils.getColor("yellow")) # Yellow highlight with red border
#                         highlight.set_opacity(0.4) # Semi-transparent
#                         highlight.update()
#                         highlight_count += 1
#                     else:
#                          print(f"WARN: Skipping invalid highlight instance type on page {page_number}: {type(inst)}") # Log warning

#                 if highlight_count > 0:
#                     highlight_applied = True
#                     # Status message from search function will override this if search was done
#                     render_status_message = f"✨ Rendered page {page_number} with {highlight_count} highlight(s)."
#                 elif highlight_instances: # Instances provided but none were valid Rect/Quad
#                     render_status_message = f"⚠️ Rendered page {page_number}, but no valid highlights found in provided instances."

#             except Exception as highlight_err:
#                 print(f"ERROR applying highlights on page {page_number}: {highlight_err}") # Log to console
#                 render_status_message = f"⚠️ Error applying highlights on page {page_number}: {highlight_err}"

#         # --- Render page ---
#         pix = page.get_pixmap(dpi=dpi, alpha=False) # alpha=False for potential performance gain if transparency not needed
#         image_bytes = pix.tobytes("png")

#     except Exception as e:
#         print(f"ERROR rendering PDF page {page_number}: {e}") # Log error to console
#         # print(traceback.format_exc()) # Optional full traceback
#         render_status_message = f"❌ Error rendering page {page_number}: {e}"
#         image_bytes = None # Ensure no image returned on error
#     finally:
#         if doc: doc.close()

#     return image_bytes, render_status_message


# # --- 4. Initialize Session State ---
# if 'current_page' not in st.session_state: st.session_state.current_page = 1
# if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
# if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None
# if 'pdf_bytes_processed' not in st.session_state: st.session_state.pdf_bytes_processed = None
# if 'processing_complete' not in st.session_state: st.session_state.processing_complete = False
# if 'run_key' not in st.session_state: st.session_state.run_key = 0
# if 'run_status_summary' not in st.session_state: st.session_state.run_status_summary = []
# if 'excel_data' not in st.session_state: st.session_state.excel_data = None
# # New state for search mechanism
# if 'search_trigger' not in st.session_state: st.session_state.search_trigger = None # Stores {'text': search_text, 'ref': clause_ref}
# if 'last_search_result' not in st.session_state: st.session_state.last_search_result = None # Stores {'page': num, 'instances': [], 'term': str, 'status': str}


# # --- 5. Streamlit UI Logic ---

# st.title("JASPER - Facility Agreement Analyzer")
# st.markdown("Upload a PDF agreement, click 'Analyse Document'. Results are grouped below. Click **Clause References** in the 'Evidence' section to find and view the relevant page with highlights.")
# st.sidebar.markdown("## Controls")

# # --- File Upload (in Sidebar) ---
# uploaded_file_obj = st.sidebar.file_uploader(
#     "Upload Facility Agreement PDF", type="pdf", key=f"pdf_uploader_{st.session_state.run_key}"
# )

# # Process new file upload only if bytes differ from last processed
# if uploaded_file_obj is not None:
#     uploaded_bytes = uploaded_file_obj.getvalue()
#     if uploaded_bytes != st.session_state.get('pdf_bytes_processed'):
#         st.session_state.pdf_bytes = uploaded_bytes
#         st.session_state.pdf_bytes_processed = uploaded_bytes # Mark as processed for this session run
#         # Reset dependent states
#         st.session_state.analysis_results = None; st.session_state.processing_complete = False
#         st.session_state.current_page = 1; st.session_state.run_status_summary = []
#         st.session_state.excel_data = None; st.session_state.search_trigger = None; st.session_state.last_search_result = None
#         st.toast("✅ New PDF file loaded.", icon="📄")
#         st.rerun() # Rerun to reflect cleared state
# elif 'pdf_bytes_processed' in st.session_state: # If file removed, clear flag
#      st.session_state.pdf_bytes_processed = None


# # --- Analysis Trigger (in Sidebar) ---
# analyse_disabled = st.session_state.processing_complete or st.session_state.pdf_bytes is None
# if st.sidebar.button("✨ Analyse Document", key="analyse_button", disabled=analyse_disabled, use_container_width=True, type="primary"):
#     # Reset states for the new run
#     st.session_state.analysis_results = None; st.session_state.processing_complete = False
#     st.session_state.current_page = 1; st.session_state.run_key += 1; st.session_state.run_status_summary = []
#     st.session_state.excel_data = None; st.session_state.search_trigger = None; st.session_state.last_search_result = None

#     run_start_time = datetime.now(); run_timestamp_str = run_start_time.strftime("%Y-%m-%d %H:%M:%S")
#     base_file_name = getattr(uploaded_file_obj, 'name', 'uploaded_file')

#     # Status placeholders in main area
#     status_container = st.container()
#     progress_bar = status_container.progress(0, text="Initializing...")
#     status_text = status_container.empty()

#     temp_dir = "temp_uploads"; temp_file_path = os.path.join(APP_DIR, temp_dir, f"{run_start_time.strftime('%Y%m%d%H%M%S')}_{base_file_name}")
#     os.makedirs(os.path.dirname(temp_file_path), exist_ok=True) # Ensure directory exists

#     gemini_uploaded_file_ref = None; all_validated_data = []; overall_success = True

#     try:
#         status_text.info("💾 Saving file temporarily..."); progress_bar.progress(5, text="Saving...")
#         with open(temp_file_path, "wb") as f: f.write(st.session_state.pdf_bytes)

#         status_text.info("🚀 Uploading to Google Cloud..."); progress_bar.progress(10, text="Uploading...")
#         # Add retry logic for file upload if needed
#         for upload_attempt in range(3):
#             try:
#                 gemini_uploaded_file_ref = genai.upload_file(path=temp_file_path)
#                 st.toast(f"File '{gemini_uploaded_file_ref.display_name}' uploaded.", icon="☁️")
#                 break # Success
#             except Exception as upload_err:
#                 if upload_attempt < 2:
#                     st.warning(f"File upload failed (Attempt {upload_attempt+1}): {upload_err}. Retrying...")
#                     time.sleep(2)
#                 else:
#                     st.error(f"File upload failed after multiple attempts: {upload_err}")
#                     raise # Re-raise the last error
#         if not gemini_uploaded_file_ref:
#              raise Exception("Failed to upload file to Gemini.")

#         progress_bar.progress(15, text="Uploaded.")

#         num_sections = len(SECTIONS_TO_RUN); progress_per_section = (95 - 15) / num_sections if num_sections > 0 else 0

#         # --- Process Sections ---
#         for i, section_name in enumerate(SECTIONS_TO_RUN.keys()):
#             current_progress = int(15 + (i * progress_per_section))
#             progress_bar.progress(current_progress, text=f"Starting {section_name}...")
#             section_data, section_status, section_warnings = generate_section_analysis(section_name, gemini_uploaded_file_ref, status_text)
#             st.session_state.run_status_summary.append({"section": section_name, "status": section_status, "warnings": section_warnings})
#             if section_status == "Success" and section_data:
#                 for item in section_data: item["File Name"] = base_file_name; item["Generation Time"] = run_timestamp_str
#                 all_validated_data.extend(section_data)
#             else: overall_success = False

#         st.session_state.analysis_results = all_validated_data
#         progress_bar.progress(100, text="Analysis Complete!")
#         if overall_success: status_text.success("🏁 Analysis finished successfully!")
#         else: status_text.warning("🏁 Analysis finished with some issues (see summary below).")
#         st.session_state.processing_complete = True

#     except Exception as main_err:
#          st.error(f"❌ CRITICAL ERROR during analysis: {main_err}"); st.error(traceback.format_exc());
#          overall_success = False; st.session_state.processing_complete = False
#          st.session_state.run_status_summary.append({"section": "Overall Process", "status": "Critical Error", "warnings": [str(main_err), "Check server logs for traceback."]})
#          status_text.error("Analysis failed due to a critical error.")
#     finally: # Cleanup
#         # Delay before clearing status, keeps final message visible longer
#         time.sleep(4)
#         status_text.empty()
#         progress_bar.empty()
#         # Cleanup temp files and Gemini file
#         if gemini_uploaded_file_ref and hasattr(gemini_uploaded_file_ref, 'name'):
#              try:
#                  # Use a separate placeholder for cleanup messages
#                  cleanup_status = st.sidebar.empty() # Place it in sidebar maybe
#                  cleanup_status.info(f"🧹 Deleting temporary cloud file: {gemini_uploaded_file_ref.display_name}...")
#                  genai.delete_file(name=gemini_uploaded_file_ref.name)
#                  cleanup_status.info("🧹 Cloud file deleted.")
#                  time.sleep(2) # Short pause
#                  cleanup_status.empty()
#              except Exception as delete_err:
#                  st.warning(f"Could not delete cloud file '{gemini_uploaded_file_ref.name}': {delete_err}")
#         if os.path.exists(temp_file_path):
#              try: os.remove(temp_file_path)
#              except Exception: pass

#     st.rerun() # Rerun to update display after analysis


# # --- Run Status Summary Expander (Displayed near the top) ---
# if st.session_state.run_status_summary:
#     final_status = "✅ Success"
#     if any(s['status'] != "Success" for s in st.session_state.run_status_summary): final_status = "⚠️ Completed with Issues"
#     if any("Critical" in s['status'] or "Fail" in s['status'] for s in st.session_state.run_status_summary): final_status = "❌ Failed"
#     # Expand automatically if not pure success
#     with st.expander(f"📊 Last Analysis Run Summary ({final_status})", expanded=(final_status != "✅ Success")):
#         for item in st.session_state.run_status_summary:
#             icon = "✅" if item['status'] == "Success" else "❌" if "Fail" in item['status'] or "Error" in item['status'] else "⚠️"
#             st.markdown(f"**{item['section']}**: {icon} {item['status']}")
#             if item['warnings']:
#                  # Display warnings associated with the section status
#                  with st.container():
#                      # Filter out the basic validation summary header if present
#                      filtered_warnings = [msg for msg in item['warnings'] if not (isinstance(msg, str) and msg.startswith("Validation Issues [Section:"))]
#                      if filtered_warnings:
#                          st.caption("Details:")
#                          for msg in filtered_warnings:
#                             # Improved formatting for common errors/warnings
#                             msg_str = str(msg).strip() # Ensure string
#                             if msg_str.startswith("CRITICAL:"): st.error(f" L> {msg_str}")
#                             elif msg_str.startswith(("JSON Error", "API Block", "API Stop", "Blocked Response", "Empty Response")):
#                                 st.warning(f" L> {msg_str}") # Use warning for most API/parsing issues
#                             elif "Missing:" in msg_str: st.warning(f" L> {msg_str}") # Validation field missing
#                             elif isinstance(msg, list): st.warning(" L> "+"\n L> ".join(map(str, msg))) # Handle list case
#                             else: st.caption(f" L> {msg_str}") # General warnings/info


# # --- Display Area (Results and PDF Viewer) ---
# if st.session_state.analysis_results is not None:
#     col1, col2 = st.columns([6, 4]) # Results | PDF Viewer

#     # --- Column 1: Analysis Results (Grouped by Category using Tabs) ---
#     with col1:
#         st.subheader("Analysis Results")
#         results_list = st.session_state.analysis_results

#         if not results_list:
#             st.info("Analysis complete, but no valid results were generated. Check the summary above.")
#         else:
#             # Group results by category
#             grouped_results = defaultdict(list)
#             categories_ordered = []
#             for item in results_list:
#                 category = item.get("Question Category", "Uncategorized")
#                 if category not in grouped_results: categories_ordered.append(category)
#                 grouped_results[category].append(item)

#             if categories_ordered:
#                 category_tabs = st.tabs(categories_ordered)
#                 for i, category in enumerate(categories_ordered):
#                     with category_tabs[i]:
#                         category_items = grouped_results[category]
#                         # Sort items by Question Number within the category
#                         category_items.sort(key=lambda x: x.get('Question Number', float('inf')))

#                         for index, result_item in enumerate(category_items):
#                             q_num = result_item.get('Question Number', 'N/A'); question_text = result_item.get('Question', 'N/A')

#                             with st.expander(f"**Q{q_num}:** {question_text}"):
#                                 st.markdown(f"**Answer:** {result_item.get('Answer', 'N/A')}")
#                                 st.markdown("---")

#                                 # Evidence Section
#                                 evidence_list = result_item.get('Evidence', [])
#                                 if evidence_list:
#                                     st.markdown("**Evidence:** (Click reference to find & view page with highlights)")
#                                     for ev_index, evidence_item in enumerate(evidence_list):
#                                         clause_ref = evidence_item.get('Clause Reference', 'N/A')
#                                         clause_wording = evidence_item.get('Clause Wording', None) # Keep for context maybe
#                                         search_text = evidence_item.get('Searchable Clause Text', None) # << Key field

#                                         # Button to trigger search (only if search_text exists)
#                                         if search_text:
#                                             button_key = f"search_btn_{category}_{q_num}_{index}_{ev_index}"
#                                             button_label = f"Clause: **{clause_ref or 'Link'}** (Find & View)"
#                                             if st.button(button_label, key=button_key, help=f"Search for text related to '{clause_ref or 'this evidence'}' and view the page."):
#                                                 st.session_state.search_trigger = {'text': search_text, 'ref': clause_ref}
#                                                 st.session_state.last_search_result = None # Clear previous search result immediately
#                                                 st.rerun() # Rerun to trigger search in viewer column
#                                         elif clause_ref != 'N/A':
#                                             # Show reference without button if no searchable text provided
#                                             st.markdown(f"- Clause: **{clause_ref}** (No searchable text provided)")

#                                         # Optionally display Clause Wording or Searchable Text for context (can be long)
#                                         # Use popovers for less clutter
#                                         cols = st.columns([1,1])
#                                         with cols[0]:
#                                              with st.popover("Show Clause Wording"):
#                                                 st.caption(f"Reference: {clause_ref}")
#                                                 st.text(clause_wording or "N/A")
#                                         with cols[1]:
#                                              with st.popover("Show Searchable Text"):
#                                                  st.caption(f"Reference: {clause_ref}")
#                                                  st.text(search_text or "N/A")
#                                         st.divider() # Add small divider between evidence items

#                                 else: st.markdown("**Evidence:** None provided.")
#                                 st.markdown("---") # Separator after evidence block

#                                 # Justification Section
#                                 st.markdown("**Answer Justification:**")
#                                 just_key = f"justification_{category}_{q_num}_{index}" # Unique key
#                                 st.text_area(label="Justification", value=result_item.get('Answer Justification', ''), height=100, disabled=True, label_visibility="collapsed", key=just_key)

#             else: st.warning("Results generated, but could not group by category.")


#             # --- Excel Download Preparation (in Sidebar) ---
#             st.sidebar.markdown("---")
#             st.sidebar.markdown("## Export")
#             if st.sidebar.button("Prepare Data for Excel Download", key="prep_excel", use_container_width=True):
#                 st.session_state.excel_data = None; excel_prep_status = st.sidebar.empty()
#                 excel_prep_status.info("Preparing Excel...")
#                 try:
#                     excel_rows = []
#                     for item in results_list: # Use full results list
#                         references = []; first_search_text = "N/A"
#                         evidence = item.get("Evidence")
#                         if evidence:
#                             for i, ev in enumerate(evidence):
#                                 references.append(str(ev.get("Clause Reference", "N/A")))
#                                 if i == 0: first_search_text = ev.get("Searchable Clause Text", "N/A") # Get first searchable text
#                         excel_row = {
#                             "File Name": item.get("File Name", ""),
#                             "Generation Time": item.get("Generation Time", ""),
#                             "Question Number": item.get("Question Number"),
#                             "Question Category": item.get("Question Category"),
#                             "Question": item.get("Question"),
#                             "Answer": item.get("Answer"),
#                             "Answer Justification": item.get("Answer Justification"),
#                             "Clause References (Concatenated)": "; ".join(references) if references else "N/A",
#                             "First Searchable Clause Text": first_search_text
#                         }
#                         excel_rows.append(excel_row)

#                     if not excel_rows:
#                          excel_prep_status.warning("No data to export.")
#                          st.session_state.excel_data = None
#                     else:
#                         df_excel = pd.DataFrame(excel_rows)
#                         # Apply column order, handle missing columns gracefully
#                         final_columns = []
#                         for col in EXCEL_COLUMN_ORDER:
#                             if col in df_excel.columns:
#                                 final_columns.append(col)
#                             else:
#                                 st.warning(f"Expected Excel column '{col}' not found in data, skipping.")

#                         df_excel = df_excel[final_columns] # Reorder with available columns

#                         # Write to BytesIO
#                         output = io.BytesIO()
#                         with pd.ExcelWriter(output, engine='openpyxl') as writer:
#                             df_excel.to_excel(writer, index=False, sheet_name='Analysis')
#                         st.session_state.excel_data = output.getvalue()
#                         excel_prep_status.success("Excel ready for download!")
#                         time.sleep(2); excel_prep_status.empty() # Clear message

#                 except Exception as excel_err:
#                      excel_prep_status.error(f"Excel Prep Error: {excel_err}")
#                      print(traceback.format_exc()) # Log traceback for debug

#             # --- Actual Download Button (in Sidebar) ---
#             if st.session_state.excel_data:
#                  current_filename = "analysis"
#                  if uploaded_file_obj: # Use current state if available
#                      current_filename = uploaded_file_obj.name
#                  elif 'pdf_bytes_processed' in st.session_state and isinstance(st.session_state.get('analysis_results'), list) and st.session_state.analysis_results:
#                      current_filename = st.session_state.analysis_results[0].get("File Name", "analysis")

#                  safe_base_name = re.sub(r'[^\w\s-]', '', current_filename.split('.')[0]).strip().replace(' ', '_')
#                  download_filename = f"Analysis_{safe_base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
#                  st.sidebar.download_button(
#                      label="📥 Download Results as Excel", data=st.session_state.excel_data,
#                      file_name=download_filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#                      key="download_excel_final", use_container_width=True
#                  )

# # --- Column 2: Page Viewer ---
#     with col2:
#         # Wrap content in the styled div for sticky effect
#         st.markdown('<div class="sticky-viewer-content">', unsafe_allow_html=True)

#         st.subheader("📄 Page Viewer")
#         viewer_status_placeholder = st.empty() # For showing search/render status

#         if st.session_state.pdf_bytes:
#             # --- Perform Search if Triggered ---
#             if st.session_state.search_trigger:
#                 search_info = st.session_state.search_trigger
#                 st.session_state.search_trigger = None # Consume the trigger
#                 with st.spinner(f"🔎 Searching for: '{search_info['ref']}'..."):
#                     found_page, instances, term_used, search_status = find_text_in_pdf(
#                         st.session_state.pdf_bytes, search_info['text']
#                     )
#                 if found_page:
#                     st.session_state.current_page = found_page
#                     st.session_state.last_search_result = {
#                         'page': found_page,
#                         'instances': instances,
#                         'term': term_used,
#                         'status': search_status,
#                         'ref': search_info['ref'] # Store ref for context
#                     }
#                     viewer_status_placeholder.empty() # Clear searching message, status shown below image
#                     st.toast(f"Found '{search_info['ref']}' on page {found_page}", icon="🎯")
#                     # Don't rerun here, let the rest of the column logic execute
#                 else:
#                     st.session_state.last_search_result = None # Clear previous result
#                     viewer_status_placeholder.error(search_status) # Show error message permanently

#             # --- Render Page ---
#             try:
#                 # Get total pages ONLY when needed, avoid reopening PDF constantly
#                 doc_info = fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf")
#                 total_pages = doc_info.page_count
#                 doc_info.close()
#             except Exception as e:
#                 st.error(f"Failed to load PDF for page count: {e}")
#                 total_pages = 1 # Avoid crashing navigation

#             # Ensure current page is valid
#             current_display_page = max(1, min(int(st.session_state.get('current_page', 1)), total_pages))
#             if current_display_page != st.session_state.get('current_page'):
#                 st.session_state.current_page = current_display_page # Correct if somehow invalid

#             # --- Navigation Buttons ---
#             nav_cols = st.columns([1, 3, 1]) # Prev | Page Info | Next
#             with nav_cols[0]:
#                 if st.button("⬅️ Prev", key="prev_page", disabled=(current_display_page <= 1)):
#                     st.session_state.current_page = max(1, current_display_page - 1)
#                     st.session_state.last_search_result = None # Clear search result on manual nav
#                     st.rerun()
#             with nav_cols[1]:
#                 page_info_text = f"Page {current_display_page} of {total_pages}"
#                 # Add search context if relevant
#                 if st.session_state.last_search_result and st.session_state.last_search_result['page'] == current_display_page:
#                      search_ref = st.session_state.last_search_result.get('ref', 'N/A')
#                      search_status = st.session_state.last_search_result.get('status', '')
#                      icon = "🎯" if "Found" in search_status else "⚠️"
#                      page_info_text = f"{icon} Page {current_display_page} / {total_pages} (Ref: {search_ref})"

#                 st.markdown(f"<div style='text-align: center; padding-top: 0.5rem;'>{page_info_text}</div>", unsafe_allow_html=True)

#             with nav_cols[2]:
#                 if st.button("Next ➡️", key="next_page", disabled=(current_display_page >= total_pages)):
#                     st.session_state.current_page = min(total_pages, current_display_page + 1)
#                     st.session_state.last_search_result = None # Clear search result on manual nav
#                     st.rerun()

#             # --- Determine Highlights and Render Image ---
#             st.markdown("---")
#             highlights_to_apply = None
#             render_status_override = None # Use status from search result if applicable
#             if st.session_state.last_search_result and st.session_state.last_search_result['page'] == current_display_page:
#                 highlights_to_apply = st.session_state.last_search_result['instances']
#                 render_status_override = st.session_state.last_search_result['status'] # Display search status

#             # Call the render function
#             image_bytes, render_status = render_pdf_page_to_image(
#                 st.session_state.pdf_bytes,
#                 current_display_page,
#                 highlight_instances=highlights_to_apply
#             )

#             # Display Image and Status
#             if image_bytes:
#                 st.image(image_bytes, caption=f"Page {current_display_page}", use_column_width='always')
#                 # Display status message (use search status if available, else render status)
#                 final_status = render_status_override if render_status_override else render_status
#                 if final_status and final_status != viewer_status_placeholder.error: # Avoid overwriting permanent error msg
#                     if "❌" in final_status or "Error" in final_status or "error" in final_status:
#                          viewer_status_placeholder.error(final_status) # Show errors prominently
#                     elif "⚠️" in final_status:
#                          viewer_status_placeholder.warning(final_status) # Show warnings
#                     elif "✨" in final_status or "Found" in final_status:
#                          # Use toast for success/found messages, clear placeholder
#                          # st.toast(final_status, icon="✨") # Toast might be too quick, use success box
#                          viewer_status_placeholder.success(final_status)
#                     else:
#                          viewer_status_placeholder.caption(final_status) # Use caption for other notes (e.g., plain render)
#             else:
#                 # Handle case where image rendering failed completely
#                 viewer_status_placeholder.error(f"Could not render page {current_display_page}. {render_status or ''}") # Show error

#         else:
#             st.info("Upload a PDF and run analysis to view pages.")
#             viewer_status_placeholder.empty() # Clear any residual status messages

#         # --- Close the sticky div ---
#         st.markdown('</div>', unsafe_allow_html=True)


# # --- Fallback messages if analysis hasn't run or no PDF ---
# elif st.session_state.pdf_bytes is not None and not st.session_state.processing_complete:
#      st.info("PDF loaded. Click 'Analyse Document' in the sidebar.")
# elif st.session_state.pdf_bytes is None:
#      st.info("⬆️ Upload a PDF file using the sidebar to begin.")




# ---------------------------


# # app.py
# # --- COMPLETE FILE ---

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

# # --- 1. SET PAGE CONFIG (MUST BE FIRST st COMMAND) ---
# st.set_page_config(
#     layout="wide",
#     page_title="JASPER - Agreement Analyzer",
#     page_icon="📄" # Optional: Set an emoji icon
# )

# # --- Inject custom CSS for sticky column (AFTER set_page_config) ---
# st.markdown("""
# <style>
#     /* Define a class for the sticky container */
#     .sticky-viewer-content {
#         position: sticky;
#         top: 55px; /* Adjust vertical offset from top */
#         z-index: 101; /* Ensure it's above other elements */
#         padding-bottom: 1rem; /* Add some space at the bottom */
#     }
#     /* Ensure background color matches theme for sticky element */
#     div.sticky-viewer-content {
#         background-color: var(--streamlit-background-color);
#     }
# </style>
# """, unsafe_allow_html=True)

# # --- 2. Configuration & Setup ---
# API_KEY_ENV_VAR = "GEMINI_API_KEY"
# MODEL_NAME = "gemini-2.5-pro-preview-03-25" # Updated model as 2.5 not available via API yet
# MAX_VALIDATION_RETRIES = 1
# RETRY_DELAY_SECONDS = 3
# PROMPT_FILE = "prompt.txt"
# LOGO_FILE = "jasper-logo-1.png"
# SEARCH_FALLBACK_MIN_LENGTH = 20 # Minimum characters for fallback sentence search

# # --- Get the absolute path to the directory containing app.py ---
# APP_DIR = os.path.dirname(os.path.abspath(__file__))

# # --- Load Prompt Text from File ---
# try:
#     prompt_path = os.path.join(APP_DIR, PROMPT_FILE)
#     with open(prompt_path, 'r', encoding='utf-8') as f:
#         full_prompt_text = f.read()
# except FileNotFoundError:
#     st.error(f"Error: Prompt file '{PROMPT_FILE}' not found in the application directory ({APP_DIR}).")
#     st.stop()
# except Exception as e:
#     st.error(f"Error reading prompt file '{PROMPT_FILE}': {e}")
#     st.stop()

# # --- Logo (in Sidebar) ---
# try:
#     logo_path = os.path.join(APP_DIR, LOGO_FILE)
#     st.sidebar.image(logo_path, width=150)
#     st.sidebar.markdown("---")
# except FileNotFoundError:
#     st.sidebar.warning(f"Logo '{LOGO_FILE}' not found.")
# except Exception as e:
#     st.sidebar.error(f"Error loading logo: {e}")

# # --- Schema Definition (Version 3 - Search Text) ---
# ai_response_schema_dict = {
#   "type": "array",
#   "description": "A list of questions, answers, justifications, and supporting evidence derived from the facility agreement analysis.",
#   "items": {
#     "type": "object",
#     "description": "Represents a single question's analysis, including potentially multiple pieces of evidence.",
#     "properties": {
#       "Question Number": {"type": "integer"},
#       "Question Category": {"type": "string"},
#       "Question": {"type": "string"},
#       "Answer": {"type": "string"},
#       "Answer Justification": {
#           "type": "string",
#           "description": "Overall justification for the answer, considering *all* supporting evidence found."
#       },
#       "Evidence": {
#         "type": "array",
#         "description": "A list of evidence items supporting the answer. Each item links a clause to its wording and the specific text used for searching. This array should be empty if the Answer is 'Information Not Found' or 'N/A'.",
#         "items": {
#           "type": "object",
#           "properties": {
#             "Clause Reference": {
#                 "type": "string",
#                 "description": "Specific clause number(s) or section(s) (e.g., 'Clause 14.1(a)', 'Section 5', 'Definition of Confidential Information')."
#             },
#             "Clause Wording": {
#                 "type": "string",
#                 "description": "Exact, full text of the referenced clause(s) or relevant sentence(s), including leading reference/heading."
#             },
#             "Searchable Clause Text": {
#                 "type": "string",
#                 "description": "The exact, searchable text content of the clause, excluding the clause reference itself."
#             }
#           },
#           "required": ["Clause Reference", "Clause Wording", "Searchable Clause Text"]
#         }
#       }
#     },
#     "required": [
#       "Question Number", "Question Category", "Question", "Answer", "Answer Justification", "Evidence"
#     ]
#   }
# }

# # Keys required for validation
# AI_REQUIRED_KEYS = set(ai_response_schema_dict['items']['required'])
# AI_EVIDENCE_REQUIRED_KEYS = set(ai_response_schema_dict['items']['properties']['Evidence']['items']['required'])

# # --- Excel Column Order ---
# EXCEL_COLUMN_ORDER = [
#     "File Name", "Generation Time", "Question Number", "Question Category",
#     "Question", "Answer", "Answer Justification",
#     "Clause References (Concatenated)",
#     "First Searchable Clause Text" # Changed from Page Numbers
# ]

# # --- Section Definitions ---
# SECTIONS_TO_RUN = {
#     "agreement_details": (1, 4),
#     "eligibility": (5, 36),
#     "confidentiality": (37, 63),
#     "additional_borrowers": (64, 66),
#     "interest_rate_provisions": (67, 71),
#     "prepayment_interest": (72, 78) # Corrected key from prepayment_interest to prepayment_fee
# }

# # --- System Instruction ---
# system_instruction_text = """You are analysing a facility agreement to understand whether the asset can be included within a Significant Risk Transfer or not (or with conditions, requirements, or exceptions) at NatWest. Your output must be precise, factual, and directly supported by evidence from the provided document(s). You must answer with UK spelling, not US. (e.g. 'analyse' is correct while 'analyse' is not). Adhere strictly to the JSON schema provided, ensuring every object in the output array contains all required keys."""


# # --- 3. Helper Function Definitions ---

# def filter_prompt_by_section(initial_full_prompt, section):
#     """Filters the main prompt to include only questions for a specific section."""
#     if section not in SECTIONS_TO_RUN:
#         raise ValueError(f"Invalid section specified: {section}. Must be one of {list(SECTIONS_TO_RUN.keys())}")
#     start_q, end_q = SECTIONS_TO_RUN[section]
#     questions_start_marker = "**Questions to Answer:**"
#     questions_end_marker = "**Final Instruction:**"
#     try:
#         start_index = initial_full_prompt.index(questions_start_marker)
#         end_index = initial_full_prompt.index(questions_end_marker)
#         prompt_header = initial_full_prompt[:start_index]
#         full_questions_block = initial_full_prompt[start_index + len(questions_start_marker):end_index].strip()
#         prompt_footer = initial_full_prompt[end_index:]
#     except ValueError:
#         st.error("Could not find question block markers ('**Questions to Answer:**' or '**Final Instruction:**') in the main prompt text. Check prompt.txt.")
#         raise ValueError("Could not find prompt markers.")

#     # Split questions based on the numbered format
#     question_entries = re.split(r'\n(?=\s*\d+\.\s*?\*\*Question Category:)', full_questions_block)
#     if not question_entries or len(question_entries) < 2: # Fallback splits
#         question_entries = re.split(r'\n(?=\d+\.\s)', full_questions_block)
#         if not question_entries or len(question_entries) < 2:
#              question_entries = re.split(r'\n(?=\*\*Question Category:)', full_questions_block)

#     filtered_question_texts = []
#     processed_q_nums = set()
#     for entry in question_entries:
#         entry = entry.strip()
#         if not entry: continue
#         match = re.match(r'^\s*(\d+)\.', entry)
#         if match:
#             q_num = int(match.group(1))
#             if start_q <= q_num <= end_q:
#                 filtered_question_texts.append(entry)
#                 processed_q_nums.add(q_num)

#     expected_q_nums = set(range(start_q, end_q + 1))
#     missing_q_nums = expected_q_nums - processed_q_nums
#     if missing_q_nums:
#          try: st.warning(f"Parsing might have missed expected question numbers in range {start_q}-{end_q} for section '{section}': {sorted(list(missing_q_nums))}")
#          except Exception: pass # Avoid errors if st called too early

#     if not filtered_question_texts:
#         st.error(f"No questions found for section '{section}' in range {start_q}-{end_q}. Check prompt formatting and split logic.")
#         raise ValueError(f"Failed to extract questions for section '{section}'.")

#     filtered_questions_string = "\n\n".join(filtered_question_texts)
#     task_end_marker = "specified section." # Add focus note to prompt
#     insert_pos = prompt_header.find(task_end_marker)
#     section_note = f"\n\n**Current Focus:** Answering ONLY questions for the '{section.upper()}' section (Questions {start_q}-{end_q}). The list below contains ONLY these questions.\n"
#     if insert_pos != -1:
#          insert_pos += len(task_end_marker)
#          final_header = prompt_header[:insert_pos] + section_note + prompt_header[insert_pos:]
#     else: final_header = prompt_header + section_note
#     final_prompt_for_api = f"{final_header}{questions_start_marker}\n\n{filtered_questions_string}\n\n{prompt_footer}"
#     return final_prompt_for_api


# def validate_ai_data(data, section_name):
#     """Validates the structure of the AI response based on the new schema. Returns validated data (list) and list of issue strings."""
#     if not isinstance(data, list):
#         st.error(f"Crit Val Error: AI response for '{section_name}' not list.")
#         return None, [f"CRITICAL: Response for '{section_name}' was not a list."]

#     validated_data = []
#     issues_list = []
#     for index, item in enumerate(data): # Loop outer items
#         q_num_str = f"Q#{item.get('Question Number', f'Item {index}')}"
#         is_outer_valid = True
#         if not isinstance(item, dict):
#             issues_list.append(f"{q_num_str}: Not a dictionary."); continue

#         missing_outer_keys = AI_REQUIRED_KEYS - set(item.keys())
#         if missing_outer_keys: issues_list.append(f"{q_num_str}: Missing: {missing_outer_keys}"); is_outer_valid = False
#         evidence_list = item.get("Evidence")
#         if not isinstance(evidence_list, list): issues_list.append(f"{q_num_str}: 'Evidence' not list."); is_outer_valid = False
#         else: # Validate inner evidence items
#             for ev_index, ev_item in enumerate(evidence_list):
#                 if not isinstance(ev_item, dict): issues_list.append(f"{q_num_str} Ev[{ev_index}]: Not dict."); is_outer_valid = False; continue
#                 missing_ev_keys = AI_EVIDENCE_REQUIRED_KEYS - set(ev_item.keys())
#                 if missing_ev_keys: issues_list.append(f"{q_num_str} Ev[{ev_index}]: Missing: {missing_ev_keys}"); is_outer_valid = False
#                 # Check types for new schema
#                 if not isinstance(ev_item.get("Clause Reference"), str): issues_list.append(f"{q_num_str} Ev[{ev_index}]: Ref not str."); is_outer_valid = False
#                 if not isinstance(ev_item.get("Clause Wording"), str): issues_list.append(f"{q_num_str} Ev[{ev_index}]: Wording not str."); is_outer_valid = False
#                 if not isinstance(ev_item.get("Searchable Clause Text"), str): issues_list.append(f"{q_num_str} Ev[{ev_index}]: SearchText not str."); is_outer_valid = False # Check new field

#         if is_outer_valid: validated_data.append(item)

#     if issues_list: # Prepend summary header & show warning
#         issues_list.insert(0, f"Validation Issues [Section: {section_name}] ({len(validated_data)} passed):")
#         st.warning("Validation Issues Detected:\n" + "\n".join([f"- {issue}" for issue in issues_list[1:]]))
#     return validated_data, issues_list


# def generate_section_analysis(section, uploaded_file_ref, status_placeholder):
#     """Generates analysis, handles retries, validation. Returns (data, status, warnings)."""
#     status_placeholder.info(f"🔄 Starting: {section}...")
#     section_warnings = []
#     model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction_text)
#     generation_config = types.GenerationConfig(
#         response_mime_type="application/json", response_schema=ai_response_schema_dict,
#         temperature=0.0, top_p=0.05, top_k=1
#     )
#     final_validated_data = None

#     for attempt in range(1, 1 + MAX_VALIDATION_RETRIES + 1):
#         if attempt > 1: status_placeholder.info(f"⏳ Retrying: '{section}' (Attempt {attempt})..."); time.sleep(RETRY_DELAY_SECONDS)
#         try:
#             prompt_for_api = filter_prompt_by_section(full_prompt_text, section)
#             contents = [uploaded_file_ref, prompt_for_api]
#             status_placeholder.info(f"🧠 Calling AI: {section} (Attempt {attempt})..."); response = model.generate_content(contents=contents, generation_config=generation_config)
#             parsed_ai_data = None; validated_ai_data = None; validation_issues = []
#             status_placeholder.info(f"🔍 Processing: {section}...");
#             if response.parts:
#                 full_response_text = response.text
#                 try:
#                     # Gemini sometimes wraps the JSON in ```json ... ```, try to strip it
#                     match = re.search(r"```json\s*([\s\S]*?)\s*```", full_response_text)
#                     if match:
#                          json_text = match.group(1).strip()
#                     else:
#                          json_text = full_response_text.strip()

#                     parsed_ai_data = json.loads(json_text)
#                     status_placeholder.info(f"✔️ Validating: {section}..."); validated_ai_data, validation_issues = validate_ai_data(parsed_ai_data, section); section_warnings.extend(validation_issues)
#                     if validated_ai_data is not None and len(validated_ai_data) > 0: final_validated_data = validated_ai_data; break
#                     elif validated_ai_data is None: error_msg = f"Crit Val Error: '{section}'."; section_warnings.append(error_msg); break # Error shown by validate_ai_data
#                     else: status_placeholder.warning(f"⚠️ Validation 0 items: {section} (Attempt {attempt}).") # Retry if possible
#                 except json.JSONDecodeError as json_err: error_msg = f"JSON Error {attempt}: {section}: {json_err}"; st.error(error_msg); st.code(full_response_text); section_warnings.append(error_msg) # Show raw response on JSON error
#             else: # Handle empty response
#                 block_reason = "Unknown"; block_message = "N/A";
#                 try:
#                    if response.prompt_feedback:
#                      if response.prompt_feedback.block_reason: block_reason = response.prompt_feedback.block_reason.name
#                      block_message = response.prompt_feedback.block_reason_message or "N/A"
#                 except AttributeError: pass
#                 warn_msg = f"Empty Response {attempt}: {section}. Reason: {block_reason}. Msg: {block_message}"; st.warning(warn_msg); section_warnings.append(warn_msg)
#         except types.generation_types.BlockedPromptException as bpe: error_msg = f"API Block Error {attempt}: {section}: {bpe}"; st.error(error_msg); section_warnings.append(error_msg)
#         except types.generation_types.StopCandidateException as sce: error_msg = f"API Stop Error {attempt}: {section}: {sce}"; st.error(error_msg); section_warnings.append(error_msg)
#         except ValueError as ve: error_msg = f"Prompt Filter Error {attempt}: {section}: {ve}"; st.error(error_msg); section_warnings.append(error_msg); break # Stop section on prompt error
#         except Exception as e: error_msg = f"Processing Error {attempt}: {section}: {type(e).__name__}"; st.error(error_msg); section_warnings.append(error_msg); section_warnings.append(f"Traceback in logs."); print(traceback.format_exc()); # Print traceback for debug
#         if final_validated_data is not None: break # Exit loop if successful

#     if final_validated_data is not None: status_placeholder.info(f"✅ Completed: {section}."); st.toast(f"Processed: {section}", icon="✅"); return final_validated_data, "Success", section_warnings
#     else: status_placeholder.error(f"❌ Failed: {section} after {attempt} attempts."); return None, "Failed", section_warnings


# @st.cache_data(show_spinner=False)
# def find_text_in_pdf(_pdf_bytes, search_text):
#     """
#     Searches the entire PDF for the given text and returns the page number and instances.
#     Includes fallback logic to search for the first sentence.

#     Returns:
#         tuple: (found_page_number, highlight_instances, search_term_used, status_message)
#                - found_page_number (int): 1-based page number where text was found, or None.
#                - highlight_instances (list): List of fitz.Rect/Quad instances for highlighting, or None.
#                - search_term_used (str): The actual text string used for the successful search, or None.
#                - status_message (str): Message indicating search outcome (found, fallback, not found, error).
#     """
#     if not _pdf_bytes or not search_text:
#         return None, None, None, "Invalid input (missing PDF bytes or search text)."

#     doc = None
#     try:
#         doc = fitz.open(stream=_pdf_bytes, filetype="pdf")
#         search_flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES

#         # --- Attempt 1: Search for the full text ---
#         full_search_term = search_text.strip()
#         for page_index in range(doc.page_count):
#             page = doc.load_page(page_index)
#             instances = page.search_for(full_search_term, flags=search_flags, quads=False)
#             if instances:
#                 doc.close()
#                 status = f"✨ Found full text on page {page_index + 1}."
#                 return page_index + 1, instances, full_search_term, status

#         # --- Attempt 2: Fallback to the first sentence ---
#         first_sentence = ""
#         # Basic sentence split - adjust regex if needed for more complex cases
#         sentences = re.split(r'(?<=[.?!])\s+', full_search_term)
#         if sentences and len(sentences[0].strip()) >= SEARCH_FALLBACK_MIN_LENGTH:
#             first_sentence = sentences[0].strip()

#         if first_sentence:
#             # st.warning(f"Full text not found. Trying first sentence: '{first_sentence[:60]}...'") # Optional debug
#             for page_index in range(doc.page_count):
#                 page = doc.load_page(page_index)
#                 instances = page.search_for(first_sentence, flags=search_flags, quads=False)
#                 if instances:
#                     doc.close()
#                     status = f"⚠️ Found using first sentence on page {page_index + 1} (full text query failed)."
#                     return page_index + 1, instances, first_sentence, status

#         # --- Text Not Found ---
#         doc.close()
#         return None, None, None, f"❌ Text not found in document (tried full text and first sentence)."

#     except Exception as e:
#         if doc: doc.close()
#         print(f"ERROR searching PDF: {e}") # Log error to console
#         print(traceback.format_exc())
#         return None, None, None, f"❌ Error during PDF search: {e}"


# # @st.cache_data(show_spinner=False) # Caching render might interfere with dynamic highlights
# def render_pdf_page_to_image(_pdf_bytes, page_number, highlight_instances=None, dpi=150):
#     """
#     Renders a specific PDF page to an image, applying highlights if instances are provided.

#     Args:
#         _pdf_bytes: Bytes of the PDF file.
#         page_number: The 1-based page number to render.
#         highlight_instances: List of fitz.Rect/Quad objects to highlight on the page.
#         dpi: Resolution for rendering.

#     Returns:
#         tuple: (image_bytes or None, status_message or None)
#                Status message indicates success/failure of rendering/highlighting.
#     """
#     if not _pdf_bytes or page_number < 1:
#         return None, "Invalid input (no PDF bytes or page number < 1)."

#     doc = None
#     image_bytes = None
#     render_status_message = f"Rendered page {page_number}." # Default status

#     try:
#         doc = fitz.open(stream=_pdf_bytes, filetype="pdf")
#         page_index = page_number - 1
#         if page_index < 0 or page_index >= doc.page_count:
#             doc.close()
#             return None, f"Page number {page_number} is out of range (1-{doc.page_count})."

#         page = doc.load_page(page_index)
#         highlight_applied = False
#         highlight_count = 0

#         # --- Add Annotations if instances provided ---
#         if highlight_instances:
#             try:
#                 for inst in highlight_instances:
#                     highlight = page.add_highlight_annot(inst)
#                     # Optional: Style the highlight
#                     # highlight.set_colors(stroke=[1, 1, 0], fill=[1, 1, 0.8]) # Yellowish
#                     # highlight.set_opacity(0.5) # Semi-transparent
#                     highlight.update()
#                     highlight_count += 1
#                 if highlight_count > 0:
#                     highlight_applied = True
#                     # Status message from search function will override this if search was done
#                     render_status_message = f"✨ Rendered page {page_number} with {highlight_count} highlight(s)."
#             except Exception as highlight_err:
#                 print(f"ERROR applying highlights on page {page_number}: {highlight_err}") # Log to console
#                 render_status_message = f"⚠️ Error applying highlights on page {page_number}: {highlight_err}"

#         # --- Render page ---
#         pix = page.get_pixmap(dpi=dpi)
#         image_bytes = pix.tobytes("png")

#     except Exception as e:
#         print(f"ERROR rendering PDF page {page_number}: {e}") # Log error to console
#         # print(traceback.format_exc()) # Optional full traceback
#         render_status_message = f"❌ Error rendering page {page_number}: {e}"
#         image_bytes = None # Ensure no image returned on error
#     finally:
#         if doc: doc.close()

#     return image_bytes, render_status_message


# # --- 4. Initialize Session State ---
# if 'current_page' not in st.session_state: st.session_state.current_page = 1
# if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
# if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None
# if 'pdf_bytes_processed' not in st.session_state: st.session_state.pdf_bytes_processed = None
# if 'processing_complete' not in st.session_state: st.session_state.processing_complete = False
# if 'run_key' not in st.session_state: st.session_state.run_key = 0
# if 'run_status_summary' not in st.session_state: st.session_state.run_status_summary = []
# if 'excel_data' not in st.session_state: st.session_state.excel_data = None
# # New state for search mechanism
# if 'search_trigger' not in st.session_state: st.session_state.search_trigger = None # Stores {'text': search_text, 'ref': clause_ref}
# if 'last_search_result' not in st.session_state: st.session_state.last_search_result = None # Stores {'page': num, 'instances': [], 'term': str, 'status': str}


# # --- 5. Streamlit UI Logic ---

# st.title("JASPER - Facility Agreement Analyzer")
# st.markdown("Upload a PDF agreement, click 'Analyse Document'. Results are grouped below. Click **Clause References** in the 'Evidence' section to find and view the relevant page with highlights.")
# st.sidebar.markdown("## Controls")

# # --- File Upload (in Sidebar) ---
# uploaded_file_obj = st.sidebar.file_uploader(
#     "Upload Facility Agreement PDF", type="pdf", key=f"pdf_uploader_{st.session_state.run_key}"
# )

# # Process new file upload only if bytes differ from last processed
# if uploaded_file_obj is not None:
#     uploaded_bytes = uploaded_file_obj.getvalue()
#     if uploaded_bytes != st.session_state.get('pdf_bytes_processed'):
#         st.session_state.pdf_bytes = uploaded_bytes
#         st.session_state.pdf_bytes_processed = uploaded_bytes # Mark as processed for this session run
#         # Reset dependent states
#         st.session_state.analysis_results = None; st.session_state.processing_complete = False
#         st.session_state.current_page = 1; st.session_state.run_status_summary = []
#         st.session_state.excel_data = None; st.session_state.search_trigger = None; st.session_state.last_search_result = None
#         st.toast("✅ New PDF file loaded.", icon="📄")
#         st.rerun() # Rerun to reflect cleared state
# elif 'pdf_bytes_processed' in st.session_state: # If file removed, clear flag
#      st.session_state.pdf_bytes_processed = None


# # --- Analysis Trigger (in Sidebar) ---
# analyse_disabled = st.session_state.processing_complete or st.session_state.pdf_bytes is None
# if st.sidebar.button("✨ Analyse Document", key="analyse_button", disabled=analyse_disabled, use_container_width=True, type="primary"):
#     # Reset states for the new run
#     st.session_state.analysis_results = None; st.session_state.processing_complete = False
#     st.session_state.current_page = 1; st.session_state.run_key += 1; st.session_state.run_status_summary = []
#     st.session_state.excel_data = None; st.session_state.search_trigger = None; st.session_state.last_search_result = None

#     run_start_time = datetime.now(); run_timestamp_str = run_start_time.strftime("%Y-%m-%d %H:%M:%S")
#     base_file_name = getattr(uploaded_file_obj, 'name', 'uploaded_file')

#     # Status placeholders in main area
#     status_container = st.container()
#     progress_bar = status_container.progress(0, text="Initializing...")
#     status_text = status_container.empty()

#     temp_dir = "temp_uploads"; temp_file_path = os.path.join(APP_DIR, temp_dir, f"{run_start_time.strftime('%Y%m%d%H%M%S')}_{base_file_name}")
#     os.makedirs(os.path.dirname(temp_file_path), exist_ok=True) # Ensure directory exists

#     gemini_uploaded_file_ref = None; all_validated_data = []; overall_success = True

#     try:
#         status_text.info("💾 Saving file temporarily..."); progress_bar.progress(5, text="Saving...")
#         with open(temp_file_path, "wb") as f: f.write(st.session_state.pdf_bytes)

#         status_text.info("🚀 Uploading to Google Cloud..."); progress_bar.progress(10, text="Uploading...")
#         # Add retry logic for file upload if needed
#         for upload_attempt in range(3):
#             try:
#                 gemini_uploaded_file_ref = genai.upload_file(path=temp_file_path)
#                 st.toast(f"File '{gemini_uploaded_file_ref.display_name}' uploaded.", icon="☁️")
#                 break # Success
#             except Exception as upload_err:
#                 if upload_attempt < 2:
#                     st.warning(f"File upload failed (Attempt {upload_attempt+1}): {upload_err}. Retrying...")
#                     time.sleep(2)
#                 else:
#                     st.error(f"File upload failed after multiple attempts: {upload_err}")
#                     raise # Re-raise the last error
#         if not gemini_uploaded_file_ref:
#              raise Exception("Failed to upload file to Gemini.")

#         progress_bar.progress(15, text="Uploaded.")

#         num_sections = len(SECTIONS_TO_RUN); progress_per_section = (95 - 15) / num_sections if num_sections > 0 else 0

#         # --- Process Sections ---
#         for i, section_name in enumerate(SECTIONS_TO_RUN.keys()):
#             current_progress = int(15 + (i * progress_per_section))
#             progress_bar.progress(current_progress, text=f"Starting {section_name}...")
#             section_data, section_status, section_warnings = generate_section_analysis(section_name, gemini_uploaded_file_ref, status_text)
#             st.session_state.run_status_summary.append({"section": section_name, "status": section_status, "warnings": section_warnings})
#             if section_status == "Success" and section_data:
#                 for item in section_data: item["File Name"] = base_file_name; item["Generation Time"] = run_timestamp_str
#                 all_validated_data.extend(section_data)
#             else: overall_success = False

#         st.session_state.analysis_results = all_validated_data
#         progress_bar.progress(100, text="Analysis Complete!")
#         if overall_success: status_text.success("🏁 Analysis finished successfully!")
#         else: status_text.warning("🏁 Analysis finished with some issues (see summary below).")
#         st.session_state.processing_complete = True

#     except Exception as main_err:
#          st.error(f"❌ CRITICAL ERROR during analysis: {main_err}"); st.error(traceback.format_exc());
#          overall_success = False; st.session_state.processing_complete = False
#          st.session_state.run_status_summary.append({"section": "Overall Process", "status": "Critical Error", "warnings": [str(main_err), "Check server logs for traceback."]})
#          status_text.error("Analysis failed due to a critical error.")
#     finally: # Cleanup
#         # Delay before clearing status, keeps final message visible longer
#         time.sleep(4)
#         status_text.empty()
#         progress_bar.empty()
#         # Cleanup temp files and Gemini file
#         if gemini_uploaded_file_ref and hasattr(gemini_uploaded_file_ref, 'name'):
#              try:
#                  status_text.info(f"🧹 Deleting temporary cloud file: {gemini_uploaded_file_ref.display_name}...")
#                  genai.delete_file(name=gemini_uploaded_file_ref.name)
#                  status_text.info("🧹 Cloud file deleted.")
#                  time.sleep(1) # Short pause
#                  status_text.empty()
#              except Exception as delete_err:
#                  st.warning(f"Could not delete cloud file '{gemini_uploaded_file_ref.name}': {delete_err}")
#         if os.path.exists(temp_file_path):
#              try: os.remove(temp_file_path)
#              except Exception: pass

#     st.rerun() # Rerun to update display after analysis


# # --- Run Status Summary Expander (Displayed near the top) ---
# if st.session_state.run_status_summary:
#     final_status = "✅ Success"
#     if any(s['status'] != "Success" for s in st.session_state.run_status_summary): final_status = "⚠️ Completed with Issues"
#     if any("Critical" in s['status'] or "Fail" in s['status'] for s in st.session_state.run_status_summary): final_status = "❌ Failed"
#     # Expand automatically if not pure success
#     with st.expander(f"📊 Last Analysis Run Summary ({final_status})", expanded=(final_status != "✅ Success")):
#         for item in st.session_state.run_status_summary:
#             icon = "✅" if item['status'] == "Success" else "❌" if "Fail" in item['status'] or "Error" in item['status'] else "⚠️"
#             st.markdown(f"**{item['section']}**: {icon} {item['status']}")
#             if item['warnings']:
#                  # Display warnings associated with the section status
#                  with st.container():
#                      # Filter out the basic validation summary header if present
#                      filtered_warnings = [msg for msg in item['warnings'] if not (isinstance(msg, str) and msg.startswith("Validation Issues [Section:"))]
#                      if filtered_warnings:
#                          st.caption("Details:")
#                          for msg in filtered_warnings:
#                             if isinstance(msg, str) and msg.startswith("CRITICAL:"): st.error(f" L> {msg}")
#                             elif isinstance(msg, str) and msg.startswith("JSON Error"): st.error(f" L> {msg}")
#                             elif isinstance(msg, str) and msg.startswith("API Block"): st.error(f" L> {msg}")
#                             elif isinstance(msg, str) and msg.startswith("API Stop"): st.error(f" L> {msg}")
#                             elif isinstance(msg, str) and msg.startswith("Empty Response"): st.warning(f" L> {msg}")
#                             elif isinstance(msg, str) and "Missing:" in msg: st.warning(f" L> {msg}") # Validation field missing
#                             elif isinstance(msg, list): st.warning(" L> "+"\n L> ".join(map(str, msg))) # Handle list case
#                             else: st.caption(f" L> {msg}") # General warnings/info


# # --- Display Area (Results and PDF Viewer) ---
# if st.session_state.analysis_results is not None:
#     col1, col2 = st.columns([6, 4]) # Results | PDF Viewer

#     # --- Column 1: Analysis Results (Grouped by Category using Tabs) ---
#     with col1:
#         st.subheader("Analysis Results")
#         results_list = st.session_state.analysis_results

#         if not results_list:
#             st.info("Analysis complete, but no valid results were generated. Check the summary above.")
#         else:
#             # Group results by category
#             grouped_results = defaultdict(list)
#             categories_ordered = []
#             for item in results_list:
#                 category = item.get("Question Category", "Uncategorized")
#                 if category not in grouped_results: categories_ordered.append(category)
#                 grouped_results[category].append(item)

#             if categories_ordered:
#                 category_tabs = st.tabs(categories_ordered)
#                 for i, category in enumerate(categories_ordered):
#                     with category_tabs[i]:
#                         category_items = grouped_results[category]
#                         # Sort items by Question Number within the category
#                         category_items.sort(key=lambda x: x.get('Question Number', float('inf')))

#                         for index, result_item in enumerate(category_items):
#                             q_num = result_item.get('Question Number', 'N/A'); question_text = result_item.get('Question', 'N/A')

#                             with st.expander(f"**Q{q_num}:** {question_text}"):
#                                 st.markdown(f"**Answer:** {result_item.get('Answer', 'N/A')}")
#                                 st.markdown("---")

#                                 # Evidence Section
#                                 evidence_list = result_item.get('Evidence', [])
#                                 if evidence_list:
#                                     st.markdown("**Evidence:** (Click reference to find & view page with highlights)")
#                                     for ev_index, evidence_item in enumerate(evidence_list):
#                                         clause_ref = evidence_item.get('Clause Reference', 'N/A')
#                                         # page_num = evidence_item.get('Page Number', 0) # REMOVED
#                                         clause_wording = evidence_item.get('Clause Wording', None) # Keep for context maybe
#                                         search_text = evidence_item.get('Searchable Clause Text', None) # << NEW

#                                         # Button to trigger search (only if search_text exists)
#                                         if search_text:
#                                             button_key = f"search_btn_{category}_{q_num}_{index}_{ev_index}"
#                                             button_label = f"Clause: **{clause_ref or 'Link'}** (Find & View)"
#                                             if st.button(button_label, key=button_key, help=f"Search for text related to '{clause_ref or 'this evidence'}' and view the page."):
#                                                 st.session_state.search_trigger = {'text': search_text, 'ref': clause_ref}
#                                                 st.session_state.last_search_result = None # Clear previous search result immediately
#                                                 st.rerun() # Rerun to trigger search in viewer column
#                                         elif clause_ref != 'N/A':
#                                             # Show reference without button if no searchable text provided
#                                             st.markdown(f"- Clause: **{clause_ref}** (No searchable text provided)")

#                                         # Optionally display Clause Wording or Searchable Text for context (can be long)
#                                         # with st.popover("Show Full Clause Wording"):
#                                         #    st.caption(f"Reference: {clause_ref}")
#                                         #    st.code(clause_wording or "N/A", language=None)
#                                         # with st.popover("Show Searchable Text"):
#                                         #     st.caption(f"Reference: {clause_ref}")
#                                         #     st.code(search_text or "N/A", language=None)

#                                 else: st.markdown("**Evidence:** None provided.")
#                                 st.markdown("---") # Separator after evidence block

#                                 # Justification Section
#                                 st.markdown("**Answer Justification:**")
#                                 just_key = f"justification_{category}_{q_num}_{index}" # Unique key
#                                 st.text_area(label="Justification", value=result_item.get('Answer Justification', ''), height=100, disabled=True, label_visibility="collapsed", key=just_key)

#             else: st.warning("Results generated, but could not group by category.")


#             # --- Excel Download Preparation (in Sidebar) ---
#             st.sidebar.markdown("---")
#             st.sidebar.markdown("## Export")
#             if st.sidebar.button("Prepare Data for Excel Download", key="prep_excel", use_container_width=True):
#                 st.session_state.excel_data = None; excel_prep_status = st.sidebar.empty()
#                 excel_prep_status.info("Preparing Excel...")
#                 try:
#                     excel_rows = []
#                     for item in results_list: # Use full results list
#                         references = []; first_search_text = "N/A"
#                         evidence = item.get("Evidence")
#                         if evidence:
#                             for i, ev in enumerate(evidence):
#                                 references.append(str(ev.get("Clause Reference", "N/A")))
#                                 # pages.append(str(ev.get("Page Number", "0"))) # REMOVED
#                                 if i == 0: first_search_text = ev.get("Searchable Clause Text", "N/A") # Get first searchable text
#                         excel_row = {
#                             "File Name": item.get("File Name", ""),
#                             "Generation Time": item.get("Generation Time", ""),
#                             "Question Number": item.get("Question Number"),
#                             "Question Category": item.get("Question Category"),
#                             "Question": item.get("Question"),
#                             "Answer": item.get("Answer"),
#                             "Answer Justification": item.get("Answer Justification"),
#                             "Clause References (Concatenated)": "; ".join(references) if references else "N/A",
#                             # "Page Numbers (Concatenated)": "; ".join(pages) if pages else "N/A", # REMOVED
#                             "First Searchable Clause Text": first_search_text # ADDED/RENAMED
#                         }
#                         excel_rows.append(excel_row)

#                     if not excel_rows:
#                          excel_prep_status.warning("No data to export.")
#                          st.session_state.excel_data = None
#                     else:
#                         df_excel = pd.DataFrame(excel_rows)
#                         # Apply column order, handle missing columns gracefully
#                         final_columns = []
#                         for col in EXCEL_COLUMN_ORDER:
#                             if col in df_excel.columns:
#                                 final_columns.append(col)
#                             else:
#                                 # Add missing column with None/NaN if it was expected
#                                 # df_excel[col] = None
#                                 # final_columns.append(col)
#                                 st.warning(f"Expected Excel column '{col}' not found in data, skipping.")

#                         df_excel = df_excel[final_columns] # Reorder with available columns

#                         # Write to BytesIO
#                         output = io.BytesIO()
#                         with pd.ExcelWriter(output, engine='openpyxl') as writer:
#                             df_excel.to_excel(writer, index=False, sheet_name='Analysis')
#                         st.session_state.excel_data = output.getvalue()
#                         excel_prep_status.success("Excel ready for download!")
#                         time.sleep(2); excel_prep_status.empty() # Clear message

#                 except Exception as excel_err:
#                      excel_prep_status.error(f"Excel Prep Error: {excel_err}")
#                      print(traceback.format_exc()) # Log traceback for debug

#             # --- Actual Download Button (in Sidebar) ---
#             if st.session_state.excel_data:
#                  # Try to get filename from original upload object if still available
#                  current_filename = "analysis"
#                  if uploaded_file_obj: # Use current state if available
#                      current_filename = uploaded_file_obj.name
#                  elif 'pdf_bytes_processed' in st.session_state and isinstance(st.session_state.get('analysis_results'), list) and st.session_state.analysis_results:
#                      # Fallback: Try to get from first result item if upload object gone
#                      current_filename = st.session_state.analysis_results[0].get("File Name", "analysis")

#                  safe_base_name = re.sub(r'[^\w\s-]', '', current_filename.split('.')[0]).strip()
#                  download_filename = f"Analysis_{safe_base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
#                  st.sidebar.download_button(
#                      label="📥 Download Results as Excel", data=st.session_state.excel_data,
#                      file_name=download_filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#                      key="download_excel_final", use_container_width=True
#                  )

# # --- Column 2: Page Viewer ---
#     with col2:
#         # Wrap content in the styled div for sticky effect
#         st.markdown('<div class="sticky-viewer-content">', unsafe_allow_html=True)

#         st.subheader("📄 Page Viewer")
#         viewer_status_placeholder = st.empty() # For showing search/render status

#         if st.session_state.pdf_bytes:
#             # --- Perform Search if Triggered ---
#             if st.session_state.search_trigger:
#                 search_info = st.session_state.search_trigger
#                 st.session_state.search_trigger = None # Consume the trigger
#                 viewer_status_placeholder.info(f"🔎 Searching for text related to: '{search_info['ref']}'...")
#                 found_page, instances, term_used, search_status = find_text_in_pdf(
#                     st.session_state.pdf_bytes, search_info['text']
#                 )
#                 if found_page:
#                     st.session_state.current_page = found_page
#                     st.session_state.last_search_result = {
#                         'page': found_page,
#                         'instances': instances,
#                         'term': term_used,
#                         'status': search_status,
#                         'ref': search_info['ref'] # Store ref for context
#                     }
#                     viewer_status_placeholder.empty() # Clear searching message, status shown below image
#                     st.toast(f"Found reference '{search_info['ref']}' on page {found_page}", icon="🎯")
#                 else:
#                     st.session_state.last_search_result = None # Clear previous result
#                     viewer_status_placeholder.error(search_status) # Show error message permanantly
#                     # Optionally keep current page, or reset to 1? Let's keep it.
#                     # st.session_state.current_page = 1

#             # --- Render Page ---
#             try:
#                 with fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf") as doc:
#                     total_pages = doc.page_count
#             except Exception as e:
#                 st.error(f"Failed to load PDF for page count: {e}")
#                 total_pages = 1 # Avoid crashing navigation

#             # Ensure current page is valid
#             current_display_page = max(1, min(int(st.session_state.get('current_page', 1)), total_pages))
#             if current_display_page != st.session_state.get('current_page'):
#                 st.session_state.current_page = current_display_page # Correct if somehow invalid

#             # --- Navigation Buttons ---
#             nav_cols = st.columns([1, 3, 1]) # Prev | Page Info | Next
#             with nav_cols[0]:
#                 if st.button("⬅️ Prev", key="prev_page", disabled=(current_display_page <= 1)):
#                     st.session_state.current_page = max(1, current_display_page - 1)
#                     st.session_state.last_search_result = None # Clear search result on manual nav
#                     st.rerun()
#             with nav_cols[1]:
#                 page_info_text = f"Page {current_display_page} of {total_pages}"
#                 if st.session_state.last_search_result and st.session_state.last_search_result['page'] == current_display_page:
#                      page_info_text += f" (Found Ref: '{st.session_state.last_search_result['ref']}')"
#                 st.markdown(f"<div style='text-align: center; padding-top: 0.5rem;'>{page_info_text}</div>", unsafe_allow_html=True)
#             with nav_cols[2]:
#                 if st.button("Next ➡️", key="next_page", disabled=(current_display_page >= total_pages)):
#                     st.session_state.current_page = min(total_pages, current_display_page + 1)
#                     st.session_state.last_search_result = None # Clear search result on manual nav
#                     st.rerun()

#             # --- Determine Highlights and Render Image ---
#             st.markdown("---")
#             highlights_to_apply = None
#             render_status_override = None # Use status from search result if applicable
#             if st.session_state.last_search_result and st.session_state.last_search_result['page'] == current_display_page:
#                 highlights_to_apply = st.session_state.last_search_result['instances']
#                 render_status_override = st.session_state.last_search_result['status']

#             # Call the render function
#             image_bytes, render_status = render_pdf_page_to_image(
#                 st.session_state.pdf_bytes,
#                 current_display_page,
#                 highlight_instances=highlights_to_apply
#             )

#             # Display Image and Status
#             if image_bytes:
#                 st.image(image_bytes, caption=f"Page {current_display_page}", use_column_width='always')
#                 # Display status message (use search status if available, else render status)
#                 final_status = render_status_override if render_status_override else render_status
#                 if final_status:
#                     if "❌" in final_status or "Error" in final_status or "error" in final_status:
#                          viewer_status_placeholder.error(final_status) # Show errors prominently
#                     elif "⚠️" in final_status:
#                          viewer_status_placeholder.warning(final_status) # Show warnings
#                     elif "✨" in final_status or "Found" in final_status:
#                          # Use toast for success/found messages, clear placeholder
#                          st.toast(final_status, icon="✨")
#                          viewer_status_placeholder.empty()
#                     else:
#                          viewer_status_placeholder.caption(final_status) # Use caption for other notes (e.g., plain render)
#             else:
#                 # Handle case where image rendering failed completely
#                 viewer_status_placeholder.error(f"Could not render page {current_display_page}. {render_status or ''}") # Show error

#         else:
#             st.info("Upload a PDF and run analysis to view pages.")
#             viewer_status_placeholder.empty() # Clear any residual status messages

#         # --- Close the sticky div ---
#         st.markdown('</div>', unsafe_allow_html=True)


# # --- Fallback messages if analysis hasn't run or no PDF ---
# elif st.session_state.pdf_bytes is not None and not st.session_state.processing_complete:
#      st.info("PDF loaded. Click 'Analyse Document' in the sidebar.")
# elif st.session_state.pdf_bytes is None:
#      st.info("⬆️ Upload a PDF file using the sidebar to begin.")