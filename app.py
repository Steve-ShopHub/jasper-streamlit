# app.py
# --- COMPLETE FILE (v4 - Added Plotly Scatter Plot) ---

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
import plotly.express as px # <-- IMPORT PLOTLY

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
MODEL_NAME = "gemini-2.5-pro-preview-03-25" 
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
# Define ALL possible sections here. Selection happens in the UI.
ALL_SECTIONS = {
    "agreement_details": (1, 4),
    "eligibility_part_1": (5, 20),
    "eligibility_part_2": (21, 34),
    "eligibility_summary": (35, 36),
    "confidentiality": (37, 63),
    "additional_borrowers": (64, 66),
    "interest_rate_provisions": (67, 71),
    "prepayment_fee": (72, 78)
}
# This dictionary holds the master definition. Helper functions will use this.

# --- System Instruction ---
system_instruction_text = """You are an AI assistant specialized in analyzing legal facility agreements. Carefully read the provided document and answer the specific questions listed in the user prompt for the designated section only. Adhere strictly to the requested JSON output schema. Prioritize accuracy and extract evidence directly from the text."""

# --- 3. Helper Function Definitions ---

def filter_prompt_by_section(initial_full_prompt, section):
    """Filters the full prompt to include only questions for the specified section."""
    if section not in ALL_SECTIONS: raise ValueError(f"Invalid section specified: {section}")
    start_q, end_q = ALL_SECTIONS[section]; questions_start_marker = "**Questions to Answer:**"; questions_end_marker = "**Final Instruction:**"
    try: start_index = initial_full_prompt.index(questions_start_marker); end_index = initial_full_prompt.index(questions_end_marker)
    except ValueError: raise ValueError(f"Could not find '{questions_start_marker}' or '{questions_end_marker}' markers in the prompt file.")
    prompt_header = initial_full_prompt[:start_index]; full_questions_block = initial_full_prompt[start_index + len(questions_start_marker):end_index].strip(); prompt_footer = initial_full_prompt[end_index:]
    question_entries = re.split(r'\n\s*(?=\d+\.\s*?\*\*Question Category:)', full_questions_block)
    filtered_question_texts = []
    for entry in question_entries:
        entry = entry.strip()
        if not entry: continue
        # More robust regex for question number extraction
        match = re.match(r'^\s*(\d+)\.', entry)
        if match:
            # This try-except block looks syntactically correct
            try:
                q_num = int(match.group(1))
                # Check if the question number is within the desired range for the section
                if start_q <= q_num <= end_q:
                    filtered_question_texts.append(entry)
            except ValueError:
                # If int() fails (e.g., regex matched something unexpected),
                # simply skip this entry and continue to the next iteration
                # of the 'for' loop.
                continue
    if not filtered_question_texts: raise ValueError(f"No questions found for section '{section}' (range {start_q}-{end_q}) in the parsed questions block. Check prompt formatting and section ranges in ALL_SECTIONS.")
    filtered_questions_string = "\n\n".join(filtered_question_texts)
    section_note = f"\n\n**Current Focus:** You MUST answer ONLY the questions listed below for the '{section.upper()}' section (Questions {start_q}-{end_q}). Ignore all other questions for this specific task.\n"
    final_prompt_for_api = f"{prompt_header}{section_note}{questions_start_marker}\n\n{filtered_questions_string}\n\n{prompt_footer}"
    return final_prompt_for_api

def validate_ai_data(data, section_name):
    """Validates AI response against the schema. Returns (validated_data, issues_list)."""
    if not isinstance(data, list): return None, [f"CRITICAL VALIDATION ERROR: Response for section '{section_name}' is not a list."]
    validated_data = []; issues_list = []
    if section_name not in ALL_SECTIONS: issues_list.append(f"Warning: Section '{section_name}' not found in master section definitions (ALL_SECTIONS). Cannot validate question range."); expected_q_nums_in_section = set()
    else: expected_q_nums_in_section = set(range(ALL_SECTIONS[section_name][0], ALL_SECTIONS[section_name][1] + 1))
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

def generate_section_analysis(section, uploaded_file_ref, status_placeholder, api_key_to_use):
    """Generates analysis for a section using a specific API key."""
    try: genai.configure(api_key=api_key_to_use)
    except Exception as config_err: status_placeholder.error(f"❌ Invalid API Key provided or configuration failed: {config_err}"); return None, "Failed", [f"Invalid API Key or config error: {config_err}"]
    status_placeholder.info(f"🔄 Starting Analysis: {section}..."); section_warnings = []
    try:
        model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction_text)
        generation_config = types.GenerationConfig(response_mime_type="application/json", response_schema=ai_response_schema_dict, temperature=0.0,)
        safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        final_validated_data = None
        for attempt in range(1, MAX_VALIDATION_RETRIES + 2):
            if attempt > 1: status_placeholder.info(f"⏳ Retrying generation/validation for '{section}' (Attempt {attempt}/{MAX_VALIDATION_RETRIES+1})..."); time.sleep(RETRY_DELAY_SECONDS)
            try:
                prompt_for_api = filter_prompt_by_section(full_prompt_text, section)
                if not uploaded_file_ref or not hasattr(uploaded_file_ref, 'name'): raise ValueError("Invalid or missing uploaded file reference for GenAI call.")
                contents = [uploaded_file_ref, prompt_for_api]
                status_placeholder.info(f"🧠 Calling AI for '{section}' (Attempt {attempt})...")
                response = model.generate_content(contents=contents, generation_config=generation_config, safety_settings=safety_settings, request_options={'timeout': 600})
                parsed_ai_data = None; validated_ai_data = None; validation_issues = []
                status_placeholder.info(f"🔍 Processing response for '{section}'...")
                if response.parts:
                    full_response_text = response.text
                    try:
                        match = re.search(r"```json\s*([\s\S]*?)\s*```", full_response_text, re.IGNORECASE); json_text = match.group(1).strip() if match else full_response_text.strip()
                        if not json_text: raise json.JSONDecodeError("Extracted JSON content is empty.", json_text, 0)
                        parsed_ai_data = json.loads(json_text)
                        status_placeholder.info(f"✔️ Validating structure for '{section}'...")
                        validated_ai_data, validation_issues = validate_ai_data(parsed_ai_data, section); section_warnings.extend(validation_issues)
                        if validated_ai_data is not None and len(validated_ai_data) > 0: final_validated_data = validated_ai_data; status_placeholder.info(f"✅ Validation successful for '{section}'."); break
                        elif validated_ai_data is not None and len(validated_ai_data) == 0 and not validation_issues: status_placeholder.warning(f"⚠️ AI returned an empty list for '{section}' (Attempt {attempt}). Check prompt/section definition."); section_warnings.append(f"AI returned empty list for '{section}'.")
                        else: error_msg = f"Validation failed for '{section}'. Issues: {validation_issues}"; status_placeholder.warning(f"⚠️ {error_msg} (Attempt {attempt}).");
                        if validated_ai_data is None: section_warnings.append(f"CRITICAL validation error for '{section}'.")
                    except json.JSONDecodeError as json_err: error_msg = f"JSON Decode Error on attempt {attempt} for '{section}': {json_err}."; st.error(error_msg); st.code(full_response_text, language='text'); section_warnings.append(error_msg)
                    except Exception as parse_validate_err: error_msg = f"Unexpected Error during parsing/validation on attempt {attempt} for '{section}': {type(parse_validate_err).__name__}: {parse_validate_err}"; st.error(error_msg); section_warnings.append(error_msg); print(traceback.format_exc())
                else:
                    block_reason = "Unknown"; block_message = "N/A"; finish_reason = "Unknown"
                    try:
                        if response.prompt_feedback: block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown'); block_reason = block_reason.name if hasattr(block_reason, 'name') else str(block_reason); block_message = response.prompt_feedback.block_reason_message or "N/A"
                        if response.candidates: finish_reason = getattr(response.candidates[0], 'finish_reason', 'Unknown'); finish_reason = finish_reason.name if hasattr(finish_reason, 'name') else str(finish_reason)
                    except AttributeError: pass
                    if finish_reason == "SAFETY": warn_msg = f"API Response Blocked (Attempt {attempt}, Section: {section}): Reason: SAFETY. Detail: {block_reason}. Message: {block_message}"; st.error(warn_msg)
                    elif finish_reason == "RECITATION": warn_msg = f"API Response Potentially Blocked (Attempt {attempt}, Section: {section}): Finish Reason: RECITATION. Block Reason: {block_reason}."; st.warning(warn_msg)
                    elif finish_reason == "STOP" and not final_validated_data: warn_msg = f"API Response Ended (Attempt {attempt}, Section: {section}): Finish Reason: STOP, but no valid data parsed yet."; st.info(warn_msg)
                    else: warn_msg = f"API Issue (Attempt {attempt}, Section: {section}): Finish Reason: {finish_reason}. Block Reason: {block_reason}. Response may be incomplete or empty."; st.warning(warn_msg)
                    section_warnings.append(warn_msg)
            except types.StopCandidateException as sce: error_msg = f"Generation Stopped Error (Attempt {attempt}, Section: {section}): {sce}."; st.error(error_msg); section_warnings.append(error_msg); print(traceback.format_exc())
            except Exception as e: error_msg = f"Processing Error during API call/prompt generation (Attempt {attempt}, Section: {section}): {type(e).__name__}: {e}"; st.error(error_msg); section_warnings.append(error_msg); section_warnings.append("Traceback logged to console."); print(traceback.format_exc())
            if final_validated_data is not None: break
        if final_validated_data is not None: status_placeholder.success(f"✅ Analysis completed successfully for: {section}."); return final_validated_data, "Success", section_warnings
        else: status_placeholder.error(f"❌ Analysis failed for: {section} after {attempt} attempts."); section_warnings.append(f"Failed to get valid response for section '{section}' after {MAX_VALIDATION_RETRIES + 1} attempts."); return None, "Failed", section_warnings
    except Exception as outer_err: error_msg = f"Critical Error during setup or execution for section '{section}': {type(outer_err).__name__}: {outer_err}"; st.error(error_msg); section_warnings.append(error_msg); section_warnings.append("Traceback logged to console."); print(traceback.format_exc()); status_placeholder.error(f"❌ Critical failure processing section: {section}."); return None, "Failed", section_warnings

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

# REMOVED @st.cache_data from this function due to UnhashableParamError with fitz.Rect
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

# --- 4. Initialize Session State ---
if 'show_wording_states' not in st.session_state: st.session_state.show_wording_states = defaultdict(bool)
if 'current_page' not in st.session_state: st.session_state.current_page = 1
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None
if 'pdf_display_ready' not in st.session_state: st.session_state.pdf_display_ready = False
if 'processing_in_progress' not in st.session_state: st.session_state.processing_in_progress = False
if 'analysis_complete' not in st.session_state: st.session_state.analysis_complete = False
if 'run_key' not in st.session_state: st.session_state.run_key = 0
if 'run_status_summary' not in st.session_state: st.session_state.run_status_summary = []
if 'excel_data' not in st.session_state: st.session_state.excel_data = None
if 'search_trigger' not in st.session_state: st.session_state.search_trigger = None
if 'last_search_result' not in st.session_state: st.session_state.last_search_result = None
if 'api_key' not in st.session_state: st.session_state.api_key = None
if 'selected_sections_to_run' not in st.session_state: st.session_state.selected_sections_to_run = list(ALL_SECTIONS.keys())[:1] # Default to only first section for testing

# --- 5. Streamlit UI Logic ---
st.title("JASPER - Just A Smart Platform for Extraction and Review")
st.markdown("Upload a PDF agreement, **enter your Gemini API Key**, **select sections**, click 'Analyse'. Results grouped below. Click clause references to view & highlight.")

# --- Sidebar Setup ---
st.sidebar.markdown("## Controls")

# --- API Key Input (in Sidebar) ---
api_key_input = st.sidebar.text_input("Enter your Google AI Gemini API Key", type="password", key="api_key_input", help="Your API key is used only for this session and is not stored long-term.", value=st.session_state.get("api_key", ""))
if api_key_input != st.session_state.api_key: st.session_state.api_key = api_key_input; st.rerun()
if not st.session_state.api_key and not st.session_state.analysis_complete and st.session_state.pdf_bytes: st.sidebar.warning("API Key required to run analysis.", icon="🔑")

# --- File Upload (in Sidebar) ---
uploaded_file_obj = st.sidebar.file_uploader("Upload Facility Agreement PDF", type="pdf", key=f"pdf_uploader_{st.session_state.run_key}")
if uploaded_file_obj is not None:
    new_file_bytes = uploaded_file_obj.getvalue()
    if new_file_bytes != st.session_state.get('pdf_bytes'):
        st.session_state.pdf_bytes = new_file_bytes; st.session_state.pdf_display_ready = True
        st.session_state.analysis_results = None; st.session_state.analysis_complete = False; st.session_state.processing_in_progress = False; st.session_state.current_page = 1
        st.session_state.run_status_summary = []; st.session_state.excel_data = None; st.session_state.search_trigger = None; st.session_state.last_search_result = None
        st.session_state.show_wording_states = defaultdict(bool)
        st.toast("✅ New PDF file loaded. Viewer ready.", icon="📄"); st.rerun()

# --- Section Selection for Analysis (NEW) ---
st.sidebar.markdown("### Select Sections to Analyse")
selected_sections = st.sidebar.multiselect("Choose sections:", options=list(ALL_SECTIONS.keys()), default=st.session_state.selected_sections_to_run, key="section_selector", help="Select which parts of the document you want to analyse in this run.")
if selected_sections != st.session_state.selected_sections_to_run: st.session_state.selected_sections_to_run = selected_sections; st.rerun()

# --- Analysis Trigger (in Sidebar) ---
can_analyse = (st.session_state.pdf_bytes is not None and st.session_state.api_key is not None and not st.session_state.processing_in_progress and not st.session_state.analysis_complete and st.session_state.selected_sections_to_run)
analyse_button_tooltip = "Analysis complete for the current file." if st.session_state.analysis_complete else "Analysis is currently running." if st.session_state.processing_in_progress else "Upload a PDF file first." if not st.session_state.pdf_bytes else "Enter your Gemini API key first." if not st.session_state.api_key else "Select at least one section to analyse." if not st.session_state.selected_sections_to_run else "Start analyzing the selected sections"
if st.sidebar.button("✨ Analyse Document", key="analyse_button", disabled=not can_analyse, help=analyse_button_tooltip, use_container_width=True, type="primary"):
    if not st.session_state.api_key: st.error("API Key is missing. Please enter it in the sidebar.")
    elif not st.session_state.pdf_bytes: st.error("No PDF file uploaded. Please upload a file.")
    elif not st.session_state.selected_sections_to_run: st.error("No sections selected for analysis. Please select sections in the sidebar.")
    else:
        st.session_state.processing_in_progress = True; st.session_state.analysis_complete = False; st.session_state.analysis_results = []
        st.session_state.run_key += 1; st.session_state.run_status_summary = []; st.session_state.excel_data = None; st.session_state.search_trigger = None
        st.session_state.last_search_result = None; st.session_state.show_wording_states = defaultdict(bool)
        current_api_key = st.session_state.api_key; run_start_time = datetime.now(); run_timestamp_str = run_start_time.strftime("%Y-%m-%d %H:%M:%S")
        base_file_name = getattr(uploaded_file_obj, 'name', f"upload_{run_start_time.strftime('%Y%m%d%H%M%S')}.pdf")
        try: genai.configure(api_key=current_api_key); st.toast("API Key validated and configured.", icon="🔑")
        except Exception as config_err: st.error(f"❌ Failed to configure Gemini API with provided key: {config_err}"); st.session_state.processing_in_progress = False; st.stop()
        status_container = st.container(); progress_bar = status_container.progress(0, text="Initializing analysis..."); status_text = status_container.empty()
        temp_dir = "temp_uploads"; safe_base_name = re.sub(r'[^\w\-.]', '_', base_file_name); temp_file_path = os.path.join(APP_DIR, temp_dir, f"{run_start_time.strftime('%Y%m%d%H%M%S')}_{safe_base_name}")
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        gemini_uploaded_file_ref = None; all_validated_data = []; overall_success = True
        try:
            status_text.info("💾 Saving file temporarily..."); progress_bar.progress(5, text="Saving temp file...");
            with open(temp_file_path, "wb") as f: f.write(st.session_state.pdf_bytes)
            status_text.info("☁️ Uploading file to Google Cloud AI..."); progress_bar.progress(10, text="Uploading to cloud...")
            for upload_attempt in range(3):
                try: gemini_uploaded_file_ref = genai.upload_file(path=temp_file_path); st.toast(f"File '{gemini_uploaded_file_ref.display_name}' uploaded to cloud.", icon="☁️"); break
                except Exception as upload_err:
                    err_str = str(upload_err).lower()
                    if "api key" in err_str or "authenticat" in err_str or "permission" in err_str: st.error(f"❌ File upload failed due to API key/permission issue: {upload_err}"); st.error("Please verify the API key has File API permissions enabled."); raise
                    elif upload_attempt < 2: st.warning(f"Upload attempt {upload_attempt+1} failed: {upload_err}. Retrying..."); time.sleep(2 + upload_attempt)
                    else: st.error(f"Upload failed after multiple attempts: {upload_err}"); raise
            if not gemini_uploaded_file_ref: raise Exception("Failed to upload file to Google Cloud AI after retries.")
            progress_bar.progress(15, text="File uploaded. Starting section analysis...")
            sections_to_process = st.session_state.selected_sections_to_run; num_sections = len(sections_to_process); progress_per_section = (95 - 15) / num_sections if num_sections > 0 else 0
            for i, section_name in enumerate(sections_to_process):
                if section_name not in ALL_SECTIONS: st.warning(f"Skipping invalid section '{section_name}' found in selection."); continue
                current_progress = int(15 + (i * progress_per_section)); progress_bar.progress(current_progress, text=f"Analysing Section: {section_name}...")
                section_data, section_status, section_warnings = generate_section_analysis(section_name, gemini_uploaded_file_ref, status_text, current_api_key)
                st.session_state.run_status_summary.append({"section": section_name, "status": section_status, "warnings": section_warnings})
                if section_status == "Success" and section_data:
                    for item in section_data: item["File Name"] = base_file_name; item["Generation Time"] = run_timestamp_str
                    all_validated_data.extend(section_data)
                else: overall_success = False
                progress_bar.progress(int(15 + ((i + 1) * progress_per_section)), text=f"Completed: {section_name}")
            st.session_state.analysis_results = all_validated_data; progress_bar.progress(100, text="Analysis process finished!")
            if overall_success: status_text.success("🏁 Analysis finished successfully!")
            else: status_text.warning("🏁 Analysis finished, but some sections encountered issues (see summary below).")
            st.session_state.analysis_complete = True
        except Exception as main_err:
            st.error(f"❌ CRITICAL ERROR during analysis workflow: {main_err}"); print(traceback.format_exc())
            overall_success = False; st.session_state.analysis_complete = False
            st.session_state.run_status_summary.append({"section": "Overall Process Control", "status": "Critical Error", "warnings": [str(main_err), "Analysis halted. See logs."]})
            status_text.error(f"Analysis stopped due to critical error: {main_err}")
        finally:
            st.session_state.processing_in_progress = False; time.sleep(4); status_text.empty(); progress_bar.empty()
            if gemini_uploaded_file_ref and hasattr(gemini_uploaded_file_ref, 'name'):
                try: status_text.info(f"☁️ Deleting temporary cloud file: {gemini_uploaded_file_ref.name}..."); genai.delete_file(name=gemini_uploaded_file_ref.name); st.toast("Cloud file deleted.", icon="🗑️"); time.sleep(1); status_text.empty()
                except Exception as del_err: st.sidebar.warning(f"Cloud cleanup issue: {del_err}", icon="⚠️"); status_text.warning(f"Could not delete cloud file: {del_err}"); print(f"WARN: Failed to delete cloud file {gemini_uploaded_file_ref.name}: {del_err}")
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except Exception as local_del_err: st.sidebar.warning(f"Local temp file cleanup issue: {local_del_err}", icon="⚠️"); print(f"WARN: Failed to delete local temp file {temp_file_path}: {local_del_err}")
        st.rerun()

# --- Display Area (Results and PDF Viewer) ---
if st.session_state.pdf_bytes is not None:
    col1, col2 = st.columns([6, 4]) # Results | PDF Viewer

    # --- Column 1: Analysis Results ---
    with col1:
        # --- Run Status Summary ---
        if st.session_state.run_status_summary:
            final_status = "✅ Success"; has_failures = any(s['status'] == "Failed" or "Error" in s['status'] for s in st.session_state.run_status_summary); has_warnings = any(s['status'] != "Success" and not has_failures for s in st.session_state.run_status_summary)
            # Corrected if/elif logic
            if has_failures:
                final_status = "❌ Failed"
            elif has_warnings:
                final_status = "⚠️ Issues"
            with st.expander(f"📊 Analysis Run Summary ({final_status})", expanded=(final_status != "✅ Success")):
                for item in st.session_state.run_status_summary:
                    icon = "✅" if item['status'] == "Success" else "❌" if item['status'] == "Failed" or "Error" in item['status'] else "⚠️"
                    st.markdown(f"**{item['section']}**: {icon} {item['status']}")
                    if item['warnings']:
                        filtered_warnings = [msg for msg in item['warnings'] if not (isinstance(msg, str) and msg.startswith("Validation Issues Found"))]
                        if filtered_warnings:
                            with st.container(): st.caption("Details:")
                            for msg in filtered_warnings:
                                msg_str = str(msg);
                                if "CRITICAL" in msg_str or "Error" in msg_str or "Block" in msg_str or "Fail" in msg_str: st.error(f" L> {msg_str}")
                                elif "Warn" in msg_str or "Missing" in msg_str or "Unexpected" in msg_str or "Empty list" in msg_str or "validation issues" in msg_str.lower(): st.warning(f" L> {msg_str}")
                                else: st.caption(f" L> {msg_str}")

        st.subheader("Analysis Results")

        # --- Display Results (if analysis complete and results exist) ---
        if st.session_state.analysis_complete and st.session_state.analysis_results:
            results_list = st.session_state.analysis_results

            # --- NEW: Scatter Plot Expander ---
            try:
                plot_data = []
                for item in results_list:
                    plot_data.append({
                        'Question Number': item.get('Question Number', 0), # Default to 0 if missing
                        'Number of Evidence Items': len(item.get('Evidence', [])),
                        'Question Category': item.get('Question Category', 'Uncategorized'),
                        'Question': item.get('Question', 'N/A') # For hover
                    })

                if plot_data:
                    df_plot = pd.DataFrame(plot_data)
                    with st.expander("📊 Evidence Count Analysis (Scatter Plot)", expanded=False):
                        fig = px.scatter(
                            df_plot,
                            x='Question Number',
                            y='Number of Evidence Items',
                            color='Question Category',
                            title="Number of Evidence Clauses Found per Question",
                            labels={'Number of Evidence Items': 'Evidence Count'},
                            hover_data=['Question'] # Show question text on hover
                        )
                        fig.update_traces(marker=dict(size=10, opacity=0.7), selector=dict(mode='markers'))
                        fig.update_layout(xaxis_title="Question Number", yaxis_title="Number of Evidence Clauses")
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("This plot shows how many separate evidence clauses the AI referenced for each question. Hover over points for question details.")
                # else: # Optional: message if no data for plot
                    # st.info("No data available to generate the evidence count plot.")

            except Exception as plot_err:
                 st.warning(f"Could not generate scatter plot: {plot_err}")
                 print(f"Plotting Error: {plot_err}\n{traceback.format_exc()}")
            # --- END: Scatter Plot Expander ---


            # --- Tabbed Results Display ---
            grouped_results = defaultdict(list); categories_ordered = []
            for item in results_list:
                category = item.get("Question Category", "Uncategorized")
                if category not in grouped_results: categories_ordered.append(category)
                grouped_results[category].append(item)

            if categories_ordered:
                category_tabs = st.tabs(categories_ordered)
                for i, category in enumerate(categories_ordered):
                    with category_tabs[i]:
                        category_items = sorted(grouped_results[category], key=lambda x: x.get('Question Number', float('inf')))
                        for index, result_item in enumerate(category_items):
                            q_num = result_item.get('Question Number', 'N/A'); question_text = result_item.get('Question', 'N/A')
                            with st.expander(f"**Q{q_num}:** {question_text}"): # Removed key=...
                                st.markdown(f"**Answer:**"); st.markdown(f"> {result_item.get('Answer', 'N/A')}"); st.markdown("---")
                                evidence_list = result_item.get('Evidence', [])
                                if evidence_list:
                                    st.markdown("**Evidence:**")
                                    for ev_index, evidence_item in enumerate(evidence_list):
                                        clause_ref = evidence_item.get('Clause Reference', 'N/A'); search_text = evidence_item.get('Searchable Clause Text', None)
                                        clause_wording = evidence_item.get('Clause Wording', 'N/A'); base_key = f"ev_{category}_{q_num}_{index}_{ev_index}"
                                        ev_cols = st.columns([3, 1])
                                        with ev_cols[0]:
                                            if search_text:
                                                button_key = f"search_btn_{base_key}"; button_label = f"Clause: **{clause_ref or 'Link'}** (Find & View)"
                                                if st.button(button_label, key=button_key, help=f"Search for text related to '{clause_ref or 'this evidence'}' and view the page."):
                                                    st.session_state.search_trigger = {'text': search_text, 'ref': clause_ref}; st.session_state.last_search_result = None; st.rerun()
                                            elif clause_ref != 'N/A': st.markdown(f"- Clause: **{clause_ref}** (No searchable text provided by AI)")
                                            else: st.caption("No clause reference provided.")
                                        with ev_cols[1]:
                                            if clause_wording != 'N/A':
                                                toggle_key = f"toggle_wording_{base_key}"; show_wording = st.toggle("Show Wording", key=toggle_key, value=st.session_state.show_wording_states[toggle_key], help="Show/hide the exact clause wording extracted by the AI.")
                                                if show_wording != st.session_state.show_wording_states[toggle_key]: st.session_state.show_wording_states[toggle_key] = show_wording; st.rerun()
                                        if st.session_state.show_wording_states.get(f"toggle_wording_{base_key}", False): st.text_area(f"AI Extracted Wording for '{clause_ref}':", value=clause_wording, height=150, disabled=True, key=f"wording_area_{base_key}")
                                        st.markdown("---")
                                else: st.markdown("**Evidence:** None provided.")
                                st.markdown("---"); st.markdown("**Answer Justification:**")
                                justification_text = result_item.get('Answer Justification', ''); just_key = f"justification_{category}_{q_num}_{index}"
                                st.text_area(label="Justification Text Area", value=justification_text, height=100, disabled=True, label_visibility="collapsed", key=just_key)
            else: st.warning("Analysis generated results, but they could not be grouped by category. Displaying raw list."); st.json(results_list)

            # --- Excel Download ---
            st.sidebar.markdown("---"); st.sidebar.markdown("## Export Results")
            if st.sidebar.button("Prepare Data for Excel Download", key="prep_excel", use_container_width=True):
                st.session_state.excel_data = None; excel_prep_status = st.sidebar.empty(); excel_prep_status.info("Preparing Excel data...")
                try:
                    excel_rows = [];
                    for item in results_list:
                        references = []; first_search_text = "N/A"; evidence = item.get("Evidence")
                        if evidence:
                            for i, ev in enumerate(evidence):
                                if isinstance(ev, dict): references.append(str(ev.get("Clause Reference", "N/A")));
                                if i == 0 and isinstance(ev, dict): first_search_text = ev.get("Searchable Clause Text", "N/A")
                                else: references.append("[Invalid Evidence Item]") # Should not happen with valid AI response
                        excel_row = {"File Name": item.get("File Name", ""), "Generation Time": item.get("Generation Time", ""), "Question Number": item.get("Question Number"), "Question Category": item.get("Question Category", "Uncategorized"), "Question": item.get("Question", "N/A"), "Answer": item.get("Answer", "N/A"), "Answer Justification": item.get("Answer Justification", "N/A"), "Clause References (Concatenated)": "; ".join(references) if references else "N/A", "First Searchable Clause Text": first_search_text}
                        excel_rows.append(excel_row)
                    if not excel_rows: excel_prep_status.warning("No data available to export."); st.session_state.excel_data = None
                    else:
                        df_excel = pd.DataFrame(excel_rows); final_columns = [col for col in EXCEL_COLUMN_ORDER if col in df_excel.columns]; extra_cols = [col for col in df_excel.columns if col not in final_columns]; df_excel = df_excel[final_columns + extra_cols]
                        output = io.BytesIO();
                        with pd.ExcelWriter(output, engine='openpyxl') as writer: df_excel.to_excel(writer, index=False, sheet_name='Analysis Results')
                        st.session_state.excel_data = output.getvalue(); excel_prep_status.success("✅ Excel file ready for download!"); time.sleep(2); excel_prep_status.empty()
                except Exception as excel_err: excel_prep_status.error(f"Excel Prep Error: {excel_err}"); print(traceback.format_exc())

            if st.session_state.excel_data:
                current_filename = "analysis_results"
                if st.session_state.analysis_results and isinstance(st.session_state.analysis_results, list) and len(st.session_state.analysis_results) > 0:
                     first_result = st.session_state.analysis_results[0]
                     if isinstance(first_result, dict) and "File Name" in first_result: current_filename = first_result["File Name"]
                safe_base_name = re.sub(r'[^\w\s-]', '', os.path.splitext(current_filename)[0]).strip().replace(' ', '_'); download_filename = f"Analysis_{safe_base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                st.sidebar.download_button(label="📥 Download Results as Excel", data=st.session_state.excel_data, file_name=download_filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_excel_final", use_container_width=True)

        # --- Fallback messages within Col 1 ---
        elif st.session_state.analysis_complete and not st.session_state.analysis_results: st.info("Analysis process completed, but no valid results were generated. Check the run summary above for potential issues.")
        elif st.session_state.processing_in_progress: st.info("Analysis is currently in progress...")
        elif not st.session_state.analysis_complete and st.session_state.pdf_bytes is not None: st.info("PDF loaded. Select sections and click 'Analyse Document' in the sidebar to start.")

    # --- Column 2: PDF Viewer (Sticky) ---
    with col2:
        st.markdown('<div class="sticky-viewer-content">', unsafe_allow_html=True) # Sticky wrapper start
        st.subheader("📄 PDF Viewer"); viewer_status_placeholder = st.empty()

        if st.session_state.search_trigger:
            search_info = st.session_state.search_trigger; st.session_state.search_trigger = None
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
            if current_display_page != st.session_state.get('current_page'): st.session_state.current_page = current_display_page

            nav_cols = st.columns([1, 3, 1])
            with nav_cols[0]:
                if st.button("⬅️ Prev", key="prev_page", disabled=(current_display_page <= 1), use_container_width=True): st.session_state.current_page -= 1; st.session_state.last_search_result = None; st.rerun()
            with nav_cols[2]:
                if st.button("Next ➡️", key="next_page", disabled=(current_display_page >= total_pages), use_container_width=True): st.session_state.current_page += 1; st.session_state.last_search_result = None; st.rerun()

            page_info_text = f"Page {current_display_page} of {total_pages}"; search_context_ref = None
            if st.session_state.last_search_result and st.session_state.last_search_result.get('all_findings'): search_context_ref = st.session_state.last_search_result['ref']; page_info_text += f" (🎯 Multi-match: '{search_context_ref}')"
            elif st.session_state.last_search_result and st.session_state.last_search_result['page'] == current_display_page: search_context_ref = st.session_state.last_search_result['ref']; page_info_text += f" (🎯 Ref: '{search_context_ref}')"
            nav_cols[1].markdown(f"<div style='text-align: center; padding-top: 0.5rem;'>{page_info_text}</div>", unsafe_allow_html=True)

            if st.session_state.last_search_result and st.session_state.last_search_result.get('all_findings'):
                multi_findings = st.session_state.last_search_result['all_findings']; found_pages = sorted([f[0] for f in multi_findings])
                status_msg = st.session_state.last_search_result['status']; viewer_status_placeholder.info(status_msg); st.write("Jump to other matches for this reference:")
                num_buttons = len(found_pages); btn_cols = st.columns(min(num_buttons, 5)) # Limit columns to avoid crowding
                for idx, p_num in enumerate(found_pages):
                    col_idx = idx % len(btn_cols) # Distribute buttons across columns
                    is_current = (p_num == current_display_page)
                    if btn_cols[col_idx].button(f"Page {p_num}", key=f"jump_{p_num}_{search_context_ref}", disabled=is_current, use_container_width=True):
                        st.session_state.current_page = p_num; new_instances = next((inst for pg, inst in multi_findings if pg == p_num), None)
                        st.session_state.last_search_result['instances'] = new_instances; st.session_state.last_search_result['page'] = p_num; st.rerun()

            st.markdown("---")
            highlights_to_apply = None; render_status_override = None
            if st.session_state.last_search_result and st.session_state.last_search_result.get('page') == current_display_page:
                highlights_to_apply = st.session_state.last_search_result.get('instances')
                if not st.session_state.last_search_result.get('all_findings'): render_status_override = st.session_state.last_search_result.get('status')

            image_bytes, render_status = render_pdf_page_to_image(st.session_state.pdf_bytes, current_display_page, highlight_instances=highlights_to_apply, dpi=150)

            if image_bytes:
                st.image(image_bytes, caption=f"Page {current_display_page} - View", use_container_width=True) # Use container width
                final_status = render_status_override if render_status_override else render_status
                if not (st.session_state.last_search_result and st.session_state.last_search_result.get('all_findings')):
                    if final_status:
                        if "✅" in final_status or "✨" in final_status or "Found" in final_status : viewer_status_placeholder.success(final_status)
                        elif "⚠️" in final_status or "warning" in final_status.lower(): viewer_status_placeholder.warning(final_status)
                        elif "❌" in final_status or "error" in final_status.lower(): viewer_status_placeholder.error(final_status)
                        else: viewer_status_placeholder.caption(final_status)
                    else: viewer_status_placeholder.empty()
            else: viewer_status_placeholder.error(f"Failed to render page {current_display_page}. {render_status or ''}")
        else: st.info("PDF loaded, preparing viewer..."); viewer_status_placeholder.empty()
        st.markdown('</div>', unsafe_allow_html=True) # Sticky wrapper end

# --- Fallback messages if no PDF is loaded ---
elif not st.session_state.pdf_bytes:
     st.info("⬆️ Upload a PDF file using the sidebar to begin.")


# # app.py
# # --- COMPLETE FILE (v3 - Expander fix + Section Selection) ---

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

# # --- 2. Configuration & Setup ---
# MODEL_NAME = "gemini-2.5-pro-preview-03-25" 
# # MODEL_NAME = "gemini-1.5-flash-preview-0514" # Consider Flash for faster testing if Pro is slow
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
# # Keep your system instruction concise but effective
# system_instruction_text = """You are an AI assistant specialized in analyzing legal facility agreements. Carefully read the provided document and answer the specific questions listed in the user prompt for the designated section only. Adhere strictly to the requested JSON output schema. Prioritize accuracy and extract evidence directly from the text."""

# # --- 3. Helper Function Definitions ---

# def filter_prompt_by_section(initial_full_prompt, section):
#     """Filters the full prompt to include only questions for the specified section."""
#     if section not in ALL_SECTIONS: # Use master list for validation
#         raise ValueError(f"Invalid section specified: {section}")

#     start_q, end_q = ALL_SECTIONS[section] # Use master list to get ranges
#     questions_start_marker = "**Questions to Answer:**"
#     questions_end_marker = "**Final Instruction:**"

#     try:
#         start_index = initial_full_prompt.index(questions_start_marker)
#         end_index = initial_full_prompt.index(questions_end_marker)
#     except ValueError:
#         raise ValueError(f"Could not find '{questions_start_marker}' or '{questions_end_marker}' markers in the prompt file.")

#     prompt_header = initial_full_prompt[:start_index]
#     full_questions_block = initial_full_prompt[start_index + len(questions_start_marker):end_index].strip()
#     prompt_footer = initial_full_prompt[end_index:]

#     # Improved regex to handle potential leading whitespace and variations
#     # This regex should target lines starting with a number, dot, optional space, then **Question Category:**
#     question_entries = re.split(r'\n\s*(?=\d+\.\s*?\*\*Question Category:)', full_questions_block)

#     filtered_question_texts = []
#     for entry in question_entries:
#         entry = entry.strip()
#         if not entry: continue
#         # More robust regex for question number extraction
#         match = re.match(r'^\s*(\d+)\.', entry)
#         if match:
#             try:
#                 q_num = int(match.group(1))
#                 if start_q <= q_num <= end_q:
#                     # Re-add the leading number and formatting for the AI
#                     filtered_question_texts.append(entry)
#             except ValueError:
#                 continue # Skip if number conversion fails
#         # else: # Optional: Log entries that don't start with expected number format
#             # print(f"Debug: Skipping entry, could not parse question number: {entry[:60]}...")

#     if not filtered_question_texts:
#         # Provide more context in the error message
#         raise ValueError(f"No questions found for section '{section}' (range {start_q}-{end_q}) in the parsed questions block. Check prompt formatting and section ranges in ALL_SECTIONS.")

#     filtered_questions_string = "\n\n".join(filtered_question_texts)

#     # Add a clear note about the current focus
#     section_note = f"\n\n**Current Focus:** You MUST answer ONLY the questions listed below for the '{section.upper()}' section (Questions {start_q}-{end_q}). Ignore all other questions for this specific task.\n"

#     final_prompt_for_api = f"{prompt_header}{section_note}{questions_start_marker}\n\n{filtered_questions_string}\n\n{prompt_footer}"
#     return final_prompt_for_api


# def validate_ai_data(data, section_name):
#     """Validates AI response against the schema. Returns (validated_data, issues_list)."""
#     if not isinstance(data, list):
#         return None, [f"CRITICAL VALIDATION ERROR: Response for section '{section_name}' is not a list."]

#     validated_data = []
#     issues_list = []

#     # Determine expected question numbers using the master definition
#     if section_name not in ALL_SECTIONS:
#          issues_list.append(f"Warning: Section '{section_name}' not found in master section definitions (ALL_SECTIONS). Cannot validate question range.")
#          expected_q_nums_in_section = set() # Cannot determine expected range
#     else:
#         expected_q_nums_in_section = set(range(ALL_SECTIONS[section_name][0], ALL_SECTIONS[section_name][1] + 1))

#     found_q_nums = set()

#     for index, item in enumerate(data):
#         q_num = item.get('Question Number')
#         q_num_str = f"Q#{q_num}" if q_num is not None else f"Item Index {index}"
#         is_outer_valid = True

#         if not isinstance(item, dict):
#             issues_list.append(f"{q_num_str}: Item is not a dictionary.")
#             is_outer_valid = False
#             continue

#         if isinstance(q_num, int):
#             found_q_nums.add(q_num)
#         else:
#              issues_list.append(f"Item Index {index}: 'Question Number' is missing or not an integer.")
#              is_outer_valid = False

#         missing_outer_keys = AI_REQUIRED_KEYS - set(item.keys())
#         if missing_outer_keys:
#             issues_list.append(f"{q_num_str}: Missing required top-level keys: {missing_outer_keys}")
#             is_outer_valid = False

#         evidence_list = item.get("Evidence")
#         if not isinstance(evidence_list, list):
#             if "Evidence" in AI_REQUIRED_KEYS:
#                  issues_list.append(f"{q_num_str}: 'Evidence' field is not a list (found type: {type(evidence_list).__name__}).")
#                  is_outer_valid = False
#         else:
#             for ev_index, ev_item in enumerate(evidence_list):
#                 ev_id_str = f"Ev[{ev_index}]"
#                 if not isinstance(ev_item, dict):
#                     issues_list.append(f"{q_num_str} {ev_id_str}: Evidence item is not a dictionary.")
#                     is_outer_valid = False
#                     continue

#                 missing_ev_keys = AI_EVIDENCE_REQUIRED_KEYS - set(ev_item.keys())
#                 if missing_ev_keys:
#                     issues_list.append(f"{q_num_str} {ev_id_str}: Missing required evidence keys: {missing_ev_keys}")
#                     is_outer_valid = False

#                 for key, expected_type in [("Clause Reference", str), ("Clause Wording", str), ("Searchable Clause Text", str)]:
#                      if key in AI_EVIDENCE_REQUIRED_KEYS and key in ev_item and not isinstance(ev_item.get(key), expected_type):
#                          issues_list.append(f"{q_num_str} {ev_id_str}: Key '{key}' has incorrect type (expected {expected_type.__name__}, got {type(ev_item.get(key)).__name__}).")
#                          is_outer_valid = False

#         if is_outer_valid:
#             validated_data.append(item)

#     # --- Perform checks based on expected vs found question numbers ---
#     if expected_q_nums_in_section: # Only if we could determine the expected range
#         missing_q_nums = expected_q_nums_in_section - found_q_nums
#         if missing_q_nums:
#             issues_list.append(f"Section '{section_name}': Missing answers for expected Question Numbers: {sorted(list(missing_q_nums))}")

#         unexpected_q_nums = found_q_nums - expected_q_nums_in_section
#         if unexpected_q_nums:
#             issues_list.append(f"Section '{section_name}': Received unexpected Question Numbers: {sorted(list(unexpected_q_nums))}")
#     # --- End checks ---

#     if issues_list:
#         issues_list.insert(0, f"Validation Issues Found [Section: {section_name}] ({len(validated_data)} items passed validation):")

#     if validated_data is None and isinstance(data, list):
#          return [], issues_list
#     elif validated_data is None:
#         return None, issues_list
#     else:
#         return validated_data, issues_list


# def generate_section_analysis(section, uploaded_file_ref, status_placeholder, api_key_to_use):
#     """Generates analysis for a section using a specific API key."""
#     # Configure GenAI client INSIDE the function for this specific call
#     try:
#         genai.configure(api_key=api_key_to_use)
#     except Exception as config_err:
#         status_placeholder.error(f"❌ Invalid API Key provided or configuration failed: {config_err}")
#         return None, "Failed", [f"Invalid API Key or config error: {config_err}"]

#     status_placeholder.info(f"🔄 Starting Analysis: {section}...")
#     section_warnings = []

#     try:
#         model = genai.GenerativeModel(
#             model_name=MODEL_NAME,
#             system_instruction=system_instruction_text
#         )
#         generation_config = types.GenerationConfig(
#             response_mime_type="application/json",
#             response_schema=ai_response_schema_dict,
#             temperature=0.0,
#         )
#         safety_settings = [
#             {"category": c, "threshold": "BLOCK_NONE"}
#             for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
#         ]

#         final_validated_data = None

#         for attempt in range(1, MAX_VALIDATION_RETRIES + 2):
#             if attempt > 1:
#                 status_placeholder.info(f"⏳ Retrying generation/validation for '{section}' (Attempt {attempt}/{MAX_VALIDATION_RETRIES+1})...")
#                 time.sleep(RETRY_DELAY_SECONDS)

#             try:
#                 # Filter prompt correctly using the master list (ALL_SECTIONS)
#                 prompt_for_api = filter_prompt_by_section(full_prompt_text, section)
#                 if not uploaded_file_ref or not hasattr(uploaded_file_ref, 'name'):
#                      raise ValueError("Invalid or missing uploaded file reference for GenAI call.")

#                 contents = [uploaded_file_ref, prompt_for_api]

#                 status_placeholder.info(f"🧠 Calling AI for '{section}' (Attempt {attempt})...")
#                 response = model.generate_content(
#                     contents=contents,
#                     generation_config=generation_config,
#                     safety_settings=safety_settings,
#                     request_options={'timeout': 600}
#                 )

#                 parsed_ai_data = None
#                 validated_ai_data = None
#                 validation_issues = []

#                 status_placeholder.info(f"🔍 Processing response for '{section}'...")

#                 if response.parts:
#                     full_response_text = response.text
#                     try:
#                         match = re.search(r"```json\s*([\s\S]*?)\s*```", full_response_text, re.IGNORECASE)
#                         json_text = match.group(1).strip() if match else full_response_text.strip()
#                         if not json_text:
#                              raise json.JSONDecodeError("Extracted JSON content is empty.", json_text, 0)

#                         parsed_ai_data = json.loads(json_text)
#                         status_placeholder.info(f"✔️ Validating structure for '{section}'...")
#                         # Validate data using the master list (ALL_SECTIONS) for ranges
#                         validated_ai_data, validation_issues = validate_ai_data(parsed_ai_data, section)
#                         section_warnings.extend(validation_issues)

#                         if validated_ai_data is not None and len(validated_ai_data) > 0:
#                             final_validated_data = validated_ai_data
#                             status_placeholder.info(f"✅ Validation successful for '{section}'.")
#                             break
#                         elif validated_ai_data is not None and len(validated_ai_data) == 0 and not validation_issues:
#                              status_placeholder.warning(f"⚠️ AI returned an empty list for '{section}' (Attempt {attempt}). Check prompt/section definition.")
#                              section_warnings.append(f"AI returned empty list for '{section}'.")
#                         else:
#                             error_msg = f"Validation failed for '{section}'. Issues: {validation_issues}"
#                             status_placeholder.warning(f"⚠️ {error_msg} (Attempt {attempt}).")
#                             if validated_ai_data is None:
#                                 section_warnings.append(f"CRITICAL validation error for '{section}'.")

#                     except json.JSONDecodeError as json_err:
#                         error_msg = f"JSON Decode Error on attempt {attempt} for '{section}': {json_err}."
#                         st.error(error_msg)
#                         st.code(full_response_text, language='text')
#                         section_warnings.append(error_msg)
#                     except Exception as parse_validate_err:
#                          error_msg = f"Unexpected Error during parsing/validation on attempt {attempt} for '{section}': {type(parse_validate_err).__name__}: {parse_validate_err}"
#                          st.error(error_msg)
#                          section_warnings.append(error_msg)
#                          print(traceback.format_exc())

#                 else:
#                     block_reason = "Unknown"; block_message = "N/A"; finish_reason = "Unknown"
#                     try:
#                         if response.prompt_feedback:
#                             block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown')
#                             block_reason = block_reason.name if hasattr(block_reason, 'name') else str(block_reason)
#                             block_message = response.prompt_feedback.block_reason_message or "N/A"
#                         if response.candidates:
#                             finish_reason = getattr(response.candidates[0], 'finish_reason', 'Unknown')
#                             finish_reason = finish_reason.name if hasattr(finish_reason, 'name') else str(finish_reason)
#                     except AttributeError: pass

#                     if finish_reason == "SAFETY": warn_msg = f"API Response Blocked (Attempt {attempt}, Section: {section}): Reason: SAFETY. Detail: {block_reason}. Message: {block_message}"; st.error(warn_msg)
#                     elif finish_reason == "RECITATION": warn_msg = f"API Response Potentially Blocked (Attempt {attempt}, Section: {section}): Finish Reason: RECITATION. Block Reason: {block_reason}."; st.warning(warn_msg)
#                     elif finish_reason == "STOP" and not final_validated_data: warn_msg = f"API Response Ended (Attempt {attempt}, Section: {section}): Finish Reason: STOP, but no valid data parsed yet."; st.info(warn_msg) # Only warn if stopped without success
#                     else: warn_msg = f"API Issue (Attempt {attempt}, Section: {section}): Finish Reason: {finish_reason}. Block Reason: {block_reason}. Response may be incomplete or empty."; st.warning(warn_msg)
#                     section_warnings.append(warn_msg)

#             except types.StopCandidateException as sce:
#                 error_msg = f"Generation Stopped Error (Attempt {attempt}, Section: {section}): {sce}."; st.error(error_msg); section_warnings.append(error_msg); print(traceback.format_exc())
#             except Exception as e:
#                 error_msg = f"Processing Error during API call/prompt generation (Attempt {attempt}, Section: {section}): {type(e).__name__}: {e}"; st.error(error_msg); section_warnings.append(error_msg); section_warnings.append("Traceback logged to console."); print(traceback.format_exc())

#             if final_validated_data is not None: break

#         if final_validated_data is not None:
#             status_placeholder.success(f"✅ Analysis completed successfully for: {section}.")
#             return final_validated_data, "Success", section_warnings
#         else:
#             status_placeholder.error(f"❌ Analysis failed for: {section} after {attempt} attempts.")
#             section_warnings.append(f"Failed to get valid response for section '{section}' after {MAX_VALIDATION_RETRIES + 1} attempts.")
#             return None, "Failed", section_warnings

#     except Exception as outer_err:
#          error_msg = f"Critical Error during setup or execution for section '{section}': {type(outer_err).__name__}: {outer_err}"; st.error(error_msg); section_warnings.append(error_msg); section_warnings.append("Traceback logged to console."); print(traceback.format_exc())
#          status_placeholder.error(f"❌ Critical failure processing section: {section}.")
#          return None, "Failed", section_warnings


# @st.cache_data(show_spinner=False)
# def find_text_in_pdf(_pdf_bytes, search_text):
#     """
#     Searches PDF using prioritized strategy. Returns tuple:
#     (first_page_found, instances_on_first_page, term_used, status_msg, all_findings)
#     """
#     if not _pdf_bytes or not search_text:
#         return None, None, None, "Invalid input (PDF bytes or search text missing).", None

#     doc = None
#     search_text_cleaned = search_text.strip()
#     words = search_text_cleaned.split()
#     num_words = len(words)
#     search_attempts = []

# # --- Build Search Terms List (Prioritized) ---
#     search_attempts = [] # Start with an empty list

#     # 1. First 10 words (only if enough words and significantly longer than 5)
#     if num_words >= SEARCH_PREFIX_MIN_WORDS and num_words > 5:
#         term_10 = ' '.join(words[:10])
#         search_attempts.append({'term': term_10, 'desc': "first 10 words"})

#     # 2. First 5 words (if enough words and different from term_10 if term_10 exists)
#     if num_words >= SEARCH_PREFIX_MIN_WORDS:
#         term_5 = ' '.join(words[:5])
#         # Check if search_attempts is empty OR if term_5 is different from the first term (term_10)
#         if not search_attempts or term_5 != search_attempts[0]['term']:
#             search_attempts.append({'term': term_5, 'desc': "first 5 words"})

#     # 3. Full text (if provided and different from shorter prefixes already added)
#     term_full = search_text_cleaned
#     if term_full and not any(term_full == a['term'] for a in search_attempts):
#         search_attempts.append({'term': term_full, 'desc': "full text"})

#     # 4. First sentence (fallback, if long enough and different from other terms)
#     sentences = re.split(r'(?<=[.?!])\s+', term_full)
#     term_sentence = sentences[0].strip() if sentences else ""
#     if len(term_sentence) >= SEARCH_FALLBACK_MIN_LENGTH and not any(term_sentence == a['term'] for a in search_attempts):
#         search_attempts.append({'term': term_sentence, 'desc': "first sentence fallback"})

#     # --- Execute Search Attempts ---
#     try:
#         doc = fitz.open(stream=_pdf_bytes, filetype="pdf")
#         search_flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES

#         for attempt in search_attempts:
#             term = attempt['term']; desc = attempt['desc']; findings_for_term = []
#             for page_index in range(doc.page_count):
#                 page = doc.load_page(page_index)
#                 instances = page.search_for(term, flags=search_flags, quads=False)
#                 if instances: findings_for_term.append((page_index + 1, instances))

#             if findings_for_term:
#                 doc.close(); first_page_found = findings_for_term[0][0]; instances_on_first_page = findings_for_term[0][1]
#                 if len(findings_for_term) == 1:
#                     status = f"✅ Found using '{desc}' on page {first_page_found}."; return first_page_found, instances_on_first_page, term, status, None
#                 else:
#                     pages_found = sorted([f[0] for f in findings_for_term]); status = f"⚠️ Found matches using '{desc}' on multiple pages: {pages_found}. Showing first match on page {first_page_found}."; return first_page_found, instances_on_first_page, term, status, findings_for_term

#         doc.close(); tried_descs = [a['desc'] for a in search_attempts]; return None, None, None, f"❌ Text not found (tried methods: {', '.join(tried_descs)}).", None
#     except Exception as e:
#         if doc: doc.close(); print(f"ERROR searching PDF: {e}\n{traceback.format_exc()}"); return None, None, None, f"❌ Error during PDF search: {e}", None


# def render_pdf_page_to_image(_pdf_bytes, page_number, highlight_instances=None, dpi=150):
#     """Renders PDF page to PNG image bytes, applying highlights. Returns (image_bytes, status_msg)."""
#     if not _pdf_bytes or page_number < 1:
#         return None, "Invalid input for rendering (PDF bytes missing or invalid page number)."

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
#                         if highlight:
#                             # FIX: Use stroke only for highlight color
#                             highlight.set_colors(stroke=fitz.utils.getColor("yellow"))
#                             highlight.set_opacity(0.4) # Adjust opacity as needed
#                             highlight.update()
#                             highlight_applied_count += 1
#                         else: print(f"WARN: Failed to add highlight annotation for instance: {inst} on page {page_number}")

#                 if highlight_applied_count > 0: render_status_message = f"Rendered page {page_number} with {highlight_applied_count} highlight(s)."
#                 elif highlight_instances: render_status_message = f"Rendered page {page_number}, but no valid highlights applied from provided instances."
#             except Exception as highlight_err: print(f"ERROR applying highlights on page {page_number}: {highlight_err}\n{traceback.format_exc()}"); render_status_message = f"⚠️ Error applying highlights: {highlight_err}"

#         pix = page.get_pixmap(dpi=dpi, alpha=False); image_bytes = pix.tobytes("png")
#     except Exception as e: print(f"ERROR rendering page {page_number}: {e}\n{traceback.format_exc()}"); render_status_message = f"❌ Error rendering page {page_number}: {e}"; image_bytes = None
#     finally:
#         if doc: doc.close()
#     return image_bytes, render_status_message


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
# # Initialize selected_sections in session state if it doesn't exist
# if 'selected_sections_to_run' not in st.session_state:
#     # Default to selecting all sections initially, or maybe just the first for faster testing?
#     st.session_state.selected_sections_to_run = list(ALL_SECTIONS.keys())[:1] # Default to only first section for testing
#     # Or uncomment below to default to all sections:
#     # st.session_state.selected_sections_to_run = list(ALL_SECTIONS.keys())


# # --- 5. Streamlit UI Logic ---
# st.title("JASPER - Just A Smart Platform for Extraction and Review")
# st.markdown("Upload a PDF agreement, **enter your Gemini API Key**, **select sections**, click 'Analyse'. Results grouped below. Click clause references to view & highlight.")

# # --- Sidebar Setup ---
# st.sidebar.markdown("## Controls")

# # --- API Key Input (in Sidebar) ---
# api_key_input = st.sidebar.text_input(
#     "Enter your Google AI Gemini API Key",
#     type="password",
#     key="api_key_input",
#     help="Your API key is used only for this session and is not stored long-term.",
#     value=st.session_state.get("api_key", "")
# )
# if api_key_input != st.session_state.api_key:
#     st.session_state.api_key = api_key_input
#     st.rerun()

# if not st.session_state.api_key and not st.session_state.analysis_complete and st.session_state.pdf_bytes:
#     st.sidebar.warning("API Key required to run analysis.", icon="🔑")

# # --- File Upload (in Sidebar) ---
# uploaded_file_obj = st.sidebar.file_uploader(
#     "Upload Facility Agreement PDF", type="pdf", key=f"pdf_uploader_{st.session_state.run_key}"
# )

# if uploaded_file_obj is not None:
#     new_file_bytes = uploaded_file_obj.getvalue()
#     if new_file_bytes != st.session_state.get('pdf_bytes'):
#         st.session_state.pdf_bytes = new_file_bytes
#         st.session_state.pdf_display_ready = True
#         st.session_state.analysis_results = None; st.session_state.analysis_complete = False
#         st.session_state.processing_in_progress = False; st.session_state.current_page = 1
#         st.session_state.run_status_summary = []; st.session_state.excel_data = None
#         st.session_state.search_trigger = None; st.session_state.last_search_result = None
#         st.session_state.show_wording_states = defaultdict(bool)
#         st.toast("✅ New PDF file loaded. Viewer ready.", icon="📄")
#         st.rerun()

# # --- Section Selection for Analysis (NEW) ---
# st.sidebar.markdown("### Select Sections to Analyse")
# selected_sections = st.sidebar.multiselect(
#     "Choose sections:",
#     options=list(ALL_SECTIONS.keys()), # Use keys from the full definition
#     default=st.session_state.selected_sections_to_run, # Use session state for default
#     key="section_selector", # Give it a key for state management
#     help="Select which parts of the document you want to analyse in this run."
# )
# # Update session state when the selection changes
# if selected_sections != st.session_state.selected_sections_to_run:
#     st.session_state.selected_sections_to_run = selected_sections
#     st.rerun() # Rerun to potentially update button state/tooltips immediately

# # --- Analysis Trigger (in Sidebar) ---
# can_analyse = (st.session_state.pdf_bytes is not None and
#                st.session_state.api_key is not None and
#                not st.session_state.processing_in_progress and
#                not st.session_state.analysis_complete and
#                st.session_state.selected_sections_to_run) # Check if sections are selected

# analyse_button_tooltip = "Analysis complete for the current file." if st.session_state.analysis_complete else \
#                          "Analysis is currently running." if st.session_state.processing_in_progress else \
#                          "Upload a PDF file first." if not st.session_state.pdf_bytes else \
#                          "Enter your Gemini API key first." if not st.session_state.api_key else \
#                          "Select at least one section to analyse." if not st.session_state.selected_sections_to_run else \
#                          "Start analyzing the selected sections"

# if st.sidebar.button("✨ Analyse Document", key="analyse_button", disabled=not can_analyse, help=analyse_button_tooltip, use_container_width=True, type="primary"):
#     if not st.session_state.api_key: st.error("API Key is missing. Please enter it in the sidebar.")
#     elif not st.session_state.pdf_bytes: st.error("No PDF file uploaded. Please upload a file.")
#     elif not st.session_state.selected_sections_to_run: st.error("No sections selected for analysis. Please select sections in the sidebar.")
#     else:
#         # --- Start Analysis Process ---
#         st.session_state.processing_in_progress = True
#         st.session_state.analysis_complete = False
#         st.session_state.analysis_results = [] # Initialize as list
#         st.session_state.run_key += 1; st.session_state.run_status_summary = []
#         st.session_state.excel_data = None; st.session_state.search_trigger = None
#         st.session_state.last_search_result = None; st.session_state.show_wording_states = defaultdict(bool)

#         current_api_key = st.session_state.api_key
#         run_start_time = datetime.now(); run_timestamp_str = run_start_time.strftime("%Y-%m-%d %H:%M:%S")
#         base_file_name = getattr(uploaded_file_obj, 'name', f"upload_{run_start_time.strftime('%Y%m%d%H%M%S')}.pdf")

#         try:
#             genai.configure(api_key=current_api_key)
#             st.toast("API Key validated and configured.", icon="🔑")
#         except Exception as config_err:
#             st.error(f"❌ Failed to configure Gemini API with provided key: {config_err}")
#             st.session_state.processing_in_progress = False; st.stop()

#         status_container = st.container(); progress_bar = status_container.progress(0, text="Initializing analysis...")
#         status_text = status_container.empty()

#         temp_dir = "temp_uploads"; safe_base_name = re.sub(r'[^\w\-.]', '_', base_file_name)
#         temp_file_path = os.path.join(APP_DIR, temp_dir, f"{run_start_time.strftime('%Y%m%d%H%M%S')}_{safe_base_name}")
#         os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

#         gemini_uploaded_file_ref = None; all_validated_data = []; overall_success = True

#         try:
#             status_text.info("💾 Saving file temporarily..."); progress_bar.progress(5, text="Saving temp file...")
#             with open(temp_file_path, "wb") as f: f.write(st.session_state.pdf_bytes)

#             status_text.info("☁️ Uploading file to Google Cloud AI..."); progress_bar.progress(10, text="Uploading to cloud...")
#             for upload_attempt in range(3):
#                 try:
#                     gemini_uploaded_file_ref = genai.upload_file(path=temp_file_path)
#                     st.toast(f"File '{gemini_uploaded_file_ref.display_name}' uploaded to cloud.", icon="☁️"); break
#                 except Exception as upload_err:
#                     err_str = str(upload_err).lower()
#                     if "api key" in err_str or "authenticat" in err_str or "permission" in err_str: st.error(f"❌ File upload failed due to API key/permission issue: {upload_err}"); st.error("Please verify the API key has File API permissions enabled."); raise
#                     elif upload_attempt < 2: st.warning(f"Upload attempt {upload_attempt+1} failed: {upload_err}. Retrying..."); time.sleep(2 + upload_attempt)
#                     else: st.error(f"Upload failed after multiple attempts: {upload_err}"); raise
#             if not gemini_uploaded_file_ref: raise Exception("Failed to upload file to Google Cloud AI after retries.")

#             progress_bar.progress(15, text="File uploaded. Starting section analysis...")

#             # --- GET SECTIONS TO RUN FROM SESSION STATE ---
#             sections_to_process = st.session_state.selected_sections_to_run
#             num_sections = len(sections_to_process)
#             progress_per_section = (95 - 15) / num_sections if num_sections > 0 else 0

#             # --- Process ONLY the SELECTED Sections ---
#             for i, section_name in enumerate(sections_to_process):
#                 if section_name not in ALL_SECTIONS: st.warning(f"Skipping invalid section '{section_name}' found in selection."); continue

#                 current_progress = int(15 + (i * progress_per_section))
#                 progress_bar.progress(current_progress, text=f"Analysing Section: {section_name}...")

#                 # generate_section_analysis uses ALL_SECTIONS via helper functions
#                 section_data, section_status, section_warnings = generate_section_analysis(
#                     section_name, gemini_uploaded_file_ref, status_text, current_api_key
#                 )
#                 st.session_state.run_status_summary.append({"section": section_name, "status": section_status, "warnings": section_warnings})

#                 if section_status == "Success" and section_data:
#                     for item in section_data: item["File Name"] = base_file_name; item["Generation Time"] = run_timestamp_str
#                     all_validated_data.extend(section_data)
#                 else: overall_success = False

#                 progress_bar.progress(int(15 + ((i + 1) * progress_per_section)), text=f"Completed: {section_name}")

#             # --- Finalize Analysis ---
#             st.session_state.analysis_results = all_validated_data
#             progress_bar.progress(100, text="Analysis process finished!")
#             if overall_success: status_text.success("🏁 Analysis finished successfully!")
#             else: status_text.warning("🏁 Analysis finished, but some sections encountered issues (see summary below).")
#             st.session_state.analysis_complete = True

#         except Exception as main_err:
#             st.error(f"❌ CRITICAL ERROR during analysis workflow: {main_err}"); print(traceback.format_exc())
#             overall_success = False; st.session_state.analysis_complete = False
#             st.session_state.run_status_summary.append({"section": "Overall Process Control", "status": "Critical Error", "warnings": [str(main_err), "Analysis halted. See logs."]})
#             status_text.error(f"Analysis stopped due to critical error: {main_err}")
#         finally:
#             st.session_state.processing_in_progress = False; time.sleep(4); status_text.empty(); progress_bar.empty()
#             if gemini_uploaded_file_ref and hasattr(gemini_uploaded_file_ref, 'name'):
#                 try:
#                     status_text.info(f"☁️ Deleting temporary cloud file: {gemini_uploaded_file_ref.name}...")
#                     genai.delete_file(name=gemini_uploaded_file_ref.name); st.toast("Cloud file deleted.", icon="🗑️"); time.sleep(1); status_text.empty()
#                 except Exception as del_err: st.sidebar.warning(f"Cloud cleanup issue: {del_err}", icon="⚠️"); status_text.warning(f"Could not delete cloud file: {del_err}"); print(f"WARN: Failed to delete cloud file {gemini_uploaded_file_ref.name}: {del_err}")
#             if os.path.exists(temp_file_path):
#                 try: os.remove(temp_file_path)
#                 except Exception as local_del_err: st.sidebar.warning(f"Local temp file cleanup issue: {local_del_err}", icon="⚠️"); print(f"WARN: Failed to delete local temp file {temp_file_path}: {local_del_err}")
#         st.rerun()

# # --- Display Area (Results and PDF Viewer) ---
# if st.session_state.pdf_bytes is not None:
#     col1, col2 = st.columns([6, 4]) # Results | PDF Viewer

#     # --- Column 1: Analysis Results ---
#     with col1:
#         if st.session_state.run_status_summary:
#             final_status = "✅ Success" # Start with success
#             has_failures = any(s['status'] == "Failed" or "Error" in s['status'] for s in st.session_state.run_status_summary)
#             has_warnings = any(s['status'] != "Success" and not has_failures for s in st.session_state.run_status_summary)

#             # Corrected logic block
#             if has_failures:
#                 final_status = "❌ Failed"
#             elif has_warnings:
#                 final_status = "⚠️ Issues"
#             # No 'else' needed, as final_status defaults to "✅ Success"

#             with st.expander(f"📊 Analysis Run Summary ({final_status})", expanded=(final_status != "✅ Success")):
#                 for item in st.session_state.run_status_summary:
#                     icon = "✅" if item['status'] == "Success" else "❌" if item['status'] == "Failed" or "Error" in item['status'] else "⚠️"
#                     st.markdown(f"**{item['section']}**: {icon} {item['status']}")
#                     if item['warnings']:
#                         filtered_warnings = [msg for msg in item['warnings'] if not (isinstance(msg, str) and msg.startswith("Validation Issues Found"))]
#                         if filtered_warnings:
#                             with st.container():
#                                 st.caption("Details:")
#                                 for msg in filtered_warnings:
#                                     msg_str = str(msg);
#                                     if "CRITICAL" in msg_str or "Error" in msg_str or "Block" in msg_str or "Fail" in msg_str: st.error(f" L> {msg_str}")
#                                     elif "Warn" in msg_str or "Missing" in msg_str or "Unexpected" in msg_str or "Empty list" in msg_str or "validation issues" in msg_str.lower(): st.warning(f" L> {msg_str}")
#                                     else: st.caption(f" L> {msg_str}")

#         st.subheader("Analysis Results")
#         if st.session_state.analysis_complete and st.session_state.analysis_results:
#             results_list = st.session_state.analysis_results; grouped_results = defaultdict(list); categories_ordered = []
#             for item in results_list:
#                 category = item.get("Question Category", "Uncategorized")
#                 if category not in grouped_results: categories_ordered.append(category)
#                 grouped_results[category].append(item)

#             if categories_ordered:
#                 category_tabs = st.tabs(categories_ordered)
#                 for i, category in enumerate(categories_ordered):
#                     with category_tabs[i]:
#                         category_items = sorted(grouped_results[category], key=lambda x: x.get('Question Number', float('inf')))
#                         for index, result_item in enumerate(category_items):
#                             q_num = result_item.get('Question Number', 'N/A'); question_text = result_item.get('Question', 'N/A')
#                             # FIX: Remove key from st.expander
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
#                                                 toggle_key = f"toggle_wording_{base_key}"
#                                                 show_wording = st.toggle("Show Wording", key=toggle_key, value=st.session_state.show_wording_states[toggle_key], help="Show/hide the exact clause wording extracted by the AI.")
#                                                 if show_wording != st.session_state.show_wording_states[toggle_key]: st.session_state.show_wording_states[toggle_key] = show_wording; st.rerun()
#                                         if st.session_state.show_wording_states.get(f"toggle_wording_{base_key}", False):
#                                             st.text_area(f"AI Extracted Wording for '{clause_ref}':", value=clause_wording, height=150, disabled=True, key=f"wording_area_{base_key}")
#                                         st.markdown("---")
#                                 else: st.markdown("**Evidence:** None provided.")
#                                 st.markdown("---")
#                                 st.markdown("**Answer Justification:**")
#                                 justification_text = result_item.get('Answer Justification', ''); just_key = f"justification_{category}_{q_num}_{index}"
#                                 st.text_area(label="Justification Text Area", value=justification_text, height=100, disabled=True, label_visibility="collapsed", key=just_key)
#             else: st.warning("Analysis generated results, but they could not be grouped by category. Displaying raw list."); st.json(results_list)

#             st.sidebar.markdown("---"); st.sidebar.markdown("## Export Results")
#             if st.sidebar.button("Prepare Data for Excel Download", key="prep_excel", use_container_width=True):
#                 st.session_state.excel_data = None; excel_prep_status = st.sidebar.empty(); excel_prep_status.info("Preparing Excel data...")
#                 try:
#                     excel_rows = [];
#                     for item in results_list:
#                         references = []; first_search_text = "N/A"; evidence = item.get("Evidence")
#                         if evidence:
#                             for i, ev in enumerate(evidence):
#                                 if isinstance(ev, dict): references.append(str(ev.get("Clause Reference", "N/A")));
#                                 if i == 0 and isinstance(ev, dict): first_search_text = ev.get("Searchable Clause Text", "N/A")
#                                 else: references.append("[Invalid Evidence Item]")
#                         excel_row = {"File Name": item.get("File Name", ""), "Generation Time": item.get("Generation Time", ""), "Question Number": item.get("Question Number"), "Question Category": item.get("Question Category", "Uncategorized"), "Question": item.get("Question", "N/A"), "Answer": item.get("Answer", "N/A"), "Answer Justification": item.get("Answer Justification", "N/A"), "Clause References (Concatenated)": "; ".join(references) if references else "N/A", "First Searchable Clause Text": first_search_text}
#                         excel_rows.append(excel_row)
#                     if not excel_rows: excel_prep_status.warning("No data available to export."); st.session_state.excel_data = None
#                     else:
#                         df_excel = pd.DataFrame(excel_rows); final_columns = [col for col in EXCEL_COLUMN_ORDER if col in df_excel.columns]; extra_cols = [col for col in df_excel.columns if col not in final_columns]; df_excel = df_excel[final_columns + extra_cols]
#                         output = io.BytesIO();
#                         with pd.ExcelWriter(output, engine='openpyxl') as writer: df_excel.to_excel(writer, index=False, sheet_name='Analysis Results')
#                         st.session_state.excel_data = output.getvalue(); excel_prep_status.success("✅ Excel file ready for download!"); time.sleep(2); excel_prep_status.empty()
#                 except Exception as excel_err: excel_prep_status.error(f"Excel Prep Error: {excel_err}"); print(traceback.format_exc())

#             if st.session_state.excel_data:
#                 current_filename = "analysis_results"
#                 if st.session_state.analysis_results and isinstance(st.session_state.analysis_results, list) and len(st.session_state.analysis_results) > 0:
#                      first_result = st.session_state.analysis_results[0]
#                      if isinstance(first_result, dict) and "File Name" in first_result: current_filename = first_result["File Name"]
#                 safe_base_name = re.sub(r'[^\w\s-]', '', os.path.splitext(current_filename)[0]).strip().replace(' ', '_')
#                 download_filename = f"Analysis_{safe_base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
#                 st.sidebar.download_button(label="📥 Download Results as Excel", data=st.session_state.excel_data, file_name=download_filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_excel_final", use_container_width=True)

#         elif st.session_state.analysis_complete and not st.session_state.analysis_results: st.info("Analysis process completed, but no valid results were generated. Check the run summary above for potential issues.")
#         elif st.session_state.processing_in_progress: st.info("Analysis is currently in progress...")
#         elif not st.session_state.analysis_complete and st.session_state.pdf_bytes is not None: st.info("PDF loaded. Select sections and click 'Analyse Document' in the sidebar to start.")

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
#                 with fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf") as doc: total_pages = doc.page_count
#             except Exception as pdf_load_err: st.error(f"Error loading PDF for page count: {pdf_load_err}"); total_pages = 1; st.session_state.current_page = 1

#             current_display_page = max(1, min(int(st.session_state.get('current_page', 1)), total_pages))
#             if current_display_page != st.session_state.get('current_page'): st.session_state.current_page = current_display_page

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
#                 num_buttons = len(found_pages); btn_cols = st.columns(num_buttons)
#                 for idx, p_num in enumerate(found_pages):
#                     is_current = (p_num == current_display_page)
#                     if btn_cols[idx].button(f"Page {p_num}", key=f"jump_{p_num}_{search_context_ref}", disabled=is_current, use_container_width=True):
#                         st.session_state.current_page = p_num; new_instances = next((inst for pg, inst in multi_findings if pg == p_num), None)
#                         st.session_state.last_search_result['instances'] = new_instances; st.session_state.last_search_result['page'] = p_num; st.rerun()

#             st.markdown("---")
#             highlights_to_apply = None; render_status_override = None
#             if st.session_state.last_search_result and st.session_state.last_search_result.get('page') == current_display_page:
#                 highlights_to_apply = st.session_state.last_search_result.get('instances')
#                 if not st.session_state.last_search_result.get('all_findings'): render_status_override = st.session_state.last_search_result.get('status')

#             image_bytes, render_status = render_pdf_page_to_image(st.session_state.pdf_bytes, current_display_page, highlight_instances=highlights_to_apply, dpi=150)

#             if image_bytes:
#                 # Use use_container_width instead of deprecated use_column_width
#                 st.image(image_bytes, caption=f"Page {current_display_page} - View", use_container_width=True)
#                 final_status = render_status_override if render_status_override else render_status
#                 if not (st.session_state.last_search_result and st.session_state.last_search_result.get('all_findings')):
#                     if final_status:
#                         if "✅" in final_status or "✨" in final_status or "Found" in final_status : viewer_status_placeholder.success(final_status)
#                         elif "⚠️" in final_status or "warning" in final_status.lower(): viewer_status_placeholder.warning(final_status)
#                         elif "❌" in final_status or "error" in final_status.lower(): viewer_status_placeholder.error(final_status)
#                         else: viewer_status_placeholder.caption(final_status)
#                     else: viewer_status_placeholder.empty()
#             else: viewer_status_placeholder.error(f"Failed to render page {current_display_page}. {render_status or ''}")
#         else: st.info("PDF loaded, preparing viewer..."); viewer_status_placeholder.empty()
#         st.markdown('</div>', unsafe_allow_html=True) # Sticky wrapper end

# # --- Fallback messages if no PDF is loaded ---
# elif not st.session_state.pdf_bytes:
#      st.info("⬆️ Upload a PDF file using the sidebar to begin.")