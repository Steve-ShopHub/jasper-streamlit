# pages/defined_terms_graph.py
# --- COMPLETE FILE vX.Y+3 (State Init & Timeout Fixes) ---

import streamlit as st
import google.generativeai as genai
from google.generativeai import types
import google.api_core.exceptions # For specific API errors
import fitz  # PyMuPDF for PDF text extraction
import re
import os
import traceback
import time
import io # For download button
import json
import graphviz # Python graphviz library for parsing DOT and rendering
import networkx as nx # For graph analysis (cycles, orphans, neighbors)
import pandas as pd # For CSV export
from streamlit_agraph import agraph, Node, Edge, Config
from PIL import Image # For Logo import
from collections import defaultdict
import math # For chunking calculation

# --- Configuration ---
MODEL_NAME = "gemini-2.5-pro-preview-03-25" # Updated to a recent preview model
PAGE_TITLE = "Defined Terms Relationship Grapher"
PAGE_ICON = "🔗"
DEFAULT_NODE_COLOR = "#ACDBC9" # Light greenish-teal
HIGHLIGHT_COLOR = "#FFA07A" # Light Salmon for selected node
NEIGHBOR_COLOR = "#ADD8E6" # Light Blue for neighbors
EDGE_EXTRACTION_DELAY = 0.5 # Seconds delay between edge extraction API calls
TERM_EXTRACTION_CHUNK_DELAY = 1.0 # Seconds delay between term chunk API calls
# Chunking Parameters (tune these based on testing and model limits)
# Reduced chunk size to mitigate timeouts
TERM_CHUNK_SIZE = 50000 # Target characters per chunk for term extraction
TERM_CHUNK_OVERLAP = 1500   # Characters overlap between chunks

# --- Set Page Config ---
st.set_page_config(layout="wide", page_title=PAGE_TITLE, page_icon=PAGE_ICON)

# --- Optional CSS ---
st.markdown("""
<style>
    /* Ensure Streamlit containers don't add excessive padding */
     div[data-testid="stVerticalBlock"] > div[style*="gap: 1rem;"] {
        gap: 0.5rem !important;
     }
    /* Style for the definition display area */
    .definition-box {
        background-color: #f0f2f6;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
        height: 150px; /* Adjust height as needed */
        overflow-y: auto; /* Add scroll if definition is long */
        font-size: 0.9em;
        white-space: pre-wrap; /* Ensure line breaks in definitions are respected */
        word-wrap: break-word; /* Break long words */
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Function for Text Extraction ---
@st.cache_data(show_spinner="Extracting text from PDF...")
def extract_text_from_pdf(pdf_bytes):
    """
    Extracts text from PDF bytes. Attempts to isolate the 'Definitions' section.
    Returns (extracted_text, definitions_only_text, error_msg).
    definitions_only_text will be None if section couldn't be isolated.
    """
    if not pdf_bytes:
        return None, None, "No PDF file provided."

    doc = None
    full_text = ""
    definitions_text = None
    error_msg = None

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        st.info(f"📄 PDF has {len(doc)} pages.")
        page_texts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text", sort=True)
            if page_text:
                page_texts.append(page_text)

        full_text = "\n\n--- Page Break --- \n\n".join(page_texts)

        if not full_text.strip():
            return None, None, "Could not extract any text from the PDF."

        # --- Attempt to Isolate Definitions Section ---
        st.info("Attempting to isolate Definitions section...")
        patterns = [
            re.compile(r"^[ \t]*(?:ARTICLE\s+)?1\.\s+(?:Definitions|Interpretation)(.*?)(?:^[ \t]*(?:ARTICLE\s+)?2\.|^[ \t]*(?:[A-Z][A-Z\s]+)\n\n)", re.IGNORECASE | re.MULTILINE | re.DOTALL),
            re.compile(r"^[ \t]*Definitions\s*\n(.*?)(?:^[ \t]*[A-Z][A-Z\s]{5,}\n|^[ \t]*\d+\.\s+[A-Z])", re.IGNORECASE | re.MULTILINE | re.DOTALL),
            re.compile(r"^[ \t]*(?:Definitions|Interpretation)\s*\n(.*?)(?:Agreed Terms|Subject to the terms|IN WITNESS WHEREOF)", re.IGNORECASE | re.MULTILINE | re.DOTALL),
            re.compile(r"(?:Section 1|Clause 1|ARTICLE 1)\s*\.?\s*(Definitions|Interpretation)\s*\n(.*?)(?:^[ \t]*(?:Section 2|Clause 2|ARTICLE 2)|SIGNATURES|SCHEDULES)", re.IGNORECASE | re.MULTILINE | re.DOTALL),
        ]
        found_section = False
        for pattern in patterns:
            match = pattern.search(full_text)
            if match:
                content_group_index = -1 # Default to last group
                if 'content' in pattern.groupindex:
                     content_group_index = 'content'
                elif len(match.groups()) >= 1:
                     # Assume the main content is the last capturing group
                     content_group_index = len(match.groups())

                if content_group_index != -1:
                     definitions_text = match.group(content_group_index).strip()

                     if definitions_text and re.search(r'"[^"]+"\s+means', definitions_text, re.IGNORECASE):
                        st.toast(f"ℹ️ Successfully isolated Definitions section ({len(definitions_text):,} chars).", icon="✂️")
                        print(f"DEBUG: Isolated definitions section, length: {len(definitions_text)}")
                        found_section = True
                        break
                else: # Reset if group logic didn't work for this pattern
                     definitions_text = None

        if not found_section:
             definitions_text = None
             st.warning("⚠️ Could not automatically isolate Definitions section, using full text.", icon="⚠️")
             print(f"DEBUG: Could not isolate definitions, using full text length: {len(full_text)}")


    except Exception as e:
        error_msg = f"Error extracting text: {e}"
        print(traceback.format_exc())
        full_text = None
        definitions_text = None
    finally:
        if doc:
            try: doc.close()
            except Exception as close_err: print(f"Warning: Error closing PDF: {close_err}")

    return full_text, definitions_text, error_msg

# --- Helper Function to Chunk Text ---
def chunk_text(text, chunk_size=TERM_CHUNK_SIZE, overlap=TERM_CHUNK_OVERLAP):
    """Splits text into chunks with overlap."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Find the last newline before the ideal end point, within the overlap zone
        # This helps keep definitions intact if they cross chunk boundaries
        effective_end = min(end, len(text))
        newline_pos = text.rfind('\n', max(0, start + chunk_size - overlap), effective_end)

        # Use the newline position if it's found after the overlap starts
        # and isn't right at the beginning of the potential chunk
        if newline_pos > start + overlap // 2 : # Ensure newline is not too early
             final_end = newline_pos
        else:
             # Otherwise, just cut at the chunk size or text end
             final_end = effective_end

        chunks.append(text[start:final_end])

        # Calculate the next start position, moving back by overlap amount
        # Ensure we make progress and don't get stuck
        next_start = final_end - overlap
        if next_start <= start:
             # If overlap pushes us back or keeps us static, force progress
             # Move forward by chunk_size minus overlap from the *previous* start,
             # or simply move to the current end if it results in a later start
             next_start = max(final_end, start + chunk_size - overlap)


        # Break if we are not making progress (e.g., overlap >= chunk_size)
        # or if the next start is beyond the text length
        if next_start >= len(text) or next_start <= start :
            # If there's remaining text not covered, add it as a final chunk
            if final_end < len(text):
                 chunks.append(text[final_end:])
            break

        start = next_start


    # Filter out potentially empty chunks resulting from edge cases
    return [c for c in chunks if c and c.strip()]


# --- Helper Function to Parse AI JSON Response (Terms Only) ---
def parse_terms_response(response_text):
    """Parses the AI's JSON response specifically for the 'terms' list."""
    raw_snippet = response_text[:500] + ("..." if len(response_text) > 500 else "")
    json_text = None
    data = None # Initialize data to None
    was_truncated = False
    final_error_msg = None

    try:
        # 1. Attempt normal parsing (stripping markdown)
        match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE | re.DOTALL)
        if match:
            json_text = match.group(1).strip()
        else:
            # Check if it looks like JSON (starts with { or [ usually)
            stripped_response = response_text.strip()
            if stripped_response.startswith("{") or stripped_response.startswith("["):
                 json_text = stripped_response
            else:
                 return None, f"Response does not appear to be JSON. Raw text snippet: {raw_snippet}", False

        if not json_text:
            return None, "AI response content is empty after stripping.", False

        data = json.loads(json_text)

    except json.JSONDecodeError as json_err:
        # 2. Attempt fallback parsing for truncated JSON
        original_error = json_err
        final_error_msg = f"Failed to decode AI JSON response: {json_err}. Raw text snippet: {raw_snippet}"
        st.warning(f"⚠️ Initial JSON parsing failed ({json_err}). Attempting recovery for truncated response...", icon="🔧")

        if json_text: # Only attempt recovery if we had potential JSON text
            try:
                # Try to find the end of the last complete term object `}` within the terms list `[...]`
                last_obj_end = -1
                search_start = 0
                nesting_level = 0 # Track curly brace nesting
                in_string = False

                # Scan towards the error position (or end of string)
                scan_limit = min(json_err.pos + 10, len(json_text)) # Look slightly beyond error pos

                for i in range(scan_limit):
                    char = json_text[i]

                    # Toggle in_string state
                    if char == '"':
                         # Handle escaped quotes
                         if i > 0 and json_text[i-1] == '\\':
                             pass # Ignore escaped quote
                         else:
                             in_string = not in_string

                    if not in_string:
                         if char == '{':
                             nesting_level += 1
                         elif char == '}':
                             nesting_level -= 1
                             # Check if this is a closing brace for a top-level object within the list
                             if nesting_level == 1: # Assuming structure {"terms": [ {..}, {..} ]} -> level 1 inside list
                                 # Check if followed by comma or closing bracket
                                 next_meaningful_char = None
                                 for next_i in range(i + 1, len(json_text)):
                                     next_c = json_text[next_i]
                                     if not next_c.isspace():
                                         next_meaningful_char = next_c
                                         break
                                 if next_meaningful_char in [',', ']']:
                                      last_obj_end = i # Found likely end of a complete object


                if last_obj_end != -1:
                    # Try to reconstruct a potentially valid partial JSON: {"terms": [...]}
                    terms_list_start = json_text.find('"terms":')
                    if terms_list_start != -1:
                        terms_list_start = json_text.find('[', terms_list_start)
                        if terms_list_start != -1:
                            # Slice from start of list '[' up to and including the last found '}'
                            potential_json_text = json_text[terms_list_start : last_obj_end + 1]
                            # Construct the full potential JSON
                            potential_json = f'{{"terms": {potential_json_text}]}}' # Add outer structure + closing list bracket
                            try:
                                data = json.loads(potential_json)
                                was_truncated = True
                                final_error_msg = None # Succeeded with fallback
                                st.warning("⚠️ Successfully recovered partial terms list from truncated response.", icon="✅")
                            except json.JSONDecodeError as inner_err:
                                print(f"DEBUG: Fallback parsing failed: {inner_err} on text: {potential_json}")
                                # If reconstruction failed, keep original error
                                final_error_msg = f"Failed to decode AI JSON response: {original_error}. Recovery attempt failed. Raw snippet: {raw_snippet}"
                        else: final_error_msg += " (Could not find start of terms list '[' for fallback.)"
                    else: final_error_msg += " (Could not find '\"terms\":' key for fallback.)"
                else: final_error_msg += " (Could not find a likely end '}' for any term object.)"

            except Exception as fallback_err:
                st.warning(f"⚠️ JSON recovery attempt failed: {fallback_err}", icon="❌")
                final_error_msg = f"Failed to decode AI JSON response: {original_error}. Recovery attempt failed ({fallback_err}). Raw snippet: {raw_snippet}"

        # If data is still None or error persists after fallback, return the error
        if data is None or final_error_msg:
            return None, final_error_msg or "Unknown parsing error.", False

    # 3. Validate structure if parsing succeeded (fully or partially)
    try:
        if not isinstance(data, dict):
            return None, f"Parsed data is not a JSON object ({type(data).__name__}). Content: {str(data)[:200]}...", was_truncated
        if "terms" not in data or not isinstance(data["terms"], list):
            # Allow parsing if the top level is the list itself (sometimes models return just the list)
            if isinstance(data, list):
                st.warning("Response was a list directly, expected {'terms': [...]}. Adapting.")
                data = {"terms": data} # Wrap it
            else:
                return None, f"Parsed data missing 'terms' list key or incorrect type. Keys: {list(data.keys())}", was_truncated

        validated_terms = []
        term_names_processed = set() # Keep track locally within validation
        for item in data.get("terms", []): # Use .get for safety
            if isinstance(item, dict) and "name" in item and "definition" in item and isinstance(item["name"], str) and isinstance(item["definition"], str):
                term_name = item["name"].strip()
                if term_name: # Allow processing even if already seen (for deduplication later)
                    validated_terms.append({"name": term_name, "definition": item["definition"]})
            # else: st.warning(f"Skipping malformed term item: {item}") # Optional debug

        if not validated_terms and not was_truncated:
             # It's possible a chunk legitimately has no terms
             print("DEBUG: AI response parsed correctly but contained no valid terms.")
             return {"terms": []}, None, False # Return empty list, no error
        elif not validated_terms and was_truncated:
             st.warning("⚠️ No valid terms found in the recovered partial response.")
             return {"terms": []}, None, was_truncated # Return empty list but no error

        return {"terms": validated_terms}, None, was_truncated

    except Exception as e:
        print(traceback.format_exc()) # Print full traceback for validation errors
        return None, f"Error validating parsed terms structure: {e}", was_truncated


# --- Helper Function to Parse AI JSON Response (Edge List Only) ---
def parse_edge_list_response(response_text, source_term_name):
    """Parses the AI's JSON response specifically for a list of target term strings."""
    raw_snippet = response_text[:500] + ("..." if len(response_text) > 500 else "")
    json_text = None
    data = None # Initialize data
    was_truncated = False
    final_error_msg = None

    try:
        # 1. Attempt normal parsing (stripping markdown)
        match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE | re.DOTALL)
        json_text = match.group(1).strip() if match else response_text.strip()

        if not json_text:
            # Treat empty response as valid empty list
             return [], None, False

        # Check if it looks like a JSON list
        if not json_text.startswith("["):
             # Maybe the AI failed and returned plain text?
             # For now, treat non-list-looking string as an error if not empty
             if json_text.strip():
                 return None, f"Edge response for '{source_term_name}' not a JSON list. Raw: {raw_snippet}", False
             else:
                 return [], None, False # Empty string is like an empty list

        data = json.loads(json_text)

    except json.JSONDecodeError as json_err:
        # 2. Attempt fallback parsing for truncated JSON list
        original_error = json_err
        final_error_msg = f"Failed to decode edge list JSON for '{source_term_name}': {json_err}. Raw: {raw_snippet}"
        st.warning(f"⚠️ Initial JSON parsing failed for '{source_term_name}' edges ({json_err}). Attempting recovery...", icon="🔧")

        if json_text:
            try:
                 # Find the last complete string `"` before the error position
                 last_quote_end = json_text.rfind('"', 0, json_err.pos)
                 if last_quote_end != -1:
                     # Find the comma before that quote, if any
                     last_comma = json_text.rfind(',', 0, last_quote_end)
                     # Find the opening bracket before the comma/quote
                     list_start = json_text.find('[')

                     if list_start != -1:
                         # Decide where to cut: after last comma or after last full string
                         cut_pos = last_quote_end + 1 # Default: cut after the closing quote

                         # Refined logic: if a comma exists between the start of the last string and the end quote,
                         # it means the truncation happened *inside* the string. In this case, we should
                         # try cutting at the *previous* comma if one exists.
                         last_quote_start = json_text.rfind('"', 0, last_quote_end)
                         if last_quote_start != -1 and json_text.rfind(',', last_quote_start, last_quote_end) != -1:
                             # Truncation likely within the last string. Find comma before *this* string.
                             prev_comma = json_text.rfind(',', 0, last_quote_start)
                             if prev_comma > list_start:
                                 cut_pos = prev_comma # Cut at the comma before the truncated string
                             else:
                                 # No comma before, maybe it was the first element? Cut after list start
                                 cut_pos = list_start + 1 # Effectively yields '[]' or similar

                         elif last_comma > list_start:
                              # Comma exists before the last quote and after the list start.
                              # Check if it's after the *start* of the last complete string.
                              if last_quote_start != -1 and last_comma > last_quote_start:
                                   cut_pos = last_comma # Cut at the comma separating complete strings

                         # Ensure cut_pos is at least after the opening bracket
                         cut_pos = max(cut_pos, list_start + 1)

                         potential_json = json_text[list_start : cut_pos] + ']' # Reconstruct list
                         # Add basic validation for the reconstructed string
                         if potential_json.count('[') == 1 and potential_json.count(']') == 1:
                             data = json.loads(potential_json)
                             was_truncated = True
                             final_error_msg = None
                             st.warning(f"⚠️ Successfully recovered partial edge list for '{source_term_name}'.", icon="✅")
                         else:
                             raise ValueError(f"Reconstructed JSON invalid: {potential_json}")

                     else: raise ValueError("Could not find starting '['")
                 else: raise ValueError("Could not find closing '\"' for fallback.")

            except Exception as fallback_err:
                st.warning(f"⚠️ JSON recovery attempt failed for '{source_term_name}' edges: {fallback_err}", icon="❌")
                final_error_msg = f"Failed to decode edge list JSON for '{source_term_name}': {original_error}. Recovery failed. Raw: {raw_snippet}"

        if data is None or final_error_msg:
            return None, final_error_msg or "Unknown parsing error for edge list.", False

    # 3. Validate structure if parsing succeeded
    try:
        if not isinstance(data, list):
            return None, f"Parsed edge data for '{source_term_name}' is not a list ({type(data).__name__}). Content: {str(data)[:200]}...", was_truncated

        validated_targets = [item.strip() for item in data if isinstance(item, str) and item.strip()]
        return validated_targets, None, was_truncated

    except Exception as e:
        print(traceback.format_exc())
        return None, f"Error validating parsed edge list structure for '{source_term_name}': {e}", was_truncated


# --- Initialize Session State ---

def initialize_dtg_state_defaults():
    """Sets default values for all DTG session state keys if they don't exist."""
    defaults = {
        'api_key': os.environ.get("GEMINI_API_KEY", ""), # Optionally load from env var
        'dtg_pdf_bytes': None,
        'dtg_pdf_name': None,
        'dtg_extracted_text': None,
        'dtg_definitions_text': None,
        'dtg_error': None,
        'dtg_processing': False,
        'dtg_graph_data': None,
        'dtg_nx_graph': None,
        'dtg_cycles': None,
        'dtg_orphans': None,
        'dtg_filter_term': "",
        'dtg_highlight_node': None,
        'dtg_layout': 'Physics',
        'dtg_total_terms': 0,
        'dtg_terms_processed': 0,
        'dtg_partial_results': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_dtg_analysis_state():
    """Resets session state related to graph analysis results and UI, keeping inputs."""
    st.session_state.dtg_processing = False # Ensure processing stops if reset externally
    st.session_state.dtg_graph_data = None
    st.session_state.dtg_nx_graph = None
    st.session_state.dtg_cycles = None
    st.session_state.dtg_orphans = None
    st.session_state.dtg_error = None
    st.session_state.dtg_total_terms = 0
    st.session_state.dtg_terms_processed = 0
    st.session_state.dtg_partial_results = False
    st.session_state.dtg_filter_term = ""
    st.session_state.dtg_highlight_node = None
    # Optionally reset layout, or keep user's preference
    # st.session_state.dtg_layout = 'Physics'
    # Keep existing: api_key, dtg_pdf_bytes, dtg_pdf_name, dtg_extracted_text, dtg_definitions_text

# --- Call Default Initializer ---
# Call this *once* near the top of the script execution flow
initialize_dtg_state_defaults()

# --- Graph Analysis Functions ---
def build_networkx_graph(graph_data):
    """Builds a NetworkX DiGraph from parsed AI data."""
    if not graph_data or 'terms' not in graph_data or 'edges' not in graph_data:
        st.warning("Cannot build graph: Missing terms or edges data.")
        return None
    G = nx.DiGraph()
    term_names = set()
    for term_data in graph_data.get('terms', []):
        name = term_data.get('name')
        if name:
            G.add_node(name, definition=term_data.get('definition', ''))
            term_names.add(name)
        else:
            st.warning(f"Skipping term with missing name: {term_data}")

    valid_edge_count = 0
    skipped_edge_count = 0
    for edge_data in graph_data.get('edges', []):
        source = edge_data.get('source')
        target = edge_data.get('target')
        if source and target and source in term_names and target in term_names:
             # Avoid self-loops unless explicitly desired
             if source != target:
                G.add_edge(source, target)
                valid_edge_count += 1
             else:
                # st.info(f"Skipping self-loop edge: {source} -> {target}")
                skipped_edge_count += 1
        elif source and target:
             # This warning can be noisy if AI hallucinates targets; maybe log instead
             print(f"DEBUG: Skipping edge: Source '{source}' or Target '{target}' not found in final unique terms list.")
             skipped_edge_count += 1
        else:
             st.warning(f"Skipping edge with missing source/target: {edge_data}")
             skipped_edge_count += 1

    st.info(f"Graph built: {len(G.nodes())} nodes, {valid_edge_count} edges. ({skipped_edge_count} edges skipped/self-loops).")
    return G

def find_cycles(G):
    """Finds simple cycles in a NetworkX DiGraph."""
    if G is None: return None
    try: return list(nx.simple_cycles(G))
    except Exception as e: print(f"Error finding cycles: {e}"); return None

def find_orphans(G):
    """Finds nodes with in-degree and out-degree of 0."""
    if G is None: return None
    return [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 0]

def get_neighbors(G, node_id):
    """Gets predecessors (pointing to node) and successors (node points to)."""
    if G is None or node_id not in G: return set(), set()
    # NetworkX DiGraph methods: predecessors() gives incoming, successors() gives outgoing
    return set(G.predecessors(node_id)), set(G.successors(node_id))

# --- Streamlit UI ---

# --- Header ---
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_FILE = "jasper-logo-1.png" # Make sure this filename matches your logo file
LOGO_PATH = os.path.join(APP_DIR, LOGO_FILE) if APP_DIR else LOGO_FILE # Handle if APP_DIR is empty

header_cols = st.columns([1, 5])
with header_cols[0]:
    # Check if the logo file exists relative to the script's directory first
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOCAL_LOGO_PATH = os.path.join(SCRIPT_DIR, LOGO_FILE)

    FINAL_LOGO_PATH = None
    if os.path.exists(LOCAL_LOGO_PATH):
        FINAL_LOGO_PATH = LOCAL_LOGO_PATH
    elif os.path.exists(LOGO_PATH): # Fallback to the potentially parent dir path
        FINAL_LOGO_PATH = LOGO_PATH

    if FINAL_LOGO_PATH and os.path.exists(FINAL_LOGO_PATH):
        try:
            logo = Image.open(FINAL_LOGO_PATH)
            st.image(logo, width=80, caption=None, output_format='PNG')
        except FileNotFoundError:
             st.warning(f"Logo file specified but not found at expected paths.")
        except Exception as img_err:
            st.warning(f"Could not load logo: {img_err}")
    else:
        st.write("") # Placeholder if no logo

with header_cols[1]:
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.caption("Upload a document, generate an interactive graph of defined terms, and analyze relationships.")
st.divider()


# --- Sidebar Controls ---
st.sidebar.title("Controls")
st.sidebar.markdown("---")
# API Key Input
api_key_input = st.sidebar.text_input(
    "Google AI Gemini API Key*", type="password", key="api_key_sidebar_dtg",
    value=st.session_state.api_key, # Use the initialized value
    help="Your Gemini API key. Get one from Google AI Studio."
)
# Update session state if the input changes
if api_key_input and api_key_input != st.session_state.api_key:
    st.session_state.api_key = api_key_input
    st.rerun() # Rerun to reflect key status change immediately

if not st.session_state.api_key:
    st.sidebar.warning("API Key required.", icon="🔑")

# File Uploader
st.sidebar.markdown("### 1. Upload Document")
uploaded_file_obj = st.sidebar.file_uploader(
    "Upload Document (PDF recommended)*", type=["pdf"], key="dtg_pdf_uploader"
)

# Process upload
if uploaded_file_obj is not None:
    new_bytes = uploaded_file_obj.getvalue()
    # Check if it's actually a new file or just a rerun
    if new_bytes != st.session_state.get('dtg_pdf_bytes') or uploaded_file_obj.name != st.session_state.get('dtg_pdf_name'):
        st.toast(f"📄 New file detected: '{uploaded_file_obj.name}'. Processing...", icon="🔄")

        # --- Reset analysis results from any previous file ---
        reset_dtg_analysis_state() # <<< FIX: Reset analysis state HERE

        # --- Store new file data ---
        st.session_state.dtg_pdf_bytes = new_bytes
        st.session_state.dtg_pdf_name = uploaded_file_obj.name
        st.session_state.dtg_extracted_text = None # Clear old text before extraction
        st.session_state.dtg_definitions_text = None
        st.session_state.dtg_error = None # Clear previous errors

        # --- Extract text from the NEW file ---
        if uploaded_file_obj.type == "application/pdf":
            full_text, def_text, error_msg = extract_text_from_pdf(st.session_state.dtg_pdf_bytes)
            st.session_state.dtg_extracted_text = full_text
            st.session_state.dtg_definitions_text = def_text # Store isolated text (or None)
            st.session_state.dtg_error = error_msg # Store potential extraction error
        else:
            st.session_state.dtg_error = f"Unsupported file type: {uploaded_file_obj.type}"
            st.session_state.dtg_extracted_text = None
            st.session_state.dtg_definitions_text = None

        if st.session_state.dtg_error:
             st.warning(f"Problem loading file: {st.session_state.dtg_error}")
        elif st.session_state.dtg_extracted_text or st.session_state.dtg_definitions_text:
            st.toast("Text extracted successfully.", icon="📝")
        else:
             # This case should be covered by error_msg, but just in case
             st.warning("File loaded, but failed to extract any text.")
             if not st.session_state.dtg_error: # Avoid duplicate message
                  st.session_state.dtg_error = "File loaded, but failed to extract any text."


        st.rerun() # Rerun to update UI after file processing

# Display persistent extraction errors if analysis hasn't started/finished
if st.session_state.dtg_error and not st.session_state.dtg_processing and not st.session_state.dtg_graph_data:
     st.error(f"Extraction Error: {st.session_state.dtg_error}")


# Generation Button
st.sidebar.markdown("### 2. Generate & Analyze")
text_available_for_analysis = st.session_state.dtg_definitions_text or st.session_state.dtg_extracted_text
can_generate = (st.session_state.api_key and
                st.session_state.dtg_pdf_bytes and
                text_available_for_analysis and
                not st.session_state.dtg_error and # Don't allow generation if extraction failed
                not st.session_state.dtg_processing)

generate_button_tooltip = ""
if st.session_state.dtg_processing: generate_button_tooltip = "Processing..."
elif not st.session_state.api_key: generate_button_tooltip = "Enter API Key"
elif not st.session_state.dtg_pdf_bytes: generate_button_tooltip = "Upload a document"
elif not text_available_for_analysis: generate_button_tooltip = "Could not extract text from document"
elif st.session_state.dtg_error: generate_button_tooltip = f"Cannot generate due to error: {st.session_state.dtg_error}"
else: generate_button_tooltip = "Generate graph and analyze term relationships using Gemini"

if st.sidebar.button("✨ Generate & Analyze Graph", key="dtg_generate", disabled=not can_generate, help=generate_button_tooltip, use_container_width=True, type="primary"):
    # --- Reset previous results before starting new analysis ---
    reset_dtg_analysis_state() # <<< FIX: Reset analysis state

    # --- Set processing flag ---
    st.session_state.dtg_processing = True
    st.rerun()


# Graph Interaction Controls
if st.session_state.dtg_graph_data:
    st.sidebar.markdown("---"); st.sidebar.markdown("### 3. Graph Interaction")
    # Filter input
    st.session_state.dtg_filter_term = st.sidebar.text_input(
        "Filter Nodes (by name)",
        value=st.session_state.dtg_filter_term,
        placeholder="Type term to filter...",
        key="dtg_filter_input"
    ).strip()

    # Node selection dropdown
    available_nodes = ["--- Select Node ---"]; current_highlight_index = 0
    if st.session_state.dtg_nx_graph:
         nodes_to_consider = list(st.session_state.dtg_nx_graph.nodes())
         if st.session_state.dtg_filter_term:
              try:
                  filter_regex = re.compile(st.session_state.dtg_filter_term, re.IGNORECASE)
                  nodes_to_consider = [n for n in nodes_to_consider if filter_regex.search(n)]
              except re.error:
                  st.sidebar.warning("Invalid filter regex.", icon="⚠️")
                  nodes_to_consider = [] # Show no nodes if filter is invalid
         available_nodes.extend(sorted(nodes_to_consider))

         # Preserve selection if possible after filtering
         if st.session_state.dtg_highlight_node and st.session_state.dtg_highlight_node in available_nodes:
              current_highlight_index = available_nodes.index(st.session_state.dtg_highlight_node)
         else:
              # If previous selection is filtered out or was null, reset
              st.session_state.dtg_highlight_node = None
              current_highlight_index = 0

    # Use a dynamic key based on filter to force selectbox refresh when filter changes
    highlight_key = f"highlight_select_{st.session_state.dtg_filter_term or 'all'}"
    selected_node = st.sidebar.selectbox(
        "Highlight Node & Neighbors",
        options=available_nodes,
        index=current_highlight_index,
        key=highlight_key, # Dynamic key
        help="Select node to highlight it and dependencies."
    )
    # Update state based on selection
    if selected_node != st.session_state.dtg_highlight_node:
         st.session_state.dtg_highlight_node = selected_node if selected_node != "--- Select Node ---" else None
         st.rerun() # Rerun to apply highlighting immediately


    # Layout selection
    st.session_state.dtg_layout = st.sidebar.radio(
        "Graph Layout",
        options=['Physics', 'Hierarchical'],
        index=0 if st.session_state.dtg_layout == 'Physics' else 1,
        key="dtg_layout_radio",
        help="Choose layout algorithm."
    )


# --- Main Area (Processing Logic) ---
if st.session_state.dtg_processing:
    status_placeholder = st.empty()
    progress_bar = st.progress(0, "Initializing analysis...")
    all_extracted_terms = [] # Collect terms from all chunks
    final_edges_list = []
    encountered_errors = []
    partial_results_generated = False

    try:
        if not st.session_state.api_key:
            raise ValueError("API Key is missing.")
        genai.configure(api_key=st.session_state.api_key)
        # Specify API version if needed, though usually handled by library
        # model = genai.GenerativeModel(MODEL_NAME, client_options={"api_version": "v1beta"})
        model = genai.GenerativeModel(MODEL_NAME)

        # Use isolated definitions text if available, otherwise fall back to full text
        analysis_text = st.session_state.dtg_definitions_text or st.session_state.dtg_extracted_text
        if not analysis_text:
            raise ValueError("No text available for analysis (extraction might have failed).")

        # --- Step 1: Chunk Text and Extract Terms/Definitions ---
        status_placeholder.info("💾 Chunking document text for analysis...")
        text_chunks = chunk_text(analysis_text, TERM_CHUNK_SIZE, TERM_CHUNK_OVERLAP)
        num_chunks = len(text_chunks)
        st.info(f"Split text into {num_chunks} chunk(s) for term extraction.")
        progress_bar.progress(5, text=f"Starting term extraction ({num_chunks} chunks)...")

        if num_chunks == 0:
             raise ValueError("Text chunking resulted in zero chunks. Check TERM_CHUNK_SIZE and input text.")

        # Common settings for term extraction calls
        terms_prompt_template = """
Your task is to analyze the provided text chunk, focusing ONLY on formal definitions (e.g., terms in quotes followed by "means...").

**Output Format:** Produce a single JSON object with ONLY ONE key: "terms".
1.  `"terms"`: A list of JSON objects. Each object must have:
    *   `"name"`: The exact defined term (string), accurately capturing quotes if part of the definition marker (e.g., `"Term"`). Normalize whitespace within the name.
    *   `"definition"`: The complete definition text associated with that term (string). Normalize whitespace and ensure the full definition is captured.

**Instructions for Extraction:**

*   **Focus:** Strictly analyze the provided text chunk for explicit definitions.
*   **Identify Defined Terms:** Only include terms formally defined within this chunk (typically patterns like `"Term" means...`, `Term. Means...`, etc.). Include ALL such terms found in the "terms" list.
*   **Extract Definitions:** Capture the full definition text, handling multi-line definitions correctly.
*   **Exclusions:** Do NOT include terms merely used but not defined. Do not list clause numbers, section numbers, dates, amounts, party names (unless explicitly defined) as term names. Only include terms explicitly defined in *this* chunk.
*   **Accuracy:** Be precise with the term name and its corresponding definition.

**Document Text Chunk:**
--- Start Chunk ---
{text_chunk}
--- End Chunk ---

**Final Output (Valid JSON Object with 'terms' key ONLY):**
"""
        terms_generation_config = types.GenerationConfig(response_mime_type="application/json", temperature=0.1)
        safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]]

        processed_chunk_count = 0
        for i, chunk in enumerate(text_chunks):
            chunk_num = i + 1
            status_placeholder.info(f"🧠 Requesting terms from chunk {chunk_num}/{num_chunks}...")
            # Progress: 5% to 50% allocated for term extraction
            progress = 5 + int(45 * (chunk_num / num_chunks))
            progress_bar.progress(progress, text=f"Step 1: Extracting Terms (Chunk {chunk_num}/{num_chunks})...")

            # Prepare prompt for the current chunk
            chunk_prompt = terms_prompt_template.format(text_chunk=chunk)

            try:
                response_terms_chunk = model.generate_content(
                    contents=chunk_prompt,
                    generation_config=terms_generation_config,
                    safety_settings=safety_settings,
                    request_options={'timeout': 600} # 10 min timeout per chunk
                )

                # Access response content safely
                response_text_content = ""
                try:
                     # Handle potential lack of 'text' attribute or parts
                     if response_terms_chunk.parts:
                         response_text_content = "".join(part.text for part in response_terms_chunk.parts if hasattr(part, 'text'))
                     elif hasattr(response_terms_chunk, 'text'):
                         response_text_content = response_terms_chunk.text
                     else: # Fallback if structure is unexpected
                          response_text_content = str(response_terms_chunk) # Or handle as error
                          print(f"WARN: Unexpected response structure for chunk {chunk_num}: {response_terms_chunk}")

                except (ValueError, AttributeError, types.BlockedPromptException, types.StopCandidateException) as resp_err:
                     warn_msg = f"Term Extraction Response Error (Chunk {chunk_num}): {type(resp_err).__name__} - {resp_err}. Skipping chunk."
                     encountered_errors.append(warn_msg); st.warning(f"⚠️ {warn_msg}")
                     print(f"WARN: Skipping chunk {chunk_num} due to response error: {resp_err}")
                     continue # Skip this chunk

                terms_data_chunk, term_error_msg, terms_truncated = parse_terms_response(response_text_content)

                if term_error_msg:
                    warn_msg = f"Term Parsing Warning (Chunk {chunk_num}): {term_error_msg}"
                    encountered_errors.append(warn_msg); st.warning(f"⚠️ {warn_msg}")
                    # Continue to next chunk even if parsing fails for one

                if terms_data_chunk and terms_data_chunk.get("terms"):
                    extracted_count = len(terms_data_chunk["terms"])
                    print(f"DEBUG: Chunk {chunk_num} - Extracted {extracted_count} terms.")
                    all_extracted_terms.extend(terms_data_chunk["terms"])
                    processed_chunk_count += 1
                    if terms_truncated:
                        partial_results_generated = True
                        st.session_state.dtg_partial_results = True # Persist partial flag
                        st.warning(f"⚠️ Terms list from chunk {chunk_num} might be incomplete (truncated response).")

                # Add delay between chunk calls
                time.sleep(TERM_EXTRACTION_CHUNK_DELAY)

            except google.api_core.exceptions.DeadlineExceeded as de:
                # MODIFIED: Log warning and continue instead of raising fatal error
                err_msg = f"API Timeout Error (Chunk {chunk_num}): {de}. Results from this chunk will be skipped. Consider reducing TERM_CHUNK_SIZE further or check network."
                encountered_errors.append(err_msg); st.warning(f"⚠️ {err_msg}")
                print(f"WARNING: DeadlineExceeded on chunk {chunk_num}, skipping.")
                # Continue to the next chunk
                time.sleep(TERM_EXTRACTION_CHUNK_DELAY) # Still apply delay
                continue # Skip the rest of the loop iteration for this failed chunk
            except (types.StopCandidateException, google.api_core.exceptions.GoogleAPIError) as api_err:
                 # Handle specific API errors more gracefully if needed
                 err_msg = f"API Error (Chunk {chunk_num}): {type(api_err).__name__}: {api_err}"
                 # Check for common issues like invalid API key or quota
                 if isinstance(api_err, google.api_core.exceptions.PermissionDenied):
                      st.session_state.dtg_error = f"Fatal API Error: Permission Denied. Check your API key. ({api_err})"
                      raise api_err # Stop processing
                 elif isinstance(api_err, google.api_core.exceptions.ResourceExhausted):
                      st.session_state.dtg_error = f"Fatal API Error: Quota Exceeded. ({api_err})"
                      raise api_err # Stop processing
                 else:
                      encountered_errors.append(err_msg); st.warning(f"⚠️ {err_msg}")
                      print(f"WARN: API Error during term chunk processing: {traceback.format_exc()}")
                      # Continue for most other API errors but log them.
            except Exception as chunk_err:
                 err_msg = f"Unexpected Error processing chunk {chunk_num}: {chunk_err}"
                 encountered_errors.append(err_msg); st.warning(f"⚠️ {err_msg}")
                 print(f"ERROR: Unexpected error during term chunk processing: {traceback.format_exc()}")
                 # Continue processing other chunks


        # --- Deduplicate Terms ---
        status_placeholder.info("Merging and deduplicating extracted terms...")
        progress_bar.progress(55, text="Deduplicating terms...")
        final_terms_map = {}
        processed_names = set()
        duplicates_found = 0
        for term_obj in all_extracted_terms:
             name = term_obj.get("name")
             definition = term_obj.get("definition")
             if name:
                 # Normalize name slightly for comparison (lowercase, strip extra spaces)
                 normalized_name = ' '.join(name.lower().split())
                 if normalized_name not in processed_names:
                      final_terms_map[name] = definition # Store with original casing
                      processed_names.add(normalized_name)
                 else:
                      duplicates_found += 1

        final_terms_list = [{"name": name, "definition": definition} for name, definition in final_terms_map.items()]
        st.session_state.dtg_total_terms = len(final_terms_list)
        st.session_state.dtg_terms_processed = 0 # Reset for edge processing step
        st.toast(f"✅ Step 1 Complete: Found {st.session_state.dtg_total_terms} unique terms ({duplicates_found} duplicates discarded) from {processed_chunk_count}/{num_chunks} processed chunk(s).", icon="📝")

        # --- Step 2: Extract Edges (Per Term) ---
        if st.session_state.dtg_total_terms > 0:
            status_placeholder.info(f"🧠 Identifying relationships for {st.session_state.dtg_total_terms} unique terms...")
            valid_term_names = list(final_terms_map.keys()) # Get names from deduplicated map
            # Create a case-insensitive mapping for lookup if needed, but store original case
            valid_term_names_lower = {name.lower(): name for name in valid_term_names}
            # Provide the model with the original-cased names for exact matching output
            valid_term_names_json = json.dumps(valid_term_names, indent=2)


            edge_generation_config = types.GenerationConfig(response_mime_type="application/json", temperature=0.0)

            for i, term_obj in enumerate(final_terms_list):
                source_term_name = term_obj['name']
                definition_text = term_obj['definition']
                st.session_state.dtg_terms_processed = i + 1
                # Progress: 55% to 95% allocated for edge extraction
                progress = 55 + int(40 * (st.session_state.dtg_terms_processed / st.session_state.dtg_total_terms))
                progress_bar.progress(progress, text=f"Step 2: Analyzing '{source_term_name[:30]}...' ({st.session_state.dtg_terms_processed}/{st.session_state.dtg_total_terms})...")

                if not definition_text:
                    print(f"DEBUG: Skipping edge analysis for '{source_term_name}': Definition text is empty.")
                    continue

                # Edge prompt - explicitly ask for exact matches from the list
                edge_prompt = f"""
Context: I am analyzing relationships between formally defined terms in a legal document.
Task: Examine the provided DEFINITION TEXT for the term "{source_term_name}". Identify which of the VALID DEFINED TERMS listed below are explicitly mentioned *within* this definition text. The match must be exact and case-sensitive based on the provided list.

VALID DEFINED TERMS (Case-Sensitive List):
{valid_term_names_json}

DEFINITION TEXT for "{source_term_name}":
---
{definition_text}
---

Output Format: Return ONLY a JSON list of strings. Each string in the list MUST be an EXACT match (case-sensitive) from the VALID DEFINED TERMS list that was found within the DEFINITION TEXT.
Return an empty list `[]` if none of the VALID DEFINED TERMS are found in the definition.
Example Output: ["Bank Account", "Loan Agreement", "Security Document"]
"""
                try:
                    # API call for edge extraction
                    response_edge = model.generate_content(
                        contents=edge_prompt,
                        generation_config=edge_generation_config,
                        safety_settings=safety_settings,
                        request_options={'timeout': 120} # Shorter timeout per edge is reasonable
                    )

                    # Safely access response text
                    edge_response_text = ""
                    try:
                        if response_edge.parts:
                           edge_response_text = "".join(part.text for part in response_edge.parts if hasattr(part, 'text'))
                        elif hasattr(response_edge, 'text'):
                           edge_response_text = response_edge.text
                        else:
                           print(f"WARN: Unexpected edge response structure for '{source_term_name}': {response_edge}")
                    except (ValueError, AttributeError, types.BlockedPromptException, types.StopCandidateException) as resp_err:
                         warn_msg = f"Edge Extraction Response Error ('{source_term_name}'): {type(resp_err).__name__} - {resp_err}. Skipping edges for this term."
                         encountered_errors.append(warn_msg); st.warning(f"⚠️ {warn_msg}")
                         print(f"WARN: Skipping edges for {source_term_name} due to response error: {resp_err}")
                         continue # Skip edges for this term


                    target_terms_list, edge_error_msg, edge_truncated = parse_edge_list_response(edge_response_text, source_term_name)

                    # Error handling and edge list building
                    if edge_error_msg:
                        warn_msg = f"Edge Parsing Warning ('{source_term_name}'): {edge_error_msg}"
                        encountered_errors.append(warn_msg); st.warning(f"⚠️ {warn_msg}")

                    if edge_truncated:
                         partial_results_generated = True
                         st.session_state.dtg_partial_results = True # Persist flag
                         st.warning(f"⚠️ Edge list for '{source_term_name}' might be incomplete (truncated response).")

                    if target_terms_list is not None: # Check if parsing returned a list (even empty) or None (error)
                        added_edge_count = 0
                        for target_term in target_terms_list:
                            # Ensure the target term exactly matches one of the known defined terms (case-sensitive)
                            if target_term in final_terms_map: # Check against the keys of our definitive map
                                final_edges_list.append({"source": source_term_name, "target": target_term})
                                added_edge_count += 1
                            else:
                               print(f"DEBUG: AI suggested edge '{source_term_name}' -> '{target_term}', but target is not in the exact valid terms list. Skipping.")

                        # print(f"DEBUG: Found {added_edge_count} valid edges for '{source_term_name}'.")


                except types.StopCandidateException as sce:
                    err_msg = f"Generation Stopped for '{source_term_name}' edges: {sce}."; encountered_errors.append(err_msg); st.warning(f"⚠️ {err_msg}")
                except google.api_core.exceptions.DeadlineExceeded as de:
                     # Log timeout for edge extraction but continue
                     err_msg = f"API Timeout Error for '{source_term_name}' edges: {de}."
                     encountered_errors.append(err_msg); st.warning(f"⚠️ {err_msg}")
                     print(f"WARN: DeadlineExceeded on edge extraction for {source_term_name}")
                except google.api_core.exceptions.GoogleAPIError as api_err:
                     # Treat API errors during edge extraction as warnings usually, unless critical
                     err_msg = f"API Error for '{source_term_name}' edges: {api_err}."; encountered_errors.append(err_msg); st.warning(f"⚠️ {err_msg}")
                     print(f"WARN: API error during edge extraction for {source_term_name}: {traceback.format_exc()}")
                except Exception as e:
                     err_msg = f"Processing Error for '{source_term_name}' edges: {e}"; encountered_errors.append(err_msg); st.warning(f"⚠️ {err_msg}")
                     print(f"ERROR: Unexpected error during edge extraction: {traceback.format_exc()}")

                time.sleep(EDGE_EXTRACTION_DELAY) # Delay between edge calls
        else:
            status_placeholder.info("No unique terms found, skipping edge extraction.")
            progress_bar.progress(95, text="Skipping edge extraction.")


        # --- Step 3: Combine and Finalize ---
        status_placeholder.info("Combining results and analyzing graph...")
        progress_bar.progress(98, text="Building graph...")
        st.session_state.dtg_graph_data = {
            "terms": final_terms_list,
            "edges": final_edges_list
        }

        # Perform graph analysis
        st.session_state.dtg_nx_graph = build_networkx_graph(st.session_state.dtg_graph_data)
        if st.session_state.dtg_nx_graph:
            st.session_state.dtg_cycles = find_cycles(st.session_state.dtg_nx_graph)
            st.session_state.dtg_orphans = find_orphans(st.session_state.dtg_nx_graph)
            st.toast("Graph analysis complete.", icon="🔬")
        else:
            st.warning("Could not build internal graph for analysis.")
            encountered_errors.append("Failed to build NetworkX graph from extracted data.")

        # --- Final Status Update ---
        if encountered_errors:
            st.session_state.dtg_error = "Processing completed with errors/warnings. Results might be incomplete or inaccurate."
            # Display errors prominently if they occurred
        elif partial_results_generated:
             st.session_state.dtg_error = "Processing completed, but results might be incomplete due to AI response truncation."
             # Display a warning
        else:
            st.toast("✅ Analysis successful!", icon="🎉")

        progress_bar.progress(100, text="Processing Complete.")
        time.sleep(2) # Keep completion message visible briefly


    # --- Exception Handling (Outer Try Block - includes ValueErrors from setup/chunking) ---
    except ValueError as ve:
        # User-facing errors (e.g., no text, no API key)
        st.session_state.dtg_error = f"Processing Setup Error: {ve}"
        st.error(f"❌ Fatal Setup Error: {ve}")
        print(f"ERROR: ValueError during processing setup: {traceback.format_exc()}")
    except google.api_core.exceptions.GoogleAPIError as api_err: # Catch critical API errors early
        # Fatal API errors caught outside the loops (e.g., Auth, Quota during first call)
        st.session_state.dtg_error = f"Fatal Google API Error: {type(api_err).__name__}. Check key/quota/permissions. ({api_err})"
        st.error(f"❌ {st.session_state.dtg_error}")
        print(f"ERROR: Fatal GoogleAPIError: {traceback.format_exc()}")
    except Exception as e: # Catch any other unexpected errors
        st.session_state.dtg_error = f"Unexpected Processing Error: {type(e).__name__} - {e}"
        st.error(f"❌ An unexpected error occurred: {st.session_state.dtg_error}")
        print(f"ERROR: Unexpected exception during processing: {traceback.format_exc()}")
    finally:
        # Always ensure processing flag is turned off and UI elements removed
        st.session_state.dtg_processing = False
        status_placeholder.empty()
        progress_bar.empty()
        # Display accumulated errors/warnings if any occurred
        if encountered_errors:
             with st.expander("⚠️ Processing Errors/Warnings Encountered", expanded=True):
                  for err in encountered_errors: st.warning(err)
        if st.session_state.dtg_partial_results and not any("truncation" in err for err in encountered_errors):
             st.warning("Additionally, some data might be missing due to AI response truncation (check warnings above).")

        st.rerun()


# --- Display Results ---
elif st.session_state.dtg_graph_data:
    st.subheader(f"📊 Interactive Graph & Analysis for '{st.session_state.dtg_pdf_name}'")
    if st.session_state.get('dtg_partial_results', False):
        st.warning("⚠️ Results may be incomplete due to AI response truncation during generation.", icon="⚠️")
    if st.session_state.dtg_error: # Display any errors that occurred during processing
        st.error(f"Processing completed with issues: {st.session_state.dtg_error}")


    graph_data = st.session_state.dtg_graph_data; G = st.session_state.dtg_nx_graph
    terms_map = {term['name']: term['definition'] for term in graph_data.get('terms', [])}
    filter_term = st.session_state.dtg_filter_term; highlight_node = st.session_state.dtg_highlight_node

    # Filter nodes/edges based on UI controls
    nodes_to_display_names = set(G.nodes()) if G else set()
    if filter_term:
        try:
            filter_regex = re.compile(filter_term, re.IGNORECASE)
            nodes_to_display_names = {n for n in G.nodes() if filter_regex.search(n)}
        except re.error:
            # Handle invalid regex in the filter input itself (sidebar)
            nodes_to_display_names = set() # Show nothing if filter invalid

    highlight_neighbors_predecessors, highlight_neighbors_successors = set(), set()
    if highlight_node and G and highlight_node in G:
        highlight_neighbors_predecessors, highlight_neighbors_successors = get_neighbors(G, highlight_node)

    # Prepare Agraph Nodes & Edges for the current view
    agraph_nodes = []; agraph_edges = []; agraph_edges_tuples = set()
    displayed_node_ids = set()

    if G:
        for node_id in G.nodes():
            if node_id not in nodes_to_display_names: continue # Apply filter

            displayed_node_ids.add(node_id);
            node_color = DEFAULT_NODE_COLOR;
            node_size = 15
            if node_id == highlight_node:
                node_color = HIGHLIGHT_COLOR; node_size = 25
            elif node_id in highlight_neighbors_predecessors or node_id in highlight_neighbors_successors:
                node_color = NEIGHBOR_COLOR; node_size = 20

            # Ensure label doesn't break Agraph if it contains problematic characters?
            # Basic sanitization: replace newline/tab with space
            safe_label = str(node_id).replace('\n', ' ').replace('\t', ' ')
            agraph_nodes.append(Node(id=node_id, label=safe_label, color=node_color, size=node_size, font={'color': "#000000"}))

        for u, v in G.edges():
            # Only include edges where both source and target are in the currently displayed set
            if u in displayed_node_ids and v in displayed_node_ids:
                 # Avoid duplicate edges in tuple list for DOT export
                 if (u, v) not in agraph_edges_tuples:
                     agraph_edges_tuples.add((u, v))
                     agraph_edges.append(Edge(source=u, target=v, color="#CCCCCC"))

    # Configure Agraph
    is_physics = st.session_state.dtg_layout == 'Physics'
    # Adjusted physics for potentially larger graphs
    physics_config = {
        'barnesHut': {
            'gravitationalConstant': -20000, # Increase repulsion
            'centralGravity': 0.15,         # Slightly stronger pull to center
            'springLength': 250,           # Longer preferred edge length
            'springConstant': 0.04,
            'damping': 0.09,
            'avoidOverlap': 0.2            # Stronger overlap avoidance
        },
        'minVelocity': 0.75,
        'maxVelocity': 50
    } if is_physics else {}

    hierarchical_config = {
        'enabled': (not is_physics),
        'sortMethod': 'directed', # top-down
        'levelSeparation': 200,  # Increase vertical separation
        'nodeSpacing': 150,      # Increase horizontal separation
        'treeSpacing': 250
    } if not is_physics else {}

    config = Config(
        width='100%',
        height=700,
        directed=True,
        hierarchical=not is_physics,
        node={'labelProperty':'label', 'size': 15, 'font': {'size': 12}}, # Slightly smaller default font
        edge={'smooth': {'type': 'cubicBezier', 'forceDirection': 'vertical', 'roundness': 0.4} if not is_physics else True}, # Smoother hierarchical edges
        layout={'hierarchical': hierarchical_config},
        physics=physics_config,
        interaction={'navigationButtons': True, 'keyboard': True, 'tooltipDelay': 300, 'hover': True, 'zoomView': True, 'dragView': True},
        # Ensure selection color matches highlight color if needed
        # manipulation=False # Disable manual node dragging/editing if desired
    )


    # Display Area
    graph_col, info_col = st.columns([3, 1])

    with graph_col:
        st.caption("Graph View: Click/drag to pan, scroll/pinch to zoom. Select node in sidebar to highlight.")
        if not agraph_nodes and filter_term:
            st.warning(f"No nodes match filter: '{filter_term}'")
        elif not agraph_nodes:
            st.info("No terms were extracted or matched the filter.")
        elif G is None:
            st.error("Graph object (G) is None, cannot render.")
        else:
             try:
                 # agraph component call
                 agraph_return = agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)
                 # Optional: Handle click events if needed by inspecting agraph_return
                 # if agraph_return and agraph_return != st.session_state.dtg_highlight_node:
                 #     st.session_state.dtg_highlight_node = agraph_return
                 #     st.rerun() # Rerun if click changes selection

             except Exception as agraph_err:
                 st.error(f"Error rendering graph component: {agraph_err}")
                 print(traceback.format_exc())


    with info_col:
        st.subheader("Details & Analysis")
        st.markdown("**Selected Definition:**")
        selected_def = terms_map.get(highlight_node, "_Select node in sidebar or graph_")
        # Use markdown for better formatting control if definition contains markdown
        # st.markdown(f"<div class='definition-box'>{selected_def}</div>", unsafe_allow_html=True)
        # Or stick to text_area if plain text is guaranteed/preferred
        st.text_area("Definition Display", value=selected_def, height=150, disabled=True, label_visibility="collapsed", key="def_display_box")


        st.markdown("---")
        st.markdown("**Graph Analysis:**")
        if G is None:
             st.warning("Analysis unavailable (Graph not built).")
        else:
             if st.session_state.dtg_cycles is not None:
                  num_cycles = len(st.session_state.dtg_cycles)
                  if num_cycles > 0:
                       with st.expander(f"🚨 Found {num_cycles} Circular Definition(s)", expanded=False):
                            for i, cycle in enumerate(st.session_state.dtg_cycles):
                                st.markdown(f"- Cycle {i+1}: `{' → '.join(cycle)} → {cycle[0]}`")
                  else: st.caption("✅ No circular definitions detected.")
             else: st.caption(" Cycle analysis failed or not run.")


             if st.session_state.dtg_orphans is not None:
                  num_orphans = len(st.session_state.dtg_orphans)
                  # Filter orphans based on current node filter
                  displayed_orphans = [o for o in st.session_state.dtg_orphans if o in displayed_node_ids]
                  num_displayed_orphans = len(displayed_orphans)

                  if num_displayed_orphans > 0:
                       expander_label = f"⚠️ Found {num_displayed_orphans} Orphan Term(s)"
                       if filter_term: expander_label += " (matching filter)"
                       with st.expander(expander_label, expanded=False):
                            st.markdown(f"`{', '.join(sorted(displayed_orphans))}`")
                            st.caption("_Defined but not linked within definition network (in current view)._")
                  elif num_orphans > 0 and not filter_term: # Only show if no filter applied
                       st.caption("✅ All defined terms linked (in current view).")
                  elif num_orphans == 0:
                        st.caption("✅ All defined terms linked.")

             else: st.caption(" Orphan analysis failed or not run.")


    st.divider()
    # Generate DOT Code & Downloads
    st.subheader("Export Graph (Current View)")
    export_cols = st.columns(4)
    safe_filename_base = re.sub(r'[^\w\-]+', '_', st.session_state.dtg_pdf_name or "graph").strip('_')

    # Build DOT code based on the currently displayed nodes/edges
    dot_lines = ["digraph G {"];
    # Optional: Add graph attributes like layout engine hint
    # dot_lines.append('  layout=dot; rankdir=TB;') # Example for hierarchical top-bottom
    node_style_map = {node.id: f'[label="{node.label}", color="{node.color}", fontcolor="#000000", fontsize=10, width=1.5, height=0.5, shape=box]' for node in agraph_nodes} # Example styling

    for node_id in sorted(list(displayed_node_ids)):
        style = node_style_map.get(node_id, f'[label="{node_id}"]') # Basic label if no style mapped
        # Quote node IDs if they contain spaces or special characters
        quoted_node_id = f'"{node_id}"' if re.search(r'[\s"\'\[\]\{\}\(\)\<\>\\#%;,]', node_id) else node_id
        dot_lines.append(f'  {quoted_node_id} {style};')

    # Use the agraph_edges_tuples which is already filtered
    for u, v in sorted(list(agraph_edges_tuples)):
        quoted_u = f'"{u}"' if re.search(r'[\s"\'\[\]\{\}\(\)\<\>\\#%;,]', u) else u
        quoted_v = f'"{v}"' if re.search(r'[\s"\'\[\]\{\}\(\)\<\>\\#%;,]', v) else v
        dot_lines.append(f'  {quoted_u} -> {quoted_v};')

    dot_lines.append("}")
    generated_dot_code = "\n".join(dot_lines)

    # Download Buttons
    with export_cols[0]:
        export_cols[0].download_button(label="📥 DOT Code (.dot)", data=generated_dot_code, file_name=f"{safe_filename_base}_graph.dot", mime="text/vnd.graphviz", use_container_width=True)

    with export_cols[1]:
         try:
             # Ensure graphviz executable is in PATH or handle error
             g_render = graphviz.Source(generated_dot_code)
             png_bytes = g_render.pipe(format='png')
             export_cols[1].download_button(label="🖼️ PNG Image (.png)", data=png_bytes, file_name=f"{safe_filename_base}_graph.png", mime="image/png", use_container_width=True)
         except graphviz.backend.execute.ExecutableNotFound:
             export_cols[1].warning("Install Graphviz (dot executable) for PNG export.", icon="⚠️")
         except Exception as render_err:
             export_cols[1].warning(f"PNG ERR: {render_err}", icon="⚠️")
             print(f"ERROR: PNG Rendering Error: {traceback.format_exc()}")

    with export_cols[2]:
         try:
             # Ensure graphviz executable is in PATH
             g_render_svg = graphviz.Source(generated_dot_code)
             svg_bytes = g_render_svg.pipe(format='svg')
             export_cols[2].download_button(label="📐 SVG Image (.svg)", data=svg_bytes, file_name=f"{safe_filename_base}_graph.svg", mime="image/svg+xml", use_container_width=True)
         except graphviz.backend.execute.ExecutableNotFound:
             export_cols[2].warning("Install Graphviz (dot executable) for SVG export.", icon="⚠️")
         except Exception as render_err:
             export_cols[2].warning(f"SVG ERR: {render_err}", icon="⚠️")
             print(f"ERROR: SVG Rendering Error: {traceback.format_exc()}")

    with export_cols[3]:
        if G:
            try:
                 # Export only the dependencies visible in the current filtered view
                 dep_list = [{"Source Term": u, "Depends On (Target Term)": v} for u, v in sorted(list(agraph_edges_tuples))]
                 if dep_list:
                     df_deps = pd.DataFrame(dep_list)
                 else: # Create empty dataframe with correct columns if no edges displayed
                     df_deps = pd.DataFrame(columns=["Source Term", "Depends On (Target Term)"])

                 csv_output = df_deps.to_csv(index=False).encode('utf-8')
                 export_cols[3].download_button(label="📋 Dependencies (.csv)", data=csv_output, file_name=f"{safe_filename_base}_dependencies.csv", mime="text/csv", use_container_width=True)
            except Exception as csv_err:
                 export_cols[3].warning(f"CSV ERR: {csv_err}", icon="⚠️")
                 print(f"ERROR: CSV Export Error: {traceback.format_exc()}")
        else:
            export_cols[3].button("📋 Dependencies (.csv)", disabled=True, use_container_width=True, help="Graph not available for CSV export")


    with st.expander("View Generated DOT Code (for current view)"):
        st.code(generated_dot_code, language='dot')


# --- Fallback/Error/Initial State Messages ---
# These messages show when not processing and no results are ready yet.
elif not st.session_state.dtg_processing:
    if st.session_state.dtg_error: # Display error if one exists from previous steps
        st.error(f"❌ Error state: {st.session_state.dtg_error}")
    elif not st.session_state.dtg_pdf_bytes:
        st.info("⬆️ Upload a document (PDF) using the sidebar to get started.")
    elif not (st.session_state.dtg_extracted_text or st.session_state.dtg_definitions_text):
        st.warning("⚠️ Document uploaded, but no text could be extracted. Cannot proceed.")
    elif not st.session_state.api_key:
         st.info("🔑 Enter your Google AI Gemini API Key in the sidebar to enable analysis.")
    else:
        # Ready state
        st.info("⬆️ Ready to generate. Click the 'Generate & Analyze Graph' button in the sidebar.")

# Footer in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with Streamlit & Google Gemini")
st.sidebar.caption(f"Using model: {MODEL_NAME}")