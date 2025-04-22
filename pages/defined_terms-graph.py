# pages/defined_terms_graph.py
# --- COMPLETE FILE vX.Y+2 (Input Chunking for Term Extraction) ---

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
MODEL_NAME = "gemini-2.5-pro-preview-03-25" # DO NOT CHANGE - Per user request
PAGE_TITLE = "Defined Terms Relationship Grapher"
PAGE_ICON = "🔗"
DEFAULT_NODE_COLOR = "#ACDBC9" # Light greenish-teal
HIGHLIGHT_COLOR = "#FFA07A" # Light Salmon for selected node
NEIGHBOR_COLOR = "#ADD8E6" # Light Blue for neighbors
EDGE_EXTRACTION_DELAY = 0.5 # Seconds delay between edge extraction API calls
TERM_EXTRACTION_CHUNK_DELAY = 1.0 # Seconds delay between term chunk API calls
# Chunking Parameters (tune these based on testing and model limits)
# Gemini 1.5 Pro has a large context window, but output/processing limits might still apply.
# Start with a relatively large chunk size and reduce if timeouts persist.
# Using ~100k characters as a starting point, roughly 20-30 pages?
TERM_CHUNK_SIZE = 100000 # Target characters per chunk for term extraction
TERM_CHUNK_OVERLAP = 2000   # Characters overlap between chunks

# --- Set Page Config ---
st.set_page_config(layout="wide", page_title=PAGE_TITLE, page_icon=PAGE_ICON)

# --- Optional CSS ---
# (CSS remains the same)
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


# --- Helper Function for Text Extraction (Enhanced - same as previous version) ---
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
            # Added a simpler pattern looking just for Section 1 start/end
            re.compile(r"(?:Section 1|Clause 1|ARTICLE 1)\s*\.?\s*(Definitions|Interpretation)\s*\n(.*?)(?:^[ \t]*(?:Section 2|Clause 2|ARTICLE 2)|SIGNATURES|SCHEDULES)", re.IGNORECASE | re.MULTILINE | re.DOTALL),
        ]
        found_section = False
        for pattern in patterns:
            match = pattern.search(full_text)
            if match:
                # Group index depends on the pattern structure
                if len(match.groups()) >= 2 and pattern.pattern.count('(.*?)') >= 1:
                    definitions_text = match.group(match.re.groupindex.get('content', -1) if 'content' in match.re.groupindex else (len(match.groups()))).strip() # Adjust index based on pattern
                elif len(match.groups()) == 1:
                     definitions_text = match.group(1).strip()

                if definitions_text and re.search(r'"[^"]+"\s+means', definitions_text, re.IGNORECASE):
                    st.toast(f"ℹ️ Successfully isolated Definitions section ({len(definitions_text):,} chars).", icon="✂️")
                    print(f"DEBUG: Isolated definitions section, length: {len(definitions_text)}")
                    found_section = True
                    break
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
        # Try to end chunk at a natural break (newline) near the target size
        newline_pos = text.rfind('\n', start + chunk_size - overlap, end + overlap)
        if newline_pos > start + overlap: # Ensure newline is far enough from start
            end = newline_pos
        elif end > len(text): # Handle end of text
             end = len(text)

        chunks.append(text[start:end])
        # Move start for the next chunk, considering overlap
        next_start = end - overlap
        # Avoid getting stuck if overlap pushes back too far or chunk is tiny
        if next_start <= start:
            next_start = start + chunk_size # Force move forward if stuck

        start = next_start

    # Filter out potentially empty chunks resulting from edge cases
    return [c for c in chunks if c.strip()]


# --- Helper Function to Parse AI JSON Response (Terms Only - same as previous version) ---
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
            if response_text.strip().startswith("{"): # Check if it looks like JSON
                 json_text = response_text.strip()
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
                while True:
                    brace_pos = json_text.find('}', search_start)
                    if brace_pos == -1: break # No more closing braces
                    # Check if this brace is followed by a comma or the closing bracket of the list
                    next_char_pos = brace_pos + 1
                    while next_char_pos < len(json_text) and json_text[next_char_pos].isspace():
                        next_char_pos += 1
                    if next_char_pos < len(json_text) and json_text[next_char_pos] in [',', ']']:
                        last_obj_end = brace_pos # Found a potential complete object end
                    search_start = brace_pos + 1 # Continue search after this brace

                if last_obj_end != -1:
                    # Try to reconstruct a potentially valid partial JSON: {"terms": [...]}
                    terms_list_start = json_text.find('"terms":')
                    if terms_list_start != -1:
                        terms_list_start = json_text.find('[', terms_list_start)
                        if terms_list_start != -1:
                            potential_json_text = json_text[terms_list_start : last_obj_end + 1] # Slice from '[' to last '}'
                            potential_json = f'{{"terms": {potential_json_text}]}}' # Add outer structure + closing bracket
                            try:
                                data = json.loads(potential_json)
                                was_truncated = True
                                final_error_msg = None # Succeeded with fallback
                                st.warning("⚠️ Successfully recovered partial terms list from truncated response.", icon="✅")
                            except json.JSONDecodeError as inner_err:
                                print(f"DEBUG: Fallback parsing failed: {inner_err} on text: {potential_json}")
                                # If reconstruction failed, keep original error
                                final_error_msg = f"Failed to decode AI JSON response: {original_error}. Recovery attempt failed. Raw snippet: {raw_snippet}"
                        else: raise ValueError("Could not find start of terms list '[' for fallback.")
                    else: raise ValueError("Could not find '\"terms\":' key for fallback.")
                else: raise ValueError("Could not find a likely end '}' for any term object.")

            except Exception as fallback_err:
                st.warning(f"⚠️ JSON recovery attempt failed: {fallback_err}", icon="❌")
                final_error_msg = f"Failed to decode AI JSON response: {original_error}. Recovery attempt failed. Raw snippet: {raw_snippet}"

        # If data is still None or error persists after fallback, return the error
        if data is None or final_error_msg:
            return None, final_error_msg or "Unknown parsing error.", False

    # 3. Validate structure if parsing succeeded (fully or partially)
    try:
        if not isinstance(data, dict):
            return None, f"Parsed data is not a JSON object ({type(data).__name__}). Content: {str(data)[:200]}...", was_truncated
        if "terms" not in data or not isinstance(data["terms"], list):
            return None, f"Parsed data missing 'terms' list key. Keys: {list(data.keys())}", was_truncated

        validated_terms = []
        term_names_processed = set() # Keep track locally within validation
        for item in data["terms"]:
            if isinstance(item, dict) and "name" in item and "definition" in item and isinstance(item["name"], str) and isinstance(item["definition"], str):
                term_name = item["name"].strip()
                if term_name: # Allow processing even if already seen (for deduplication later)
                    validated_terms.append({"name": term_name, "definition": item["definition"]})
            # else: st.warning(f"Skipping malformed term item: {item}") # Optional

        if not validated_terms and not was_truncated:
             return None, "AI response contained no valid terms after validation.", was_truncated
        elif not validated_terms and was_truncated:
             st.warning("⚠️ No valid terms found in the recovered partial response.")
             return {"terms": []}, None, was_truncated # Return empty list but no error

        return {"terms": validated_terms}, None, was_truncated

    except Exception as e:
        return None, f"Error validating parsed terms structure: {e}", was_truncated


# --- Helper Function to Parse AI JSON Response (Edge List Only - same as previous version) ---
def parse_edge_list_response(response_text, source_term_name):
    """Parses the AI's JSON response specifically for a list of target term strings."""
    raw_snippet = response_text[:500] + ("..." if len(response_text) > 500 else "")
    json_text = None
    data = None # Initialize data
    was_truncated = False
    final_error_msg = None

    try:
        # 1. Attempt normal parsing
        match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE | re.DOTALL)
        json_text = match.group(1).strip() if match else response_text.strip()

        if not json_text or not json_text.startswith("["):
            # Check if it might be just a plain string or empty response instead of list JSON
            if not json_text.strip():
                return [], None, False # Treat empty response as valid empty list
            else:
                 # Maybe the AI just returned the term name(s) as plain text? Unlikely but possible.
                 # For now, treat non-list as error
                 return None, f"Edge response for '{source_term_name}' not a JSON list. Raw: {raw_snippet}", False

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
                         # Decide where to cut: either after last comma or after last full string if no comma follows
                         cut_pos = last_quote_end + 1 # Cut after the closing quote of the last potentially complete string
                         if last_comma > list_start and last_comma > json_text.rfind('"', 0, last_comma): # Ensure comma is after a string start
                              # If a comma exists *after* the start of the last string but *before* its end quote, likely truncated inside string
                              # In this case, try cutting *at* the last comma if it follows a closing quote of the *previous* string
                              prev_quote_end = json_text.rfind('"', 0, last_comma)
                              if prev_quote_end != -1: cut_pos = last_comma # Cut at the comma
                              # else: stick with cutting after last quote_end

                         potential_json = json_text[list_start : cut_pos] + ']' # Reconstruct list
                         data = json.loads(potential_json)
                         was_truncated = True
                         final_error_msg = None
                         st.warning(f"⚠️ Successfully recovered partial edge list for '{source_term_name}'.", icon="✅")
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
        return None, f"Error validating parsed edge list structure for '{source_term_name}': {e}", was_truncated


# --- Initialize Session State (Expanded - same as previous version) ---
# (initialize_dtg_state function remains the same)
initialize_dtg_state()

# --- Graph Analysis Functions (build_networkx_graph, find_cycles, find_orphans, get_neighbors - same) ---
# (These functions remain the same)
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
             G.add_edge(source, target)
             valid_edge_count += 1
        elif source and target:
             st.warning(f"Skipping edge: Source '{source}' or Target '{target}' not found in extracted terms list.")
             skipped_edge_count += 1
        else:
             st.warning(f"Skipping edge with missing source/target: {edge_data}")
             skipped_edge_count += 1

    st.info(f"Graph built: {len(G.nodes())} nodes, {valid_edge_count} edges. ({skipped_edge_count} edges skipped).")
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
    return set(G.predecessors(node_id)), set(G.successors(node_id))

# --- Streamlit UI ---

# --- Header ---
# (Header code remains the same)
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_FILE = "jasper-logo-1.png" # Make sure this filename matches your logo file
LOGO_PATH = os.path.join(APP_DIR, LOGO_FILE)
header_cols = st.columns([1, 5])
with header_cols[0]:
    if os.path.exists(LOGO_PATH):
        try:
            logo = Image.open(LOGO_PATH)
            st.image(logo, width=80, caption=None, output_format='PNG') # Removed alt
        except FileNotFoundError:
             st.warning(f"Logo file not found: {LOGO_PATH}") # More specific error
        except Exception as img_err:
            st.warning(f"Could not load logo: {img_err}")
with header_cols[1]:
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.caption("Upload a document, generate an interactive graph of defined terms, and analyze relationships.")
st.divider()


# --- Sidebar Controls ---
# (Sidebar code remains mostly the same, ensures text_available_for_analysis check uses correct variable)
st.sidebar.title("Controls")
st.sidebar.markdown("---")
# API Key Input
api_key_input = st.sidebar.text_input(
    "Google AI Gemini API Key*", type="password", key="api_key_sidebar_dtg",
    value=st.session_state.get("api_key", ""), help="Your Gemini API key."
)
if api_key_input and api_key_input != st.session_state.api_key:
    st.session_state.api_key = api_key_input
if not st.session_state.api_key:
    st.sidebar.warning("API Key required.", icon="🔑")
# File Uploader
st.sidebar.markdown("### 1. Upload Document")
uploaded_file_obj = st.sidebar.file_uploader(
    "Upload Document (PDF recommended)*", type=["pdf"], key="dtg_pdf_uploader"
)
# Process upload (reset includes new dtg_definitions_text state)
if uploaded_file_obj is not None:
    new_bytes = uploaded_file_obj.getvalue()
    if new_bytes != st.session_state.get('dtg_pdf_bytes') or uploaded_file_obj.name != st.session_state.get('dtg_pdf_name'):
        st.session_state.dtg_pdf_bytes = new_bytes
        st.session_state.dtg_pdf_name = uploaded_file_obj.name
        # Reset state related to previous file/analysis
        initialize_dtg_state() # Reset all DTG state to defaults (preserves API key)
        st.session_state.dtg_pdf_bytes = new_bytes # Re-assign new bytes/name
        st.session_state.dtg_pdf_name = uploaded_file_obj.name
        st.toast(f"📄 File '{st.session_state.dtg_pdf_name}' loaded.", icon="✅")

        # Extract text
        if uploaded_file_obj.type == "application/pdf":
            full_text, def_text, error_msg = extract_text_from_pdf(st.session_state.dtg_pdf_bytes)
            st.session_state.dtg_extracted_text = full_text
            st.session_state.dtg_definitions_text = def_text # Store isolated text (or None)
            st.session_state.dtg_error = error_msg
        else:
            st.session_state.dtg_error = f"Unsupported file type: {uploaded_file_obj.type}"

        if not st.session_state.dtg_error and (st.session_state.dtg_extracted_text or st.session_state.dtg_definitions_text):
            st.toast("Text extracted.", icon="📝")
        st.rerun()

if st.session_state.dtg_error and not st.session_state.dtg_processing and not st.session_state.dtg_graph_data:
     st.error(st.session_state.dtg_error)

# Generation Button
st.sidebar.markdown("### 2. Generate & Analyze")
text_available_for_analysis = st.session_state.dtg_definitions_text or st.session_state.dtg_extracted_text
can_generate = (st.session_state.api_key and
                st.session_state.dtg_pdf_bytes and
                text_available_for_analysis and
                not st.session_state.dtg_processing)

# (Generate button tooltip and action remains the same)
generate_button_tooltip = ""
if st.session_state.dtg_processing: generate_button_tooltip = "Processing..."
elif not st.session_state.api_key: generate_button_tooltip = "Enter API Key"
elif not st.session_state.dtg_pdf_bytes: generate_button_tooltip = "Upload a document"
elif not text_available_for_analysis: generate_button_tooltip = "Could not extract text from document"
else: generate_button_tooltip = "Generate graph and analyze term relationships using Gemini"

if st.sidebar.button("✨ Generate & Analyze Graph", key="dtg_generate", disabled=not can_generate, help=generate_button_tooltip, use_container_width=True, type="primary"):
    st.session_state.dtg_processing = True
    # Reset results and intermediate state
    initialize_dtg_state() # Reset fully except for PDF data and API key
    st.session_state.dtg_processing = True # Set processing flag again after reset
    st.rerun()

# Graph Interaction Controls (remains the same)
if st.session_state.dtg_graph_data:
    st.sidebar.markdown("---"); st.sidebar.markdown("### 3. Graph Interaction")
    # ...(interaction controls code is unchanged)...
    st.session_state.dtg_filter_term = st.sidebar.text_input("Filter Nodes (by name)", value=st.session_state.dtg_filter_term, placeholder="Type term to filter...", key="dtg_filter_input").strip()
    available_nodes = ["--- Select Node ---"]; current_highlight_index = 0
    if st.session_state.dtg_nx_graph:
         nodes_to_consider = list(st.session_state.dtg_nx_graph.nodes())
         if st.session_state.dtg_filter_term:
              try: filter_regex = re.compile(st.session_state.dtg_filter_term, re.IGNORECASE); nodes_to_consider = [n for n in nodes_to_consider if filter_regex.search(n)]
              except re.error: st.sidebar.warning("Invalid filter regex.", icon="⚠️"); nodes_to_consider = []
         available_nodes.extend(sorted(nodes_to_consider))
         if st.session_state.dtg_highlight_node and st.session_state.dtg_highlight_node in available_nodes:
              current_highlight_index = available_nodes.index(st.session_state.dtg_highlight_node)
         else: st.session_state.dtg_highlight_node = None

    highlight_key = f"highlight_select_{st.session_state.dtg_filter_term}"
    st.session_state.dtg_highlight_node = st.sidebar.selectbox(
        "Highlight Node & Neighbors", options=available_nodes, index=current_highlight_index,
        key=highlight_key, help="Select node to highlight it and dependencies."
    )
    if st.session_state.dtg_highlight_node == "--- Select Node ---": st.session_state.dtg_highlight_node = None
    st.session_state.dtg_layout = st.sidebar.radio("Graph Layout", options=['Physics', 'Hierarchical'], index=0 if st.session_state.dtg_layout == 'Physics' else 1, key="dtg_layout_radio", help="Choose layout algorithm.")


# --- Main Area (Processing Logic Heavily Modified) ---
if st.session_state.dtg_processing:
    status_placeholder = st.empty()
    progress_bar = st.progress(0, "Initializing analysis...")
    all_extracted_terms = [] # Collect terms from all chunks
    final_edges_list = []
    encountered_errors = []
    partial_results_generated = False

    try:
        genai.configure(api_key=st.session_state.api_key)
        model = genai.GenerativeModel(MODEL_NAME)
        # Use isolated definitions text if available, otherwise fall back to full text
        analysis_text = st.session_state.dtg_definitions_text or st.session_state.dtg_extracted_text
        if not analysis_text:
            raise ValueError("No text available for analysis.")

        # --- Step 1: Chunk Text and Extract Terms/Definitions ---
        status_placeholder.info("💾 Chunking document text for analysis...")
        text_chunks = chunk_text(analysis_text, TERM_CHUNK_SIZE, TERM_CHUNK_OVERLAP)
        num_chunks = len(text_chunks)
        st.info(f"Split text into {num_chunks} chunk(s) for term extraction.")
        progress_bar.progress(5, text=f"Starting term extraction ({num_chunks} chunks)...")

        if num_chunks == 0:
             raise ValueError("Text chunking resulted in zero chunks.")

        # Common settings for term extraction calls
        terms_prompt_template = """
Your task is to analyze the provided text chunk, focusing ONLY on formal definitions (e.g., terms in quotes followed by "means...").

**Output Format:** Produce a single JSON object with ONLY ONE key: "terms".
1.  `"terms"`: A list of JSON objects. Each object must have:
    *   `"name"`: The exact defined term (string), accurately capturing quotes if part of the definition marker.
    *   `"definition"`: The complete definition text associated with that term (string).

**Instructions for Extraction:**

*   **Focus:** Strictly analyze the provided text chunk for explicit definitions.
*   **Identify Defined Terms:** Only include terms formally defined within this chunk. Include ALL such terms found in the "terms" list.
*   **Extract Definitions:** Capture the full definition text. Handle potential multi-line definitions correctly.
*   **Exclusions:** Do NOT include terms merely used but not defined. Do not list clause numbers, section numbers, dates, amounts, party names (unless explicitly defined) as term names.

**Document Text Chunk:**
--- Start Chunk ---
{text_chunk}
--- End Chunk ---

**Final Output (Valid JSON Object with 'terms' key ONLY):**
"""
        terms_generation_config = types.GenerationConfig(response_mime_type="application/json", temperature=0.1)
        safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]]

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

                terms_data_chunk, term_error_msg, terms_truncated = parse_terms_response(response_terms_chunk.text)

                if term_error_msg:
                    warn_msg = f"Term Extraction Warning (Chunk {chunk_num}): {term_error_msg}"
                    encountered_errors.append(warn_msg); st.warning(f"⚠️ {warn_msg}")
                    # Continue to next chunk even if one fails, unless it's a fatal API error below

                if terms_data_chunk and terms_data_chunk.get("terms"):
                    all_extracted_terms.extend(terms_data_chunk["terms"])
                    if terms_truncated:
                        partial_results_generated = True
                        st.session_state.dtg_partial_results = True
                        st.warning(f"⚠️ Terms list from chunk {chunk_num} might be incomplete (truncated response).")

                # Add delay between chunk calls
                time.sleep(TERM_EXTRACTION_CHUNK_DELAY)

            except google.api_core.exceptions.DeadlineExceeded as de:
                err_msg = f"API Timeout Error (Chunk {chunk_num}): {de}. Consider reducing TERM_CHUNK_SIZE."
                encountered_errors.append(err_msg); st.error(f"❌ {err_msg}")
                # Decide whether to continue or stop? Let's stop for timeout.
                raise ValueError(f"Processing stopped due to timeout on chunk {chunk_num}.") from de
            except (types.StopCandidateException, google.api_core.exceptions.GoogleAPIError) as api_err:
                 # Handle specific API errors more gracefully if needed, otherwise re-raise or log
                 err_msg = f"API Error (Chunk {chunk_num}): {type(api_err).__name__}: {api_err}"
                 encountered_errors.append(err_msg); st.warning(f"⚠️ {err_msg}")
                 # Continue for most API errors unless it's auth/quota related?
                 # For now, let's continue but log the error.
                 print(f"ERROR during term chunk processing: {traceback.format_exc()}")
            except Exception as chunk_err:
                 err_msg = f"Unexpected Error processing chunk {chunk_num}: {chunk_err}"
                 encountered_errors.append(err_msg); st.warning(f"⚠️ {err_msg}")
                 print(f"ERROR during term chunk processing: {traceback.format_exc()}")
                 # Continue processing other chunks


        # --- Deduplicate Terms ---
        status_placeholder.info("Merging and deduplicating extracted terms...")
        progress_bar.progress(55, text="Deduplicating terms...")
        final_terms_map = {}
        for term_obj in all_extracted_terms:
             name = term_obj.get("name")
             definition = term_obj.get("definition")
             if name and name not in final_terms_map: # Keep first encountered definition
                  final_terms_map[name] = definition

        final_terms_list = [{"name": name, "definition": definition} for name, definition in final_terms_map.items()]
        st.session_state.dtg_total_terms = len(final_terms_list)
        st.session_state.dtg_terms_processed = 0 # Reset for edge processing step
        st.toast(f"✅ Step 1 Complete: Found {st.session_state.dtg_total_terms} unique terms across {num_chunks} chunk(s).", icon="📝")

        # --- Step 2: Extract Edges (Per Term - Logic largely unchanged) ---
        if st.session_state.dtg_total_terms > 0:
            status_placeholder.info(f"🧠 Identifying relationships for {st.session_state.dtg_total_terms} terms...")
            valid_term_names = list(final_terms_map.keys()) # Get names from deduplicated map
            valid_term_names_json = json.dumps(valid_term_names)

            edge_generation_config = types.GenerationConfig(response_mime_type="application/json", temperature=0.0)

            for i, term_obj in enumerate(final_terms_list):
                source_term_name = term_obj['name']
                definition_text = term_obj['definition']
                st.session_state.dtg_terms_processed = i + 1
                # Progress: 55% to 95% allocated for edge extraction
                progress = 55 + int(40 * (st.session_state.dtg_terms_processed / st.session_state.dtg_total_terms))
                progress_bar.progress(progress, text=f"Step 2: Analyzing '{source_term_name[:30]}...' ({st.session_state.dtg_terms_processed}/{st.session_state.dtg_total_terms})...")

                if not definition_text:
                    st.warning(f"Skipping edge analysis for '{source_term_name}': Definition text is empty.")
                    continue

                # (Edge prompt remains the same)
                edge_prompt = f"""
Context: I am analyzing relationships between formally defined terms in a legal document.
Task: Examine the provided DEFINITION TEXT for the term "{source_term_name}". Identify which of the VALID DEFINED TERMS listed below are explicitly mentioned *within* this definition text.

VALID DEFINED TERMS (Case-Sensitive):
{valid_term_names_json}

DEFINITION TEXT for "{source_term_name}":
---
{definition_text}
---

Output Format: Return ONLY a JSON list of strings. Each string in the list must be an EXACT match from the VALID DEFINED TERMS list that was found within the DEFINITION TEXT.
Return an empty list `[]` if none of the VALID DEFINED TERMS are found in the definition.
Example Output: ["Bank Account", "Loan Agreement", "Security Document"]
"""
                try:
                    # (API call for edge extraction remains the same)
                    response_edge = model.generate_content(
                        contents=edge_prompt,
                        generation_config=edge_generation_config,
                        safety_settings=safety_settings,
                        request_options={'timeout': 120} # Shorter timeout per edge
                    )

                    target_terms_list, edge_error_msg, edge_truncated = parse_edge_list_response(response_edge.text, source_term_name)

                    # (Error handling and edge list building remain the same)
                    if edge_error_msg:
                        warn_msg = f"Edge Extraction Warning ('{source_term_name}'): {edge_error_msg}"
                        encountered_errors.append(warn_msg); st.warning(f"⚠️ {warn_msg}")

                    if edge_truncated:
                         partial_results_generated = True
                         st.session_state.dtg_partial_results = True
                         st.warning(f"⚠️ Edge list for '{source_term_name}' might be incomplete (truncated response).")

                    if target_terms_list:
                        for target_term in target_terms_list:
                            if target_term in valid_term_names:
                                final_edges_list.append({"source": source_term_name, "target": target_term})
                            # else: # Optional: log skipped edges pointing outside the valid list
                            #    st.warning(f"AI suggested edge '{source_term_name}' -> '{target_term}', but target is not in the valid terms list. Skipping.")

                except types.StopCandidateException as sce:
                    err_msg = f"Generation Stopped for '{source_term_name}' edges: {sce}."; encountered_errors.append(err_msg); st.warning(f"⚠️ {err_msg}")
                except google.api_core.exceptions.DeadlineExceeded as de:
                     err_msg = f"API Timeout Error for '{source_term_name}' edges: {de}."
                     encountered_errors.append(err_msg); st.warning(f"⚠️ {err_msg}") # Log but continue
                except google.api_core.exceptions.GoogleAPIError as api_err:
                     err_msg = f"API Error for '{source_term_name}' edges: {api_err}."; encountered_errors.append(err_msg); st.warning(f"⚠️ {err_msg}")
                except Exception as e:
                     err_msg = f"Processing Error for '{source_term_name}' edges: {e}"; encountered_errors.append(err_msg); st.warning(f"⚠️ {err_msg}")
                     print(traceback.format_exc())

                time.sleep(EDGE_EXTRACTION_DELAY) # Delay between edge calls
        else:
            status_placeholder.info("No terms found, skipping edge extraction.")
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
            st.session_state.dtg_error = "Processing completed with errors/warnings. Results might be incomplete."
            with st.expander("⚠️ Processing Errors/Warnings Encountered", expanded=True):
                for err in encountered_errors: st.warning(err)
            if partial_results_generated: st.warning("Additionally, some data might be missing due to AI response truncation.")
        elif partial_results_generated:
             st.session_state.dtg_error = "Processing completed, but results might be incomplete due to AI response truncation."
             st.warning(st.session_state.dtg_error)
        else:
            st.toast("✅ Analysis successful!", icon="🎉")

        progress_bar.progress(100, text="Processing Complete.")
        time.sleep(2)


    # --- Exception Handling (Outer Try Block - includes ValueErrors from setup/chunking) ---
    except ValueError as ve:
        st.session_state.dtg_error = f"Processing Setup Error: {ve}"
        st.error(f"❌ Fatal Setup Error: {ve}")
        print(traceback.format_exc())
    except google.api_core.exceptions.GoogleAPIError as api_err: # Catch critical API errors early
        st.session_state.dtg_error = f"Fatal Google API Error: {api_err}. Check key/quota/permissions."
        st.error(f"❌ {st.session_state.dtg_error}")
        print(traceback.format_exc())
    except Exception as e: # Catch any other unexpected errors
        st.session_state.dtg_error = f"Unexpected Processing Error: {e}"
        st.error(f"❌ {st.session_state.dtg_error}")
        print(traceback.format_exc())
    finally:
        st.session_state.dtg_processing = False
        status_placeholder.empty()
        progress_bar.empty()
        st.rerun()


# --- Display Results (Code remains the same as previous version) ---
elif st.session_state.dtg_graph_data:
    # ...(Existing code for displaying results, graph, analysis, downloads)...
    st.subheader(f"📊 Interactive Graph & Analysis for '{st.session_state.dtg_pdf_name}'")
    if st.session_state.get('dtg_partial_results', False):
        st.warning("⚠️ Results may be incomplete due to AI response truncation during generation.", icon="⚠️")
    graph_data = st.session_state.dtg_graph_data; G = st.session_state.dtg_nx_graph
    terms_map = {term['name']: term['definition'] for term in graph_data.get('terms', [])}
    filter_term = st.session_state.dtg_filter_term; highlight_node = st.session_state.dtg_highlight_node
    # Filter nodes/edges
    nodes_to_display_names = set(G.nodes()) if G else set()
    if filter_term:
        try: filter_regex = re.compile(filter_term, re.IGNORECASE); nodes_to_display_names = {n for n in G.nodes() if filter_regex.search(n)}
        except re.error: st.warning("Invalid filter regex.", icon="⚠️"); nodes_to_display_names = set(G.nodes()) if G else set()
    highlight_neighbors_predecessors, highlight_neighbors_successors = set(), set()
    if highlight_node and G: highlight_neighbors_predecessors, highlight_neighbors_successors = get_neighbors(G, highlight_node)
    # Prepare Agraph Nodes & Edges
    agraph_nodes = []; agraph_edges = []; agraph_edges_tuples = []
    displayed_node_ids = set()
    if G:
        for node_id in G.nodes():
            if node_id not in nodes_to_display_names: continue
            displayed_node_ids.add(node_id); node_color = DEFAULT_NODE_COLOR; node_size = 15
            if node_id == highlight_node: node_color = HIGHLIGHT_COLOR; node_size = 25
            elif node_id in highlight_neighbors_predecessors or node_id in highlight_neighbors_successors: node_color = NEIGHBOR_COLOR; node_size = 20
            agraph_nodes.append(Node(id=node_id, label=node_id, color=node_color, size=node_size, font={'color': "#000000"}))
        for u, v in G.edges():
            if u in displayed_node_ids and v in displayed_node_ids:
                 agraph_edges_tuples.append((u, v))
                 agraph_edges.append(Edge(source=u, target=v, color="#CCCCCC"))
    # Configure Agraph
    is_physics = st.session_state.dtg_layout == 'Physics'
    config = Config(width='100%', height=700, directed=True, physics=is_physics, hierarchical=not is_physics, highlightColor=HIGHLIGHT_COLOR, collapsible=False, node={'labelProperty':'label', 'size': 15}, physics_config={'barnesHut': {'gravitationalConstant': -10000, 'centralGravity': 0.1, 'springLength': 180, 'springConstant': 0.05, 'damping': 0.09, 'avoidOverlap': 0.1}, 'minVelocity': 0.75} if is_physics else {}, layout={'hierarchical': {'enabled': (not is_physics), 'sortMethod': 'directed', 'levelSeparation': 150, 'nodeSpacing': 120}} if not is_physics else {}, interaction={'navigationButtons': True, 'keyboard': True, 'tooltipDelay': 300, 'hover': True})
    # Display Area
    graph_col, info_col = st.columns([3, 1])
    with graph_col:
        st.caption("Graph View: Click/drag to pan, scroll/pinch to zoom. Select node in sidebar to highlight.")
        if not agraph_nodes and filter_term: st.warning(f"No nodes match filter: '{filter_term}'")
        elif not agraph_nodes: st.info("No terms were extracted or matched the filter.")
        elif G is None: st.error("Graph object (G) is None, cannot render.")
        else:
             try: agraph_return = agraph(nodes=agraph_nodes, edges=agraph_edges if agraph_edges is not None else [], config=config)
             except Exception as agraph_err: st.error(f"Error rendering graph component: {agraph_err}"); print(traceback.format_exc())
    with info_col:
        st.subheader("Details & Analysis"); st.markdown("**Selected Definition:**")
        selected_def = terms_map.get(highlight_node, "_Select node in sidebar_")
        st.text_area("Definition Display", value=selected_def, height=150, disabled=True, label_visibility="collapsed", key="def_display_box")
        st.markdown("---"); st.markdown("**Graph Analysis:**")
        if st.session_state.dtg_cycles is not None:
             if st.session_state.dtg_cycles:
                  with st.expander(f"🚨 Found {len(st.session_state.dtg_cycles)} Circular Definitions", expanded=False):
                       for i, cycle in enumerate(st.session_state.dtg_cycles): st.markdown(f"- Cycle {i+1}: `{' → '.join(cycle)} → {cycle[0]}`")
             else: st.caption("✅ No circular definitions detected.")
        if st.session_state.dtg_orphans is not None:
             if st.session_state.dtg_orphans:
                  with st.expander(f"⚠️ Found {len(st.session_state.dtg_orphans)} Orphan Terms", expanded=False):
                       st.markdown(f"`{', '.join(st.session_state.dtg_orphans)}`"); st.caption("_Defined but not linked within definition network._")
             else: st.caption("✅ All defined terms linked.")
    st.divider()
    # Generate DOT Code
    dot_lines = ["digraph G {"]; node_style_map = {node.id: f'[color="{node.color}", fontcolor="#000000"]' for node in agraph_nodes}
    for node_id in sorted(list(displayed_node_ids)):
        style = node_style_map.get(node_id, ""); quoted_node_id = f'"{node_id}"' if re.search(r'\s|[^a-zA-Z0-9_]', node_id) else node_id
        dot_lines.append(f'  {quoted_node_id} {style};')
    for u, v in sorted(agraph_edges_tuples):
        quoted_u = f'"{u}"' if re.search(r'\s|[^a-zA-Z0-9_]', u) else u; quoted_v = f'"{v}"' if re.search(r'\s|[^a-zA-Z0-9_]', v) else v
        dot_lines.append(f'  {quoted_u} -> {quoted_v};')
    dot_lines.append("}"); generated_dot_code = "\n".join(dot_lines)
    # Download Buttons
    st.subheader("Export Graph"); export_cols = st.columns(4); safe_filename_base = re.sub(r'[^\w\-]+', '_', st.session_state.dtg_pdf_name or "graph")
    with export_cols[0]: export_cols[0].download_button(label="📥 DOT Code (.dot)", data=generated_dot_code, file_name=f"{safe_filename_base}_graph.dot", mime="text/vnd.graphviz", use_container_width=True)
    with export_cols[1]:
         try: g_render = graphviz.Source(generated_dot_code); png_bytes = g_render.pipe(format='png'); export_cols[1].download_button(label="🖼️ PNG Image (.png)", data=png_bytes, file_name=f"{safe_filename_base}_graph.png", mime="image/png", use_container_width=True)
         except graphviz.backend.execute.ExecutableNotFound: export_cols[1].warning("Graphviz executable not found.", icon="⚠️")
         except Exception as render_err: export_cols[1].warning(f"PNG ERR: {render_err}", icon="⚠️")
    with export_cols[2]:
         try: g_render_svg = graphviz.Source(generated_dot_code); svg_bytes = g_render_svg.pipe(format='svg'); export_cols[2].download_button(label="📐 SVG Image (.svg)", data=svg_bytes, file_name=f"{safe_filename_base}_graph.svg", mime="image/svg+xml", use_container_width=True)
         except graphviz.backend.execute.ExecutableNotFound: export_cols[2].warning("Graphviz executable not found.", icon="⚠️")
         except Exception as render_err: export_cols[2].warning(f"SVG ERR: {render_err}", icon="⚠️")
    with export_cols[3]:
        if G:
            try:
                 dep_list = [{"Source Term": u, "Depends On (Target Term)": v} for u, v in agraph_edges_tuples]
                 df_deps = pd.DataFrame(dep_list) if dep_list else pd.DataFrame(columns=["Source Term", "Depends On (Target Term)"])
                 csv_output = df_deps.to_csv(index=False).encode('utf-8')
                 export_cols[3].download_button(label="📋 Dependencies (.csv)", data=csv_output, file_name=f"{safe_filename_base}_dependencies.csv", mime="text/csv", use_container_width=True)
            except Exception as csv_err: export_cols[3].warning(f"CSV ERR: {csv_err}", icon="⚠️")
    with st.expander("View Generated DOT Code (for current view)"): st.code(generated_dot_code, language='dot')


# --- Fallback/Error/Initial State Messages ---
elif st.session_state.dtg_error: st.error(f"❌ {st.session_state.dtg_error}")
elif not st.session_state.dtg_pdf_bytes: st.info("⬆️ Upload a document (PDF) using the sidebar to get started.")
else: st.info("⬆️ Ready to generate. Click the 'Generate & Analyze Graph' button in the sidebar.")

# Footer
st.sidebar.markdown("---"); st.sidebar.markdown("Developed with Streamlit & Google Gemini")