# pages/defined_terms_graph.py
# --- COMPLETE FILE vX.Y+1 (Chunking Strategy for Large Files) ---

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

# --- Configuration ---
MODEL_NAME = "gemini-2.5-pro-preview-03-25" # DO NOT CHANGE - Per user request
PAGE_TITLE = "Defined Terms Relationship Grapher"
PAGE_ICON = "🔗"
DEFAULT_NODE_COLOR = "#ACDBC9" # Light greenish-teal
HIGHLIGHT_COLOR = "#FFA07A" # Light Salmon for selected node
NEIGHBOR_COLOR = "#ADD8E6" # Light Blue for neighbors
EDGE_EXTRACTION_DELAY = 0.5 # Seconds delay between edge extraction API calls

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


# --- Helper Function for Text Extraction (Enhanced) ---
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
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text", sort=True)
            if page_text:
                full_text += page_text + "\n\n--- Page Break --- \n\n"

        if not full_text.strip():
            return None, None, "Could not extract any text from the PDF."

        # --- Attempt to Isolate Definitions Section ---
        # Simple patterns - adjust based on common document structures
        patterns = [
            # Pattern 1: Starts with "1. Definitions" or "1. Interpretation" (allowing variations) and ends before "2." or next major section marker
            re.compile(r"^[ \t]*(?:ARTICLE\s+)?1\.\s+(?:Definitions|Interpretation)(.*?)(?:^[ \t]*(?:ARTICLE\s+)?2\.|^[ \t]*(?:[A-Z][A-Z\s]+)\n\n)", re.IGNORECASE | re.MULTILINE | re.DOTALL),
            # Pattern 2: Finds text between "Definitions" header and the next all-caps header or numerical section
            re.compile(r"^[ \t]*Definitions\s*\n(.*?)(?:^[ \t]*[A-Z][A-Z\s]{5,}\n|^[ \t]*\d+\.\s+[A-Z])", re.IGNORECASE | re.MULTILINE | re.DOTALL),
            # Pattern 3: Simpler - find content after "Definitions" until a likely end pattern like "Agreed Terms" or "Subject to the terms..."
             re.compile(r"^[ \t]*(?:Definitions|Interpretation)\s*\n(.*?)(?:Agreed Terms|Subject to the terms|IN WITNESS WHEREOF)", re.IGNORECASE | re.MULTILINE | re.DOTALL),
        ]
        for pattern in patterns:
            match = pattern.search(full_text)
            if match:
                definitions_text = match.group(1).strip()
                # Basic sanity check: Does it contain quoted terms and 'means'?
                if definitions_text and re.search(r'"[^"]+"\s+means', definitions_text, re.IGNORECASE):
                    st.toast("ℹ️ Attempted to isolate Definitions section.", icon="✂️")
                    break # Found a likely candidate
        else: # If no pattern matched
             definitions_text = None # Indicate isolation failed
             st.toast("⚠️ Could not automatically isolate Definitions section, using full text.", icon="⚠️")


    except Exception as e:
        error_msg = f"Error extracting text: {e}"
        print(traceback.format_exc())
        full_text = None # Ensure no partial text is returned on error
        definitions_text = None
    finally:
        if doc:
            try:
                doc.close()
            except Exception as close_err:
                print(f"Warning: Error closing PDF document in finally block: {close_err}")
                pass

    return full_text, definitions_text, error_msg


# --- Helper Function to Parse AI JSON Response (Terms Only) ---
def parse_terms_response(response_text):
    """Parses the AI's JSON response specifically for the 'terms' list."""
    raw_snippet = response_text[:500] + ("..." if len(response_text) > 500 else "")
    json_text = None
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
                # Find the likely end of the last complete term object within the terms list
                # Look for `}]` potentially followed by whitespace and the final `}` of the outer object
                last_complete_term_end = json_text.rfind('}]')
                if last_complete_term_end != -1:
                    # Try to reconstruct a potentially valid partial JSON
                    # Assume the structure is {"terms": [...]}
                    potential_json = json_text[:last_complete_term_end + 2] # Include the found `}]`
                    # We need to find the opening `[` of the terms list
                    terms_list_start = potential_json.find('"terms":')
                    if terms_list_start != -1:
                        terms_list_start = potential_json.find('[', terms_list_start)
                        if terms_list_start != -1:
                             # Reconstruct wrapping object: {"terms": [...]}
                             potential_json = f'{{"terms": {potential_json[terms_list_start:]}}}'
                             # Try parsing the reconstructed partial JSON
                             data = json.loads(potential_json)
                             was_truncated = True
                             final_error_msg = None # Succeeded with fallback
                             st.warning("⚠️ Successfully recovered partial terms list from truncated response.", icon="✅")
                        else: raise ValueError("Could not find start of terms list '[' for fallback.")
                    else: raise ValueError("Could not find '\"terms\":' key for fallback.")
                else:
                    # Maybe it was truncated inside the last definition? Find last `}` before error pos
                    last_brace = json_text.rfind('}', 0, json_err.pos)
                    last_comma = json_text.rfind(',', 0, last_brace)
                    if last_brace != -1 and last_comma != -1 and last_comma < last_brace:
                        # Try to snip off the incomplete part after the last comma
                        potential_json_text = json_text[:last_comma]
                        # Need to re-close the structure carefully
                        # Count open/close braces/brackets up to the snip point? Too complex for now.
                        # Simpler fallback: try parsing just up to the last complete definition end `}"` ?
                        last_def_end = json_text.rfind('"}', 0, json_err.pos)
                        if last_def_end != -1:
                             potential_json = json_text[:last_def_end+2] + '] }' # Guess closing structure
                             try:
                                 data = json.loads(potential_json)
                                 was_truncated = True
                                 final_error_msg = None; st.warning("⚠️ Recovered partial terms list (guessed structure).", icon="✅")
                             except Exception: pass # Inner exception, keep original error

            except Exception as fallback_err:
                # Fallback failed, retain the original error message
                st.warning(f"⚠️ JSON recovery attempt failed: {fallback_err}", icon="❌")
                final_error_msg = f"Failed to decode AI JSON response: {original_error}. Recovery attempt failed. Raw snippet: {raw_snippet}"

        # If data is still None after fallback, return the error
        if final_error_msg:
            return None, final_error_msg, False

    # 3. Validate structure if parsing succeeded (fully or partially)
    try:
        if not isinstance(data, dict):
            return None, f"Parsed data is not a JSON object ({type(data).__name__}). Content: {str(data)[:200]}...", was_truncated
        if "terms" not in data or not isinstance(data["terms"], list):
            return None, f"Parsed data missing 'terms' list key. Keys: {list(data.keys())}", was_truncated

        validated_terms = []
        term_names = set()
        for item in data["terms"]:
            if isinstance(item, dict) and "name" in item and "definition" in item and isinstance(item["name"], str) and isinstance(item["definition"], str):
                term_name = item["name"].strip()
                if term_name and term_name not in term_names:
                    validated_terms.append({"name": term_name, "definition": item["definition"]})
                    term_names.add(term_name)
            # else: st.warning(f"Skipping malformed term item: {item}") # Optional

        if not validated_terms and not was_truncated: # Only error if not truncated and no valid terms
             return None, "AI response contained no valid terms after validation.", was_truncated
        elif not validated_terms and was_truncated:
             st.warning("⚠️ No valid terms found in the recovered partial response.")
             # Return empty list but indicate success (as recovery worked)
             return {"terms": []}, None, was_truncated


        # Return only the validated terms list
        return {"terms": validated_terms}, None, was_truncated

    except Exception as e:
        return None, f"Error validating parsed terms structure: {e}", was_truncated


# --- Helper Function to Parse AI JSON Response (Edge List Only) ---
def parse_edge_list_response(response_text, source_term_name):
    """Parses the AI's JSON response specifically for a list of target term strings."""
    raw_snippet = response_text[:500] + ("..." if len(response_text) > 500 else "")
    json_text = None
    was_truncated = False
    final_error_msg = None

    try:
        # 1. Attempt normal parsing (stripping markdown)
        match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE | re.DOTALL)
        if match:
            json_text = match.group(1).strip()
        else:
            if response_text.strip().startswith("["): # Check if it looks like JSON list
                 json_text = response_text.strip()
            else:
                 return None, f"Response for '{source_term_name}' edges does not appear to be JSON list. Raw: {raw_snippet}", False

        if not json_text:
            return None, f"Response content for '{source_term_name}' edges is empty after stripping.", False

        data = json.loads(json_text)

    except json.JSONDecodeError as json_err:
        # 2. Attempt fallback parsing for truncated JSON list
        original_error = json_err
        final_error_msg = f"Failed to decode edge list JSON for '{source_term_name}': {json_err}. Raw: {raw_snippet}"
        st.warning(f"⚠️ Initial JSON parsing failed for '{source_term_name}' edges ({json_err}). Attempting recovery...", icon="🔧")

        if json_text:
            try:
                # Find the last complete string in the list `"` , or `]`
                last_quote = json_text.rfind('"', 0, json_err.pos)
                last_comma = json_text.rfind(',', 0, last_quote)
                # Ensure the last quote is actually closing a string before the error position
                if last_quote != -1 and (last_comma == -1 or last_comma < last_quote):
                    potential_json = json_text[:last_quote + 1] # Include the quote
                    # Try to close the list
                    if not potential_json.endswith("]"):
                        potential_json += "]"
                    # Ensure it starts with '['
                    if not potential_json.startswith("["):
                         list_start = potential_json.find('[')
                         if list_start != -1: potential_json = potential_json[list_start:]
                         else: raise ValueError("Could not find starting '['")

                    data = json.loads(potential_json)
                    was_truncated = True
                    final_error_msg = None
                    st.warning(f"⚠️ Successfully recovered partial edge list for '{source_term_name}'.", icon="✅")

            except Exception as fallback_err:
                st.warning(f"⚠️ JSON recovery attempt failed for '{source_term_name}' edges: {fallback_err}", icon="❌")
                final_error_msg = f"Failed to decode edge list JSON for '{source_term_name}': {original_error}. Recovery failed. Raw: {raw_snippet}"

        if final_error_msg:
            return None, final_error_msg, False

    # 3. Validate structure if parsing succeeded
    try:
        if not isinstance(data, list):
            return None, f"Parsed edge data for '{source_term_name}' is not a list ({type(data).__name__}). Content: {str(data)[:200]}...", was_truncated

        # Validate contents are strings and strip them
        validated_targets = [item.strip() for item in data if isinstance(item, str) and item.strip()]

        # Return the list of validated target strings
        return validated_targets, None, was_truncated

    except Exception as e:
        return None, f"Error validating parsed edge list structure for '{source_term_name}': {e}", was_truncated


# --- Initialize Session State (Expanded) ---
def initialize_dtg_state():
    defaults = {
        'dtg_pdf_bytes': None,
        'dtg_pdf_name': None,
        'dtg_extracted_text': None, # Full extracted text
        'dtg_definitions_text': None, # Isolated definitions text (if successful)
        'dtg_processing': False,
        'dtg_error': None,
        'dtg_graph_data': None, # Will store {"terms": [...], "edges": [...]}
        'dtg_nx_graph': None,   # Will store the networkx graph object
        'dtg_cycles': None,     # List of cycles found
        'dtg_orphans': None,    # List of orphan nodes
        'dtg_filter_term': "",  # Text input for filtering
        'dtg_highlight_node': None, # Node selected for highlight/definition
        'dtg_layout': 'Physics', # Default layout
        'dtg_terms_processed': 0, # Track progress for edge extraction
        'dtg_total_terms': 0, # Total terms found for progress calculation
        'dtg_partial_results': False, # Flag if results were generated from truncated data
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # Ensure API key exists
    if 'api_key' not in st.session_state:
         st.session_state.api_key = None

initialize_dtg_state()

# --- Graph Analysis Functions ---
def build_networkx_graph(graph_data):
    """Builds a NetworkX DiGraph from parsed AI data."""
    if not graph_data or 'terms' not in graph_data or 'edges' not in graph_data:
        st.warning("Cannot build graph: Missing terms or edges data.")
        return None
    G = nx.DiGraph()
    # Add nodes first
    term_names = set()
    for term_data in graph_data.get('terms', []):
        name = term_data.get('name')
        if name:
            G.add_node(name, definition=term_data.get('definition', ''))
            term_names.add(name)
        else:
            st.warning(f"Skipping term with missing name: {term_data}")

    # Add edges, ensuring nodes exist
    valid_edge_count = 0
    skipped_edge_count = 0
    for edge_data in graph_data.get('edges', []):
        source = edge_data.get('source')
        target = edge_data.get('target')
        if source and target and source in term_names and target in term_names:
             G.add_edge(source, target)
             valid_edge_count += 1
        elif source and target:
             # Log edges pointing to/from terms not found in the nodes list
             # This might happen if term extraction was truncated but edge extraction (partially) succeeded
             st.warning(f"Skipping edge: Source '{source}' or Target '{target}' not found in extracted terms list.")
             skipped_edge_count += 1
        else:
             st.warning(f"Skipping edge with missing source/target: {edge_data}")
             skipped_edge_count += 1

    st.info(f"Graph built: {len(G.nodes())} nodes, {valid_edge_count} edges. ({skipped_edge_count} edges skipped).")
    return G

# --- (find_cycles, find_orphans, get_neighbors remain the same) ---
def find_cycles(G):
    """Finds simple cycles in a NetworkX DiGraph."""
    if G is None: return None
    try:
        return list(nx.simple_cycles(G))
    except Exception as e:
        print(f"Error finding cycles: {e}")
        return None

def find_orphans(G):
    """Finds nodes with in-degree and out-degree of 0."""
    if G is None: return None
    orphans_directed = [
        node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 0
    ]
    return orphans_directed

def get_neighbors(G, node_id):
    """Gets predecessors (pointing to node) and successors (node points to)."""
    if G is None or node_id not in G:
        return set(), set()
    predecessors = set(G.predecessors(node_id))
    successors = set(G.successors(node_id))
    return predecessors, successors

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
    "Upload Document (PDF recommended)*", type=["pdf"], key="dtg_pdf_uploader" # Changed type to pdf only for now
)
# Process upload
if uploaded_file_obj is not None:
    new_bytes = uploaded_file_obj.getvalue()
    if new_bytes != st.session_state.get('dtg_pdf_bytes') or uploaded_file_obj.name != st.session_state.get('dtg_pdf_name'):
        st.session_state.dtg_pdf_bytes = new_bytes
        st.session_state.dtg_pdf_name = uploaded_file_obj.name
        # Reset state related to previous file/analysis
        st.session_state.dtg_extracted_text = None
        st.session_state.dtg_definitions_text = None
        st.session_state.dtg_error = None
        st.session_state.dtg_processing = False
        st.session_state.dtg_graph_data = None
        st.session_state.dtg_nx_graph = None
        st.session_state.dtg_cycles = None
        st.session_state.dtg_orphans = None
        st.session_state.dtg_filter_term = ""
        st.session_state.dtg_highlight_node = None
        st.session_state.dtg_terms_processed = 0
        st.session_state.dtg_total_terms = 0
        st.session_state.dtg_partial_results = False
        st.toast(f"📄 File '{st.session_state.dtg_pdf_name}' loaded.", icon="✅")

        # Extract text (and potentially definitions section)
        if uploaded_file_obj.type == "application/pdf":
            full_text, def_text, error_msg = extract_text_from_pdf(st.session_state.dtg_pdf_bytes)
            st.session_state.dtg_extracted_text = full_text
            st.session_state.dtg_definitions_text = def_text # Store isolated text (or None)
            st.session_state.dtg_error = error_msg
        # Add text file handling back if needed later
        # elif uploaded_file_obj.type == "text/plain": ...
        else:
            st.session_state.dtg_error = f"Unsupported file type: {uploaded_file_obj.type}"

        if not st.session_state.dtg_error and st.session_state.dtg_extracted_text:
            st.toast("Text extracted.", icon="📝")
        st.rerun()

if st.session_state.dtg_error and not st.session_state.dtg_processing and not st.session_state.dtg_graph_data:
     st.error(st.session_state.dtg_error) # Show extraction error if it happened

# Generation Button
st.sidebar.markdown("### 2. Generate & Analyze")
# Use definitions text if available, otherwise fall back to full extracted text
text_available_for_analysis = st.session_state.dtg_definitions_text or st.session_state.dtg_extracted_text
can_generate = (st.session_state.api_key and
                st.session_state.dtg_pdf_bytes and
                text_available_for_analysis and
                not st.session_state.dtg_processing)

generate_button_tooltip = ""
if st.session_state.dtg_processing: generate_button_tooltip = "Processing..."
elif not st.session_state.api_key: generate_button_tooltip = "Enter API Key"
elif not st.session_state.dtg_pdf_bytes: generate_button_tooltip = "Upload a document"
elif not text_available_for_analysis: generate_button_tooltip = "Could not extract text from document"
else: generate_button_tooltip = "Generate graph and analyze term relationships using Gemini"

if st.sidebar.button("✨ Generate & Analyze Graph", key="dtg_generate", disabled=not can_generate, help=generate_button_tooltip, use_container_width=True, type="primary"):
    st.session_state.dtg_processing = True
    # Reset results and intermediate state
    st.session_state.dtg_graph_data = None; st.session_state.dtg_nx_graph = None
    st.session_state.dtg_cycles = None; st.session_state.dtg_orphans = None
    st.session_state.dtg_error = None; st.session_state.dtg_filter_term = ""
    st.session_state.dtg_highlight_node = None
    st.session_state.dtg_terms_processed = 0
    st.session_state.dtg_total_terms = 0
    st.session_state.dtg_partial_results = False # Reset partial results flag
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
         # Ensure highlighted node exists in potentially filtered list
         if st.session_state.dtg_highlight_node and st.session_state.dtg_highlight_node in available_nodes:
              current_highlight_index = available_nodes.index(st.session_state.dtg_highlight_node)
         else:
              st.session_state.dtg_highlight_node = None # Reset highlight if filter removes it

    highlight_key = f"highlight_select_{st.session_state.dtg_filter_term}" # Key changes with filter to reset selection
    st.session_state.dtg_highlight_node = st.sidebar.selectbox(
        "Highlight Node & Neighbors", options=available_nodes, index=current_highlight_index,
        key=highlight_key, help="Select node to highlight it and dependencies."
    )
    if st.session_state.dtg_highlight_node == "--- Select Node ---": st.session_state.dtg_highlight_node = None

    st.session_state.dtg_layout = st.sidebar.radio("Graph Layout", options=['Physics', 'Hierarchical'], index=0 if st.session_state.dtg_layout == 'Physics' else 1, key="dtg_layout_radio", help="Choose layout algorithm.")


# --- Main Area ---
if st.session_state.dtg_processing:
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    final_terms_list = []
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

        # --- Step 1: Extract Terms and Definitions ---
        status_placeholder.info("🧠 Asking Gemini to extract terms and definitions...")
        progress_bar.progress(10, text="Step 1: Extracting Terms...")

        terms_prompt = f"""
Your task is to analyze the provided text, focusing ONLY on the section containing formal definitions (often Section 1 or titled "Definitions" or "Interpretation").

**Output Format:** Produce a single JSON object with ONLY ONE key: "terms".
1.  `"terms"`: A list of JSON objects. Each object must have:
    *   `"name"`: The exact defined term (string), accurately capturing quotes if they are part of the definition marker (e.g., `"Term Name"`).
    *   `"definition"`: The complete definition text associated with that term (string).

**Instructions for Extraction:**

*   **Focus:** Strictly analyze the section containing explicit definitions (e.g., terms in quotes followed by "means..." or ":"). Ignore introductions, recitals, operative clauses unless they contain formal definitions matching this pattern.
*   **Identify Defined Terms:** Only include terms that are formally defined (e.g., `"Term Name" means...` or `Term Name: means...`). Include ALL such terms found in the "terms" list.
*   **Extract Definitions:** Capture the full definition text.
*   **Exclusions:** Do NOT include terms that are merely used but not formally defined in this section. Do not include clause numbers, section numbers, dates, amounts, party names (unless explicitly defined as a term), etc., *as term names*.

**Document Text (Focus on Definitions):**
--- Start Document Text ---
{analysis_text}
--- End Document Text ---

**Final Output (Valid JSON Object with 'terms' key ONLY):**
"""
        terms_generation_config = types.GenerationConfig(response_mime_type="application/json", temperature=0.1)
        safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]

        response_terms = model.generate_content(
            contents=terms_prompt,
            generation_config=terms_generation_config,
            safety_settings=safety_settings,
            request_options={'timeout': 600} # 10 min timeout for terms extraction
        )

        terms_data, term_error_msg, terms_truncated = parse_terms_response(response_terms.text)

        if term_error_msg:
            encountered_errors.append(f"Terms Extraction Error: {term_error_msg}")
            # Decide if fatal: If no terms could be extracted at all, stop.
            if not terms_data or not terms_data.get("terms"):
                 raise ValueError(f"Fatal Error: Could not extract any terms. {term_error_msg}")

        if terms_data and terms_data.get("terms"):
            final_terms_list = terms_data["terms"]
            st.session_state.dtg_total_terms = len(final_terms_list)
            st.session_state.dtg_terms_processed = 0
            st.toast(f"✅ Step 1: Extracted {len(final_terms_list)} potential terms.", icon="📝")
            if terms_truncated:
                st.warning("⚠️ Terms list might be incomplete due to response truncation during Step 1.")
                partial_results_generated = True
                st.session_state.dtg_partial_results = True # Set global flag
        else:
            # Handle case where parsing succeeded but returned empty list (maybe validly no terms found)
            if terms_data is not None and not term_error_msg:
                 st.warning("⚠️ No terms found in the document's definitions section based on Step 1.")
                 st.session_state.dtg_total_terms = 0
            else:
                 # If terms_data is None but no fatal error was raised, it means recovery might have happened but yielded nothing.
                 # Error message already added to encountered_errors. We can proceed but expect no edges.
                 st.error("Could not proceed with edge extraction as term extraction failed or yielded no results.")
                 st.session_state.dtg_total_terms = 0


        progress_bar.progress(30, text=f"Step 1 Complete. Found {st.session_state.dtg_total_terms} terms.")

        # --- Step 2: Extract Edges (Per Term) ---
        if st.session_state.dtg_total_terms > 0:
            status_placeholder.info(f"🧠 Asking Gemini to identify relationships for {st.session_state.dtg_total_terms} terms...")
            valid_term_names = [term['name'] for term in final_terms_list]
            valid_term_names_json = json.dumps(valid_term_names) # Prepare for prompt

            # Use a less strict generation config for edge extraction - simple list needed
            edge_generation_config = types.GenerationConfig(response_mime_type="application/json", temperature=0.0) # Low temp for consistency

            for i, term_obj in enumerate(final_terms_list):
                source_term_name = term_obj['name']
                definition_text = term_obj['definition']
                st.session_state.dtg_terms_processed = i + 1
                progress = 30 + int(65 * (st.session_state.dtg_terms_processed / st.session_state.dtg_total_terms))
                progress_bar.progress(progress, text=f"Step 2: Analyzing '{source_term_name}' ({st.session_state.dtg_terms_processed}/{st.session_state.dtg_total_terms})...")

                if not definition_text:
                    st.warning(f"Skipping edge analysis for '{source_term_name}': Definition text is empty.")
                    continue

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
                    response_edge = model.generate_content(
                        contents=edge_prompt,
                        generation_config=edge_generation_config, # Simpler config
                        safety_settings=safety_settings,
                        request_options={'timeout': 120} # Shorter timeout per edge
                    )

                    target_terms_list, edge_error_msg, edge_truncated = parse_edge_list_response(response_edge.text, source_term_name)

                    if edge_error_msg:
                        encountered_errors.append(f"Edge Error ('{source_term_name}'): {edge_error_msg}")
                        # Don't make this fatal, just skip edges for this term
                        st.warning(f"⚠️ Couldn't reliably extract edges for '{source_term_name}'.")

                    if edge_truncated:
                         partial_results_generated = True
                         st.session_state.dtg_partial_results = True # Set global flag
                         st.warning(f"⚠️ Edge list for '{source_term_name}' might be incomplete due to response truncation.")

                    if target_terms_list:
                        for target_term in target_terms_list:
                            # Final check: ensure target is actually in our list of valid terms
                            if target_term in valid_term_names:
                                final_edges_list.append({"source": source_term_name, "target": target_term})
                            else:
                                st.warning(f"AI suggested edge '{source_term_name}' -> '{target_term}', but target is not in the initially extracted valid terms list. Skipping.")


                except types.StopCandidateException as sce:
                    err_msg = f"Generation Stopped for '{source_term_name}' edges: {sce}. Might be blocked or invalid output."
                    encountered_errors.append(err_msg); st.warning(f"⚠️ {err_msg}")
                except google.api_core.exceptions.GoogleAPIError as api_err:
                     err_msg = f"API Error for '{source_term_name}' edges: {api_err}. Check quota/permissions."
                     encountered_errors.append(err_msg); st.warning(f"⚠️ {err_msg}")
                     # Potentially add logic to stop if quota is exceeded
                except Exception as e:
                     err_msg = f"Processing Error for '{source_term_name}' edges: {e}"
                     encountered_errors.append(err_msg); st.warning(f"⚠️ {err_msg}")
                     print(traceback.format_exc())

                # Add delay between calls
                time.sleep(EDGE_EXTRACTION_DELAY)

        # --- Step 3: Combine and Finalize ---
        progress_bar.progress(95, text="Combining results...")
        st.session_state.dtg_graph_data = {
            "terms": final_terms_list,
            "edges": final_edges_list
        }

        # Perform graph analysis
        status_placeholder.info("⚙️ Analyzing graph structure...")
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
            st.session_state.dtg_error = "Processing completed with errors. See details below."
            # Display errors in an expander
            with st.expander("⚠️ Processing Errors/Warnings Encountered", expanded=True):
                for err in encountered_errors:
                    st.warning(err)
            if partial_results_generated:
                st.warning("Additionally, some data might be incomplete due to AI response truncation.")

        if partial_results_generated and not encountered_errors:
             st.session_state.dtg_error = "Processing completed, but results might be incomplete due to AI response truncation."
             st.warning(st.session_state.dtg_error)

        if not encountered_errors and not partial_results_generated:
            st.toast("✅ Analysis successful!", icon="🎉")

        progress_bar.progress(100, text="Processing Complete.")
        time.sleep(2) # Keep progress bar visible briefly


    # --- Exception Handling (Outer Try Block) ---
    except ValueError as ve: # Catch our own fatal errors
        st.session_state.dtg_error = f"Processing Error: {ve}"
        st.error(f"❌ Fatal Error: {ve}")
        print(traceback.format_exc())
    except types.StopCandidateException as sce:
        st.session_state.dtg_error = f"Generation Stopped: {sce}. Response might be incomplete or blocked."
        st.error(f"❌ {st.session_state.dtg_error}")
        print(traceback.format_exc())
    except google.api_core.exceptions.GoogleAPIError as api_err:
        st.session_state.dtg_error = f"Google API Error: {api_err}. Check key/quota/permissions."
        st.error(f"❌ {st.session_state.dtg_error}")
        print(traceback.format_exc())
    except Exception as e:
        st.session_state.dtg_error = f"Unexpected Processing Error: {e}"
        st.error(f"❌ {st.session_state.dtg_error}")
        print(traceback.format_exc())
    finally:
        st.session_state.dtg_processing = False
        status_placeholder.empty()
        progress_bar.empty()
        st.rerun()


elif st.session_state.dtg_graph_data:
    # --- Display Results ---
    st.subheader(f"📊 Interactive Graph & Analysis for '{st.session_state.dtg_pdf_name}'")

    # Add warning if partial results were generated
    if st.session_state.get('dtg_partial_results', False):
        st.warning("⚠️ Results may be incomplete due to AI response truncation during generation.", icon="⚠️")

    graph_data = st.session_state.dtg_graph_data
    G = st.session_state.dtg_nx_graph
    terms_map = {term['name']: term['definition'] for term in graph_data.get('terms', [])}
    filter_term = st.session_state.dtg_filter_term
    highlight_node = st.session_state.dtg_highlight_node

    # --- (Filtering, Agraph Node/Edge Prep, Config - Code remains the same) ---
    # Filter nodes/edges
    nodes_to_display_names = set(G.nodes()) if G else set()
    if filter_term:
        try: filter_regex = re.compile(filter_term, re.IGNORECASE); nodes_to_display_names = {n for n in G.nodes() if filter_regex.search(n)}
        except re.error: st.warning("Invalid filter regex.", icon="⚠️"); nodes_to_display_names = set(G.nodes()) if G else set()
    # Determine highlight set
    highlight_neighbors_predecessors = set(); highlight_neighbors_successors = set()
    if highlight_node and G: highlight_neighbors_predecessors, highlight_neighbors_successors = get_neighbors(G, highlight_node)

    # --- Prepare Agraph Nodes & Edges ---
    agraph_nodes = []; agraph_edges = []; agraph_edges_tuples = [] # Store tuples for DOT
    displayed_node_ids = set()
    if G:
        # Create Node objects (add nodes to displayed_node_ids)
        for node_id in G.nodes():
            if node_id not in nodes_to_display_names: continue
            displayed_node_ids.add(node_id); node_color = DEFAULT_NODE_COLOR; node_size = 15
            if node_id == highlight_node: node_color = HIGHLIGHT_COLOR; node_size = 25
            elif node_id in highlight_neighbors_predecessors or node_id in highlight_neighbors_successors: node_color = NEIGHBOR_COLOR; node_size = 20
            agraph_nodes.append(Node(id=node_id, label=node_id, color=node_color, size=node_size, font={'color': "#000000"}))

        # Create Edge objects and tuples
        for u, v in G.edges():
            # Ensure both source and target are in the set of nodes *to be displayed* after filtering
            if u in displayed_node_ids and v in displayed_node_ids:
                 agraph_edges_tuples.append((u, v)) # Store tuple for DOT/CSV export
                 agraph_edges.append(Edge(source=u, target=v, color="#CCCCCC")) # Create Agraph Edge

    # --- Configure Agraph (MODIFIED - unchanged from previous version) ---
    is_physics = st.session_state.dtg_layout == 'Physics'
    config = Config(
        width='100%',
        height=700,
        directed=True,
        physics=is_physics,
        hierarchical=not is_physics,
        highlightColor=HIGHLIGHT_COLOR,
        collapsible=False,
        node={'labelProperty':'label', 'size': 15},
        physics_config={'barnesHut': {'gravitationalConstant': -10000, 'centralGravity': 0.1, 'springLength': 180, 'springConstant': 0.05, 'damping': 0.09, 'avoidOverlap': 0.1}, 'minVelocity': 0.75} if is_physics else {},
        layout={'hierarchical': {'enabled': (not is_physics), 'sortMethod': 'directed', 'levelSeparation': 150, 'nodeSpacing': 120}} if not is_physics else {},
        interaction={'navigationButtons': True, 'keyboard': True, 'tooltipDelay': 300, 'hover': True}
    )

    # --- Display Area ---
    graph_col, info_col = st.columns([3, 1])
    with graph_col:
        st.caption("Graph View: Click/drag to pan, scroll/pinch to zoom. Select node in sidebar to highlight.")
        if not agraph_nodes and filter_term: st.warning(f"No nodes match filter: '{filter_term}'")
        elif not agraph_nodes: st.info("No terms were extracted or matched the filter.")
        elif G is None: st.error("Graph object (G) is None, cannot render.") # Added check for G
        # --- Add check to ensure nodes/edges are not empty before calling agraph ---
        else: # G exists and agraph_nodes is not empty
             try:
                  # Pass empty list if agraph_edges is empty/None but nodes exist
                  agraph_return = agraph(nodes=agraph_nodes, edges=agraph_edges if agraph_edges is not None else [], config=config)
             except Exception as agraph_err:
                  st.error(f"Error rendering graph component: {agraph_err}")
                  print(traceback.format_exc()) # Log detailed error

    # --- (Info Column - Definition, Analysis, Download Buttons, DOT code - code remains the same) ---
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
                       st.markdown(f"`{', '.join(st.session_state.dtg_orphans)}`")
                       st.caption("_Defined but not linked within definition network._")
             else: st.caption("✅ All defined terms linked.")
    st.divider()

    # Generate DOT Code for Download
    dot_lines = ["digraph G {"]; node_style_map = {node.id: f'[color="{node.color}", fontcolor="#000000"]' for node in agraph_nodes}
    for node_id in sorted(list(displayed_node_ids)): # Sort nodes for consistent DOT output
        style = node_style_map.get(node_id, "")
        quoted_node_id = f'"{node_id}"' if re.search(r'\s|[^a-zA-Z0-9_]', node_id) else node_id
        dot_lines.append(f'  {quoted_node_id} {style};')
    for u, v in sorted(agraph_edges_tuples): # Sort edges for consistent DOT output
        quoted_u = f'"{u}"' if re.search(r'\s|[^a-zA-Z0-9_]', u) else u
        quoted_v = f'"{v}"' if re.search(r'\s|[^a-zA-Z0-9_]', v) else v
        dot_lines.append(f'  {quoted_u} -> {quoted_v};')
    dot_lines.append("}")
    generated_dot_code = "\n".join(dot_lines)

    # Download Buttons
    st.subheader("Export Graph"); export_cols = st.columns(4); safe_filename_base = re.sub(r'[^\w\-]+', '_', st.session_state.dtg_pdf_name or "graph")
    with export_cols[0]: export_cols[0].download_button(label="📥 DOT Code (.dot)", data=generated_dot_code, file_name=f"{safe_filename_base}_graph.dot", mime="text/vnd.graphviz", use_container_width=True)
    with export_cols[1]:
         try: g_render = graphviz.Source(generated_dot_code); png_bytes = g_render.pipe(format='png'); export_cols[1].download_button(label="🖼️ PNG Image (.png)", data=png_bytes, file_name=f"{safe_filename_base}_graph.png", mime="image/png", use_container_width=True)
         except graphviz.backend.execute.ExecutableNotFound: export_cols[1].warning("Graphviz executable not found for PNG render.", icon="⚠️")
         except Exception as render_err: export_cols[1].warning(f"PNG ERR: {render_err}", icon="⚠️")
    with export_cols[2]:
         try: g_render_svg = graphviz.Source(generated_dot_code); svg_bytes = g_render_svg.pipe(format='svg'); export_cols[2].download_button(label="📐 SVG Image (.svg)", data=svg_bytes, file_name=f"{safe_filename_base}_graph.svg", mime="image/svg+xml", use_container_width=True)
         except graphviz.backend.execute.ExecutableNotFound: export_cols[2].warning("Graphviz executable not found for SVG render.", icon="⚠️")
         except Exception as render_err: export_cols[2].warning(f"SVG ERR: {render_err}", icon="⚠️")
    with export_cols[3]:
        if G:
            try:
                 # Use the filtered edges (agraph_edges_tuples) for CSV
                 dep_list = [{"Source Term": u, "Depends On (Target Term)": v} for u, v in agraph_edges_tuples]
                 df_deps = pd.DataFrame(dep_list) if dep_list else pd.DataFrame(columns=["Source Term", "Depends On (Target Term)"]) # Handle empty case
                 csv_output = df_deps.to_csv(index=False).encode('utf-8')
                 export_cols[3].download_button(label="📋 Dependencies (.csv)", data=csv_output, file_name=f"{safe_filename_base}_dependencies.csv", mime="text/csv", use_container_width=True)
            except Exception as csv_err: export_cols[3].warning(f"CSV ERR: {csv_err}", icon="⚠️")

    with st.expander("View Generated DOT Code (for current view)"): st.code(generated_dot_code, language='dot')


elif st.session_state.dtg_error: st.error(f"❌ {st.session_state.dtg_error}") # Display final error if processing failed severely
elif not st.session_state.dtg_pdf_bytes: st.info("⬆️ Upload a document (PDF) using the sidebar to get started.")
else: st.info("⬆️ Ready to generate. Click the 'Generate & Analyze Graph' button in the sidebar.")

# Footer
st.sidebar.markdown("---"); st.sidebar.markdown("Developed with Streamlit & Google Gemini")