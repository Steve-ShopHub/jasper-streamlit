# pages/defined_terms_graph.py
# --- COMPLETE FILE vX.Y+5 (History Save/Load + GCS) ---

import streamlit as st
import google.generativeai as genai
from google.generativeai import types
import fitz  # PyMuPDF (kept for potential future use/debugging)
import re
import os
import traceback
import time # Import time for timestamps
import io # For download button
import json
import graphviz # Python graphviz library for parsing DOT and rendering
import networkx as nx # For graph analysis (cycles, orphans, neighbors)
import pandas as pd # For CSV export
from streamlit_agraph import agraph, Node, Edge, Config
from PIL import Image # For Logo import
from collections import defaultdict
import uuid # For unique temporary file names & run IDs
from datetime import datetime, timezone # For timestamps
import copy # For deep copying state

# --- Cloud Integrations ---
import google.cloud.firestore
import google.cloud.storage
import google.oauth2.service_account
from google.api_core.exceptions import NotFound # For GCS blob check

# --- Configuration ---
MODEL_NAME = "gemini-1.5-pro-preview-0514" # Specific model requested
PAGE_TITLE = "Defined Terms Relationship Grapher"
PAGE_ICON = "🔗"
DEFAULT_NODE_COLOR = "#ACDBC9" # Light greenish-teal
HIGHLIGHT_COLOR = "#FFA07A" # Light Salmon for selected node
NEIGHBOR_COLOR = "#ADD8E6" # Light Blue for neighbors
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(APP_DIR, "temp_graph_files") # Directory for temporary files
MAX_UPLOAD_RETRIES = 2
UPLOAD_RETRY_DELAY = 3 # seconds
DTG_HISTORY_COLLECTION = "dtg_runs" # Firestore collection for DTG history
GCS_PDF_FOLDER_DTG = "dtg_pdfs" # Folder within GCS bucket for persistent PDFs

# --- Set Page Config ---
st.set_page_config(layout="wide", page_title=PAGE_TITLE, page_icon=PAGE_ICON)

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

except KeyError as e:
    st.error(f"❌ Configuration Error: Missing key '{e}' in Streamlit secrets (`secrets.toml`). Please check your configuration.")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to initialize cloud clients: {e}")
    print(traceback.format_exc())
    st.stop()


# --- Optional CSS ---
# (CSS remains the same)
st.markdown("""
<style>
    /* Ensure Streamlit containers don't add excessive padding */
     div[data-testid="stVerticalBlock"] > div[style*="gap: 1rem;"] {
        gap: 0.5rem !important;
     }
    /* Style for the definition display area (now less relevant, but kept for structure) */
    .definition-box {
        background-color: #f0f2f6;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
        height: 150px; /* Adjust height as needed */
        overflow-y: auto; /* Add scroll if needed */
        font-size: 0.9em;
        white-space: pre-wrap; /* Ensure line breaks are respected */
        word-wrap: break-word; /* Break long words */
    }
    /* Style for History Banner */
    .history-banner {
        background-color: #e0f2f7; /* Light cyan */
        border-left: 5px solid #007bff; /* Blue accent */
        padding: 10px 15px;
        margin-bottom: 1rem;
        border-radius: 5px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Function for Text Extraction ---
# NOTE: No longer used for primary AI interaction, kept for potential future use.
@st.cache_data(show_spinner="Extracting text from PDF (local)...")
def extract_text_from_pdf(pdf_bytes):
    # (Function implementation remains the same)
    if not pdf_bytes: return None, "No PDF file provided."
    doc = None
    try:
        print(f"[{time.strftime('%H:%M:%S')}] (Local) Starting PDF text extraction...")
        start_time = time.time()
        if isinstance(pdf_bytes, io.BytesIO): pdf_bytes = pdf_bytes.getvalue()
        elif not isinstance(pdf_bytes, bytes): return None, f"Invalid input type for PDF extraction: {type(pdf_bytes)}"
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num); page_text = page.get_text("text", sort=True)
            if page_text: text += page_text + "\n\n--- Page Break --- \n\n"
        end_time = time.time()
        if not text.strip():
            print(f"[{time.strftime('%H:%M:%S')}] (Local) PDF text extraction finished in {end_time - start_time:.2f}s. No text found.")
            return None, "Could not extract any text from the PDF."
        print(f"[{time.strftime('%H:%M:%S')}] (Local) PDF text extraction finished in {end_time - start_time:.2f}s. Extracted {len(text)} characters.")
        return text, None
    except Exception as e:
        error_msg = f"Error extracting text locally: {e}"; print(f"[{time.strftime('%H:%M:%S')}] (Local) PDF extraction error: {error_msg}"); traceback.print_exc(); return None, error_msg
    finally:
        if doc:
            try: doc.close()
            except Exception as close_err: print(f"[{time.strftime('%H:%M:%S')}] Warning: Error closing PDF document (local extraction) in finally block: {close_err}")

# --- Helper Function to Parse AI JSON Response ---
# (Function implementation remains the same)
def parse_ai_response(response_text):
    print(f"[{time.strftime('%H:%M:%S')}] Starting AI response parsing...")
    parsing_start_time = time.time()
    if not response_text or not response_text.strip():
         error_msg = "AI response content is empty."; print(f"[{time.strftime('%H:%M:%S')}] Parsing failed: {error_msg}"); return None, error_msg
    try:
        json_start_match = re.search(r"(\{.*\}|\[.*\])", response_text, re.DOTALL)
        json_text = ""
        if json_start_match:
            json_text = json_start_match.group(0).strip(); print(f"[{time.strftime('%H:%M:%S')}] Found potential JSON block via regex.")
        else:
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE | re.DOTALL)
            if match:
                json_text = match.group(1).strip(); print(f"[{time.strftime('%H:%M:%S')}] Found potential JSON block via markdown fence.")
            elif response_text.strip().startswith(("{", "[")):
                json_text = response_text.strip(); print(f"[{time.strftime('%H:%M:%S')}] Assuming raw response is JSON.")
            else:
                 potential_json = response_text.strip()
                 if potential_json.startswith("```") and potential_json.endswith("```"): potential_json = potential_json[3:-3].strip()
                 if potential_json.startswith("json"): potential_json = potential_json[4:].strip()
                 if potential_json.startswith("{") or potential_json.startswith("["):
                      print(f"[{time.strftime('%H:%M:%S')}] Attempting parse after basic cleaning."); json_text = potential_json
                 else:
                     error_msg = f"Response does not appear to contain a JSON object/array. Raw text snippet: {response_text[:500]}..."; print(f"[{time.strftime('%H:%M:%S')}] Parsing failed: {error_msg}"); return None, error_msg
        if not json_text:
            error_msg = "Could not extract JSON content from the response."; print(f"[{time.strftime('%H:%M:%S')}] Parsing failed: {error_msg}"); return None, error_msg
        print(f"[{time.strftime('%H:%M:%S')}] Attempting json.loads() on extracted text (length: {len(json_text)})..."); data = json.loads(json_text); print(f"[{time.strftime('%H:%M:%S')}] json.loads() successful. Validating structure...")
        if not isinstance(data, dict): return None, "Extracted content is not a JSON object."
        if "terms" not in data or "edges" not in data: return None, "Extracted JSON missing required 'terms' or 'edges' keys."
        if not isinstance(data["terms"], list) or not isinstance(data["edges"], list): return None, "'terms' or 'edges' are not lists."
        validated_terms = []; term_names = set()
        for item in data["terms"]:
            if isinstance(item, dict) and "name" in item and isinstance(item["name"], str):
                term_name = item["name"].strip()
                if term_name and term_name not in term_names: validated_terms.append({"name": term_name}); term_names.add(term_name)
        validated_edges = []
        for edge in data["edges"]:
            if isinstance(edge, dict) and "source" in edge and "target" in edge and isinstance(edge["source"], str) and isinstance(edge["target"], str):
                 source = edge["source"].strip(); target = edge["target"].strip()
                 if source and target and source in term_names and target in term_names: validated_edges.append({"source": source, "target": target})
                 else: print(f"[{time.strftime('%H:%M:%S')}] Warning: Skipping edge with invalid/missing term: {source} -> {target}")
        if not validated_terms: return None, "Extracted JSON contained no valid terms after validation."
        validated_data = {"terms": validated_terms, "edges": validated_edges}; parsing_end_time = time.time(); print(f"[{time.strftime('%H:%M:%S')}] Parsing successful. Found {len(validated_terms)} terms, {len(validated_edges)} edges. Duration: {parsing_end_time - parsing_start_time:.2f}s"); return validated_data, None
    except json.JSONDecodeError as json_err:
        parsing_end_time = time.time(); error_pos = json_err.pos; context_window = 50; start = max(0, error_pos - context_window); end = min(len(json_text), error_pos + context_window); error_snippet = json_text[start:end] if json_text else response_text[start:end]; error_snippet_display = repr(error_snippet); error_msg = (f"Failed to decode AI JSON response: {json_err}. Error near char {error_pos}. Snippet: ...{error_snippet_display}..."); print(f"[{time.strftime('%H:%M:%S')}] Parsing failed (JSONDecodeError). Duration: {parsing_end_time - parsing_start_time:.2f}s. Error: {error_msg}"); return None, error_msg
    except Exception as e:
        parsing_end_time = time.time(); error_msg = f"Error parsing AI response structure: {e}"; print(f"[{time.strftime('%H:%M:%S')}] Parsing failed (Exception). Duration: {parsing_end_time - parsing_start_time:.2f}s. Error: {error_msg}"); traceback.print_exc(); return None, error_msg

# --- Initialize Session State ---
def initialize_dtg_state():
    """Initializes session state variables if they don't exist."""
    state_defaults = {
        'dtg_pdf_bytes': None, 'dtg_pdf_name': None,
        'dtg_processing': False, 'dtg_error': None, 'dtg_graph_data': None,
        'dtg_nx_graph': None, 'dtg_cycles': None, 'dtg_orphans': None,
        'dtg_filter_term': "", 'dtg_highlight_node': None, 'dtg_layout': 'Physics',
        'dtg_raw_ai_response': None,
        'dtg_load_history_id': None, # Trigger for loading history
        'dtg_viewing_history': False, # Flag for history mode
        'dtg_history_filename': None, # Filename from history
        'dtg_history_timestamp': None, # Timestamp from history
        'api_key': None, # Keep API key separate
        'run_key': 0, # Used to ensure unique widget keys on rerun
    }
    current_api_key = st.session_state.get('api_key', None) # Preserve API key
    for key, default_value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(default_value) if isinstance(default_value, (dict, list, set, defaultdict)) else default_value
    st.session_state.api_key = current_api_key # Restore API key

def reset_dtg_state(preserve_api_key=True):
    """Resets session state, optionally preserving API key."""
    current_api_key = st.session_state.get('api_key', None) if preserve_api_key else None
    keys_to_reset = list(initialize_dtg_state.__defaults__[0].keys()) # Get keys from defaults dict
    if 'api_key' in keys_to_reset: keys_to_reset.remove('api_key')

    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    initialize_dtg_state() # Re-initialize with defaults
    st.session_state.api_key = current_api_key # Restore preserved key

# --- Initial call to ensure state exists ---
initialize_dtg_state()


# --- Graph Analysis Functions ---
# (build_networkx_graph, find_cycles, find_orphans, get_neighbors remain the same)
def build_networkx_graph(graph_data):
    if not graph_data or not isinstance(graph_data, dict): return None
    G = nx.DiGraph(); added_nodes = set()
    for term_data in graph_data.get('terms', []):
         if isinstance(term_data, dict) and 'name' in term_data:
             node_name = term_data['name'].strip()
             if node_name and node_name not in added_nodes: G.add_node(node_name); added_nodes.add(node_name)
    for edge_data in graph_data.get('edges', []):
        if isinstance(edge_data, dict) and 'source' in edge_data and 'target' in edge_data:
            source = edge_data['source'].strip(); target = edge_data['target'].strip()
            if source and target and G.has_node(source) and G.has_node(target): G.add_edge(source, target)
            else: print(f"[{time.strftime('%H:%M:%S')}] Graph Build Warning: Skipping edge with invalid node(s): '{source}' -> '{target}'")
    return G

def find_cycles(G):
    if G is None: return None
    try: return list(nx.simple_cycles(G))
    except Exception as e: print(f"Error finding cycles: {e}"); return None

def find_orphans(G):
    if G is None: return None
    return [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]

def get_neighbors(G, node_id):
    if G is None or node_id not in G: return set(), set()
    return set(G.predecessors(node_id)), set(G.successors(node_id))

# --- GCS Helper Functions ---
def upload_to_gcs(bucket_name, source_bytes, destination_blob_name, status_placeholder=None):
    """Uploads bytes to GCS bucket for persistent storage."""
    if status_placeholder: status_placeholder.info(f"☁️ Uploading PDF to GCS for history...")
    print(f"[{time.strftime('%H:%M:%S')}] Uploading {len(source_bytes)} bytes to gs://{bucket_name}/{destination_blob_name}")
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        # Consider adding retry logic here if needed
        blob.upload_from_string(source_bytes, content_type='application/pdf', timeout=180) # Increased timeout for potentially large uploads
        gcs_path = f"gs://{bucket_name}/{destination_blob_name}"
        print(f"[{time.strftime('%H:%M:%S')}] Successfully uploaded PDF to {gcs_path}")
        if status_placeholder: status_placeholder.info(f"☁️ PDF saved to {gcs_path}")
        return gcs_path
    except Exception as e:
        error_msg = f"❌ GCS Upload Failed for {destination_blob_name}: {e}"
        if status_placeholder: status_placeholder.error(error_msg)
        print(f"[{time.strftime('%H:%M:%S')}] {error_msg}")
        traceback.print_exc()
        raise # Re-raise to indicate saving failed

def download_from_gcs(bucket_name, source_blob_name):
    """Downloads a blob from the bucket. Returns bytes or None."""
    print(f"[{time.strftime('%H:%M:%S')}] Downloading PDF from gs://{bucket_name}/{source_blob_name}")
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        # Add timeout to download
        pdf_bytes = blob.download_as_bytes(timeout=180)
        print(f"[{time.strftime('%H:%M:%S')}] Successfully downloaded {len(pdf_bytes)} bytes.")
        return pdf_bytes
    except NotFound:
        error_msg = f"❌ Error: PDF not found in GCS at gs://{bucket_name}/{source_blob_name}"
        st.error(error_msg)
        print(f"[{time.strftime('%H:%M:%S')}] {error_msg}")
        return None
    except Exception as e:
        error_msg = f"❌ Failed to download PDF from GCS (gs://{bucket_name}/{source_blob_name}): {e}"
        st.error(error_msg)
        print(f"[{time.strftime('%H:%M:%S')}] {error_msg}")
        traceback.print_exc()
        return None

# --- Load History Logic ---
if 'dtg_load_history_id' in st.session_state and st.session_state.dtg_load_history_id:
    history_id = st.session_state.pop('dtg_load_history_id') # Get ID and clear trigger
    st.info(f"📜 Loading historical analysis: {history_id}...")
    print(f"[{time.strftime('%H:%M:%S')}] Triggered history load for ID: {history_id}")
    load_success = False
    try:
        doc_ref = db.collection(DTG_HISTORY_COLLECTION).document(history_id)
        doc_snapshot = doc_ref.get()

        if doc_snapshot.exists:
            run_data = doc_snapshot.to_dict()
            gcs_pdf_path = run_data.get("gcs_pdf_path")
            hist_graph_data = run_data.get("graph_data")
            hist_filename = run_data.get("filename", "N/A")
            hist_timestamp_obj = run_data.get("analysis_timestamp")
            hist_cycles = run_data.get("cycles", [])
            hist_orphans = run_data.get("orphans", [])
            hist_model = run_data.get("model_name", "Unknown")

            if not gcs_pdf_path:
                 st.error("❌ History record is missing the PDF file path (gcs_pdf_path). Cannot load.")
            elif not hist_graph_data:
                 st.error("❌ History record is missing graph data. Cannot load.")
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
                         hist_graph_data = None # Prevent further processing
                else:
                    # If path doesn't start with gs://, assume it's relative to default bucket/folder
                    # This depends on how you saved it - adjust if necessary
                    # Example: If path is just "dtg_pdfs/20230101/uuid_file.pdf"
                    hist_blob_name = gcs_pdf_path

                if hist_graph_data and hist_blob_name:
                    with st.spinner(f"Downloading PDF from gs://{hist_bucket_name}/{hist_blob_name}..."):
                        pdf_bytes_from_hist = download_from_gcs(hist_bucket_name, hist_blob_name)

                    if pdf_bytes_from_hist:
                        print(f"[{time.strftime('%H:%M:%S')}] PDF downloaded. Populating state from history.")
                        # Reset state before loading historical data
                        reset_dtg_state(preserve_api_key=True)

                        # Load historical data into session state
                        st.session_state.dtg_pdf_bytes = pdf_bytes_from_hist
                        st.session_state.dtg_pdf_name = hist_filename
                        st.session_state.dtg_graph_data = hist_graph_data
                        st.session_state.dtg_cycles = hist_cycles
                        st.session_state.dtg_orphans = hist_orphans
                        st.session_state.dtg_viewing_history = True
                        st.session_state.dtg_history_filename = hist_filename
                        st.session_state.dtg_history_model = hist_model # Store model used

                        # Format timestamp safely
                        try:
                             if isinstance(hist_timestamp_obj, datetime):
                                 # Attempt to display in local timezone
                                 local_ts = hist_timestamp_obj.astimezone()
                                 hist_ts_str = local_ts.strftime('%Y-%m-%d %H:%M:%S %Z')
                             elif isinstance(hist_timestamp_obj, str): # Handle if already string
                                  hist_ts_str = hist_timestamp_obj
                             else: hist_ts_str = str(hist_timestamp_obj or "N/A")
                        except Exception as ts_format_err:
                             print(f"Warning: Could not format history timestamp: {ts_format_err}")
                             hist_ts_str = "Invalid Timestamp"
                        st.session_state.dtg_history_timestamp = hist_ts_str

                        # Rebuild NetworkX graph from loaded data
                        st.session_state.dtg_nx_graph = build_networkx_graph(hist_graph_data)
                        if st.session_state.dtg_nx_graph is None:
                             st.warning("Could not rebuild graph from historical data.")

                        st.success(f"✅ Successfully loaded history for '{hist_filename}' ({hist_ts_str}).")
                        load_success = True
                        time.sleep(1) # Give user time to see message
                        st.rerun() # Rerun to update UI immediately
                    else:
                        # Download failed, error shown by download_from_gcs
                        st.error("Failed to download the PDF associated with this history entry.")
                        reset_dtg_state(preserve_api_key=True) # Reset state if load fails mid-way
        else:
            st.error(f"❌ History record with ID '{history_id}' not found in database.")
            reset_dtg_state(preserve_api_key=True)

    except Exception as e:
        st.error(f"❌ Error loading historical data: {e}")
        print(f"[{time.strftime('%H:%M:%S')}] History Load Error: {e}\n{traceback.format_exc()}")
        reset_dtg_state(preserve_api_key=True) # Reset state on error
    finally:
        # Ensure trigger is cleared even if loading failed partway
        if 'dtg_load_history_id' in st.session_state:
             del st.session_state['dtg_load_history_id']
        if not load_success:
             # Ensure viewing_history is False if load failed
             st.session_state.dtg_viewing_history = False

# --- Streamlit UI ---

# --- Header ---
header_col1, header_col2 = st.columns([1, 5])
with header_col1:
    LOGO_FILE = "jasper-logo-1.png"
    LOGO_PATH = os.path.join(APP_DIR, LOGO_FILE)
    if os.path.exists(LOGO_PATH):
        try: st.image(Image.open(LOGO_PATH), width=80)
        except Exception as img_err: st.warning(f"Logo load error: {img_err}")
with header_col2:
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    if st.session_state.dtg_viewing_history:
        st.caption(f"Reviewing Historical Analysis | File: **{st.session_state.dtg_history_filename}** | Generated: {st.session_state.dtg_history_timestamp}")
    else:
        st.caption("Upload document, generate interactive graph of defined terms (names only), analyze relationships.")
st.divider()

# --- History Mode Banner & Exit Button ---
if st.session_state.dtg_viewing_history:
    banner_cols = st.columns([4, 1])
    with banner_cols[0]:
        st.markdown(
            f"""<div class="history-banner">
            📜 Viewing historical analysis for <strong>{st.session_state.dtg_history_filename}</strong> (Generated: {st.session_state.dtg_history_timestamp})
            </div>""",
            unsafe_allow_html=True
        )
    with banner_cols[1]:
        if st.button("⬅️ Exit History / New", key="exit_history_dtg", use_container_width=True):
            reset_dtg_state(preserve_api_key=True)
            st.rerun()

# --- Sidebar Controls ---
st.sidebar.title("Controls")
st.sidebar.markdown("---")

# API Key Input (always show)
api_key_input = st.sidebar.text_input("Google AI Gemini API Key*", type="password", key=f"api_key_sidebar_dtg_{st.session_state.run_key}", value=st.session_state.get("api_key", ""), help="Your Gemini API key (ensure File API permissions).")
if api_key_input and api_key_input != st.session_state.api_key: st.session_state.api_key = api_key_input
if not st.session_state.api_key and not st.session_state.dtg_viewing_history: st.sidebar.warning("API Key required.", icon="🔑")

# File Upload (only if not viewing history)
if not st.session_state.dtg_viewing_history:
    st.sidebar.markdown("### 1. Upload Document")
    uploaded_file_obj = st.sidebar.file_uploader("Upload Document (PDF recommended)*", type=["pdf"], key=f"dtg_pdf_uploader_{st.session_state.run_key}")

    if uploaded_file_obj is not None:
        new_bytes = uploaded_file_obj.getvalue()
        # Check if file content or name has changed
        if new_bytes != st.session_state.get('dtg_pdf_bytes') or uploaded_file_obj.name != st.session_state.get('dtg_pdf_name'):
            # Reset state on new file upload
            reset_dtg_state(preserve_api_key=True) # Resets state but keeps API key
            st.session_state.dtg_pdf_bytes = new_bytes
            st.session_state.dtg_pdf_name = uploaded_file_obj.name
            st.session_state.run_key += 1 # Increment run key to reset uploader state
            st.toast(f"📄 File '{st.session_state.dtg_pdf_name}' loaded.", icon="✅")
            print(f"[{time.strftime('%H:%M:%S')}] New file uploaded: {st.session_state.dtg_pdf_name}.")
            st.rerun()

# Display error if exists (and not processing/has graph data)
if st.session_state.dtg_error and not st.session_state.dtg_processing and not st.session_state.dtg_graph_data:
     st.error(st.session_state.dtg_error)

# Generate Button (only if not viewing history)
if not st.session_state.dtg_viewing_history:
    st.sidebar.markdown("### 2. Generate & Analyze")
    can_generate = (
        st.session_state.api_key and
        st.session_state.dtg_pdf_bytes and
        not st.session_state.dtg_processing
    )
    generate_button_tooltip = ""
    if st.session_state.dtg_processing: generate_button_tooltip = "Processing..."
    elif not st.session_state.api_key: generate_button_tooltip = "Enter API Key"
    elif not st.session_state.dtg_pdf_bytes: generate_button_tooltip = "Upload a document"
    else: generate_button_tooltip = "Generate graph (names only)"

    if st.sidebar.button("✨ Generate & Analyze Graph", key="dtg_generate", disabled=not can_generate, help=generate_button_tooltip, use_container_width=True, type="primary"):
        print(f"[{time.strftime('%H:%M:%S')}] 'Generate & Analyze Graph' button clicked.")
        # Reset only analysis-specific state before starting processing
        st.session_state.dtg_processing = True
        st.session_state.dtg_graph_data = None
        st.session_state.dtg_nx_graph = None
        st.session_state.dtg_cycles = None
        st.session_state.dtg_orphans = None
        st.session_state.dtg_error = None
        st.session_state.dtg_filter_term = ""
        st.session_state.dtg_highlight_node = None
        st.session_state.dtg_raw_ai_response = None
        # DO NOT reset pdf_bytes, pdf_name, api_key here
        st.rerun()

# Graph Interaction Controls (Show if graph data exists, regardless of history mode)
if st.session_state.dtg_graph_data:
    st.sidebar.markdown("---"); st.sidebar.markdown("### 3. Graph Interaction")
    st.session_state.dtg_filter_term = st.sidebar.text_input("Filter Nodes (by name)", value=st.session_state.dtg_filter_term, placeholder="Type term...", key=f"dtg_filter_input_{st.session_state.run_key}").strip()
    available_nodes = ["--- Select Node ---"]; current_highlight_index = 0
    if st.session_state.dtg_nx_graph:
         nodes_to_consider = list(st.session_state.dtg_nx_graph.nodes())
         if st.session_state.dtg_filter_term:
              try: filter_regex = re.compile(st.session_state.dtg_filter_term, re.IGNORECASE); nodes_to_consider = [n for n in nodes_to_consider if filter_regex.search(n)]
              except re.error: st.sidebar.warning("Invalid regex", icon="⚠️"); nodes_to_consider = []
         available_nodes.extend(sorted(nodes_to_consider))
         if st.session_state.dtg_highlight_node and st.session_state.dtg_highlight_node in available_nodes:
              current_highlight_index = available_nodes.index(st.session_state.dtg_highlight_node)
         # Auto-clear highlight if filter changes and node is no longer visible
         elif st.session_state.dtg_highlight_node and st.session_state.dtg_highlight_node not in available_nodes:
              st.session_state.dtg_highlight_node = None; current_highlight_index = 0

    # Use run_key in widget key to handle resets properly
    highlight_key = f"highlight_select_{st.session_state.dtg_filter_term}_{st.session_state.run_key}"
    new_highlight_node = st.sidebar.selectbox("Highlight Node & Neighbors", options=available_nodes, index=current_highlight_index, key=highlight_key)

    if new_highlight_node != ("--- Select Node ---" if st.session_state.dtg_highlight_node is None else st.session_state.dtg_highlight_node):
         st.session_state.dtg_highlight_node = new_highlight_node if new_highlight_node != "--- Select Node ---" else None
         st.rerun() # Rerun only if selection changed

    st.session_state.dtg_layout = st.sidebar.radio("Graph Layout", options=['Physics', 'Hierarchical'], index=0 if st.session_state.dtg_layout == 'Physics' else 1, key=f"dtg_layout_radio_{st.session_state.run_key}")

# History Page Link (always show at bottom of sidebar)
st.sidebar.markdown("---")
if st.sidebar.button("📜 View Analysis History", key="view_dtg_history", use_container_width=True):
     try:
         st.switch_page("pages/history_dtg.py")
     except Exception as e:
          st.sidebar.error(f"Could not navigate to history page: {e}")


# --- Main Area ---
if st.session_state.dtg_processing:
    status_placeholder = st.empty()
    live_response_placeholder = st.empty() # Placeholder for live response preview
    full_response_text = ""
    process_start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] ===== Starting Generation Process =====")

    temp_file_path = None          # Local temp file for Gemini upload
    uploaded_file_ref = None       # Gemini File API reference (temporary)
    gemini_file_upload_successful = False
    persistent_gcs_path = None     # GCS path for history saving
    firestore_run_id = str(uuid.uuid4()) # Unique ID for this run (used for GCS & Firestore)
    run_error_message = None       # Store error message for saving to Firestore if run fails
    current_analysis_timestamp = datetime.now(timezone.utc) # Timestamp for this run

    # Wrap the entire processing in a try...finally to ensure cleanup
    try:
        with st.spinner(f"⚙️ Analyzing '{st.session_state.dtg_pdf_name}'..."):
            status_placeholder.info("🔑 Configuring API...")
            print(f"[{time.strftime('%H:%M:%S')}] Configuring GenAI client...")
            try:
                genai.configure(api_key=st.session_state.api_key)
            except Exception as config_err:
                 run_error_message = f"Failed to configure API key: {config_err}"
                 raise ValueError(run_error_message)

            pdf_bytes = st.session_state.dtg_pdf_bytes
            pdf_name = st.session_state.dtg_pdf_name
            if not pdf_bytes or not pdf_name:
                run_error_message = "Cannot proceed without uploaded PDF data."
                raise ValueError(run_error_message)

            # --- Create Temporary File for Gemini API ---
            status_placeholder.info("💾 Saving file temporarily for analysis upload...")
            print(f"[{time.strftime('%H:%M:%S')}] Creating temporary local file...")
            safe_original_filename = re.sub(r'[^\w\-.]+', '_', pdf_name)
            temp_filename = f"dtg_temp_{firestore_run_id}_{safe_original_filename}"
            os.makedirs(TEMP_DIR, exist_ok=True)
            temp_file_path = os.path.join(TEMP_DIR, temp_filename)
            try:
                with open(temp_file_path, "wb") as f: f.write(pdf_bytes)
                print(f"[{time.strftime('%H:%M:%S')}] Temporary file saved to: {temp_file_path}")
            except Exception as temp_err:
                run_error_message = f"Failed to create temporary file: {temp_err}"
                traceback.print_exc()
                raise # Propagate error

            # --- Upload File to Gemini File API (Temporary) ---
            status_placeholder.info("☁️ Uploading file to Google AI for analysis...")
            print(f"[{time.strftime('%H:%M:%S')}] Uploading temporary file to Gemini File API: {temp_file_path}")
            upload_start_time = time.time()
            for attempt in range(MAX_UPLOAD_RETRIES + 1):
                try:
                    display_name = f"DTG_{safe_original_filename}_{firestore_run_id}"
                    uploaded_file_ref = genai.upload_file(path=temp_file_path, display_name=display_name)
                    upload_end_time = time.time(); print(f"[{time.strftime('%H:%M:%S')}] Gemini file uploaded successfully. Duration: {upload_end_time - upload_start_time:.2f}s")
                    print(f"[{time.strftime('%H:%M:%S')}] Cloud File Name (Temp): {uploaded_file_ref.name}")
                    gemini_file_upload_successful = True; st.toast("☁️ File uploaded to Google AI.", icon="✅"); break
                except Exception as upload_err:
                    print(f"[{time.strftime('%H:%M:%S')}] Gemini file upload attempt {attempt+1} failed: {upload_err}")
                    err_str = str(upload_err).lower()
                    if "api key" in err_str or "authenticat" in err_str or "permission" in err_str or "quota" in err_str:
                        run_error_message = f"Gemini file upload failed (Permissions/Quota/Key): {upload_err}."
                        traceback.print_exc(); raise ValueError(run_error_message)
                    elif attempt < MAX_UPLOAD_RETRIES:
                        print(f"[{time.strftime('%H:%M:%S')}] Retrying Gemini upload in {UPLOAD_RETRY_DELAY}s...")
                        status_placeholder.warning(f"☁️ Gemini upload attempt {attempt+1} failed, retrying...")
                        time.sleep(UPLOAD_RETRY_DELAY)
                    else: # Final attempt failed
                        run_error_message = f"Failed to upload file to Gemini after {MAX_UPLOAD_RETRIES + 1} attempts: {upload_err}"
                        traceback.print_exc(); raise ValueError(run_error_message)

            if not gemini_file_upload_successful or not uploaded_file_ref:
                 run_error_message = run_error_message or "Gemini file upload process failed unexpectedly."
                 raise ValueError(run_error_message)

            # --- Prompt & API Call ---
            status_placeholder.info("🧠 Preparing analysis prompt...")
            # (prompt_instructions definition remains the same as previous version)
            prompt_instructions = f"""
Your task is to analyze ONLY the 'Definitions' section (typically Section 1 or similar) of the **provided document file**. The goal is to identify all formally defined terms and map the interdependencies *only* between these terms based on their definitions.

**Output Format:** Produce a single JSON object with two keys: "terms" and "edges". Respond **only** with the valid JSON object, enclosed in ```json ... ``` if necessary, and nothing else (no introductory text, explanations, or summaries before or after the JSON).
1.  `"terms"`: A list of JSON objects. Each object must have ONLY:
    *   `"name"`: The exact defined term (string), properly handling quotes if they were part of the definition marker. **DO NOT include the definition text itself.**
2.  `"edges"`: A list of JSON objects. Each object represents a directed link and must have:
    *   `"source"`: The name of the defined term whose definition uses another term (string, must match a name in the "terms" list).
    *   `"target"`: The name of the defined term used within the source term's definition (string, must match a name in the "terms" list).

**Instructions for Extraction (Applied to the Provided File):**

*   **Focus:** Strictly analyze the section containing explicit definitions (e.g., terms in quotes followed by "means..."). Ignore other sections of the file.
*   **Identify Defined Terms:** Only include terms that are formally defined within this 'Definitions' section (e.g., `"Term Name" means...`). Include all such terms found in the "terms" list, providing only their `"name"`.
*   **Omit Definitions:** CRITICAL - **Do NOT** include the full definition text in the output JSON. Only provide the term's name.
*   **Identify Edges (Links):** Even though you are not outputting the definition text, you MUST read the definition for each formally defined term ("Term A"). If that definition text explicitly uses another term ("Term B") that is *also* formally defined in the same section, create an edge object from "Term A" (source) to "Term B" (target).
*   **Exclusions (CRITICAL): Do NOT include data in the "terms" or "edges" lists relating to:** Clause numbers, Section numbers, Schedule numbers, specific dates, amounts, percentages, references to external laws/acts/directives (unless the act itself is the primary term being defined), party names (unless explicitly defined as a term), or acronyms (unless formally defined). Only include formally defined terms (by name) and their direct definition-based links to other formally defined terms.
*   **Completeness:** Ensure all formally defined terms from the relevant section are included by name in the "terms" list. Ensure all valid definition-based links between these terms are included in the "edges" list.

**Final Output (Valid JSON Object Only, inside ```json ... ``` if needed):**
"""
            model = genai.GenerativeModel(MODEL_NAME)
            generation_config = types.GenerationConfig(temperature=0.1)
            safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]

            status_placeholder.info("📞 Calling Gemini API (streaming response)...")
            print(f"[{time.strftime('%H:%M:%S')}] Sending request to Gemini API with file URI: {uploaded_file_ref.uri}")
            api_call_start_time = time.time()

            response = model.generate_content(
                contents=[uploaded_file_ref, prompt_instructions], # Pass file ref and prompt
                generation_config=generation_config, safety_settings=safety_settings,
                request_options={'timeout': 900}, stream=True
            )

            # --- Stream Processing with Live Preview ---
            status_placeholder.info("⏳ Receiving streamed response from Gemini...")
            live_response_placeholder.info("Waiting for first chunk...")
            print(f"[{time.strftime('%H:%M:%S')}] Waiting for stream...")
            chunk_count = 0; stream_start_time = None
            for chunk in response:
                # (Stream processing loop remains the same as previous version)
                if stream_start_time is None: stream_start_time = time.time(); print(f"[{time.strftime('%H:%M:%S')}] First chunk received! Latency: {stream_start_time - api_call_start_time:.2f}s"); live_response_placeholder.empty()
                try:
                    if hasattr(chunk, 'text') and chunk.text:
                        full_response_text += chunk.text; chunk_count += 1
                        if chunk_count % 10 == 0 or chunk_count == 1:
                             current_time = time.time(); elapsed_stream = current_time - (stream_start_time if stream_start_time else api_call_start_time); print(f"[{time.strftime('%H:%M:%S')}] Received chunk {chunk_count}. Total length: {len(full_response_text)}. Stream duration: {elapsed_stream:.2f}s")
                             status_placeholder.info(f"⏳ Receiving streamed response... (chunk {chunk_count})")
                             live_response_placeholder.code(f"# Receiving Chunk {chunk_count}...\n{full_response_text}", language="json")
                except ValueError as ve: print(f"[{time.strftime('%H:%M:%S')}] Warning: Skipping chunk due to ValueError: {ve}")
                except Exception as chunk_err: print(f"[{time.strftime('%H:%M:%S')}] Error processing chunk {chunk_count+1}: {chunk_err}")

            api_call_end_time = time.time()
            st.session_state.dtg_raw_ai_response = full_response_text
            live_response_placeholder.code(f"# Stream Complete. Raw Response:\n{full_response_text}", language="json")
            print(f"[{time.strftime('%H:%M:%S')}] Stream finished. Chunks: {chunk_count}. Length: {len(full_response_text)} chars.")
            if stream_start_time: print(f"[{time.strftime('%H:%M:%S')}] Stream duration: {api_call_end_time - stream_start_time:.2f}s.")
            print(f"[{time.strftime('%H:%M:%S')}] Total API call duration: {api_call_end_time - api_call_start_time:.2f}s.")

            # --- Response Parsing & Graph Building ---
            live_response_placeholder.empty() # Clear live preview
            status_placeholder.info("📄 Processing Gemini's full response...")
            if not full_response_text.strip():
                 run_error_message = "AI returned an empty response."
                 raise ValueError(run_error_message)

            graph_data, error_msg = parse_ai_response(full_response_text)
            if error_msg:
                run_error_message = f"Failed to parse AI response: {error_msg}"
                raise ValueError(run_error_message)

            st.session_state.dtg_graph_data = graph_data
            st.toast("Term names & links extracted!", icon="📊")
            status_placeholder.info("⚙️ Analyzing graph structure...")

            print(f"[{time.strftime('%H:%M:%S')}] Building NetworkX graph...")
            graph_build_start_time = time.time()
            st.session_state.dtg_nx_graph = build_networkx_graph(graph_data)
            graph_build_end_time = time.time()

            if st.session_state.dtg_nx_graph is None:
                run_error_message = "Failed to build graph from parsed data."
                raise ValueError(run_error_message)

            print(f"[{time.strftime('%H:%M:%S')}] Graph built. Nodes: {len(st.session_state.dtg_nx_graph.nodes())}, Edges: {len(st.session_state.dtg_nx_graph.edges())}. Duration: {graph_build_end_time - graph_build_start_time:.2f}s")

            print(f"[{time.strftime('%H:%M:%S')}] Finding cycles & orphans...")
            analysis_start_time = time.time()
            st.session_state.dtg_cycles = find_cycles(st.session_state.dtg_nx_graph)
            st.session_state.dtg_orphans = find_orphans(st.session_state.dtg_nx_graph)
            analysis_end_time = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Found {len(st.session_state.dtg_cycles) if st.session_state.dtg_cycles is not None else 'N/A'} cycles.")
            print(f"[{time.strftime('%H:%M:%S')}] Found {len(st.session_state.dtg_orphans) if st.session_state.dtg_orphans is not None else 'N/A'} orphans.")
            print(f"[{time.strftime('%H:%M:%S')}] Graph analysis duration: {analysis_end_time - analysis_start_time:.2f}s")
            st.toast("Graph analysis complete.", icon="🔬")
            status_placeholder.info("💾 Saving analysis results to history...")

            # --- Upload ORIGINAL PDF to GCS for History ---
            # Construct persistent GCS path
            gcs_blob_name = f"{GCS_PDF_FOLDER_DTG}/{current_analysis_timestamp.strftime('%Y%m%d')}/{firestore_run_id}_{safe_original_filename}"
            persistent_gcs_path = upload_to_gcs(GCS_BUCKET_NAME, pdf_bytes, gcs_blob_name, status_placeholder)
            if not persistent_gcs_path:
                 # Error handled within upload_to_gcs, but we stop saving here
                 st.warning("Proceeding without saving to history due to GCS upload failure.")
            else:
                # --- Save successful run to Firestore ---
                data_to_save = {
                    "filename": pdf_name,
                    "analysis_timestamp": current_analysis_timestamp, # Use UTC timestamp object
                    "gcs_pdf_path": persistent_gcs_path,
                    "graph_data": st.session_state.dtg_graph_data, # The dict with terms/edges
                    "cycles": st.session_state.dtg_cycles if st.session_state.dtg_cycles is not None else [],
                    "orphans": st.session_state.dtg_orphans if st.session_state.dtg_orphans is not None else [],
                    "model_name": MODEL_NAME,
                    "error_message": None # Explicitly null for success
                }
                try:
                    db.collection(DTG_HISTORY_COLLECTION).document(firestore_run_id).set(data_to_save)
                    print(f"[{time.strftime('%H:%M:%S')}] Successfully saved run {firestore_run_id} to Firestore.")
                    status_placeholder.success("✅ Analysis complete and saved to history!")
                    st.toast("💾 Analysis saved to history.", icon="✅")
                except Exception as db_err:
                    st.error(f"❌ Failed to save results to Firestore database: {db_err}")
                    print(f"[{time.strftime('%H:%M:%S')}] Firestore save error: {db_err}")
                    traceback.print_exc()
                    # Don't set dtg_error here, as analysis itself succeeded

    # --- Exception Handling ---
    except (ValueError, types.StopCandidateException, google.api_core.exceptions.GoogleAPIError) as e:
        # Handle specific errors caught and raised within the try block or API errors
        # Error message should already be in run_error_message or generated here
        if not run_error_message: run_error_message = f"Analysis Error: {type(e).__name__}: {e}"
        st.session_state.dtg_error = run_error_message # Set session state error for display
        print(f"[{time.strftime('%H:%M:%S')}] Analysis failed: {run_error_message}")
        if not isinstance(e, ValueError): traceback.print_exc() # Print traceback for non-ValueErrors
        # Attempt to save failure record to Firestore
        try:
            status_placeholder.info("💾 Saving error details to history...")
            error_data = {
                "filename": st.session_state.get('dtg_pdf_name', 'Unknown'),
                "analysis_timestamp": current_analysis_timestamp,
                "gcs_pdf_path": None, # No persistent GCS path on failure usually
                "graph_data": None, "cycles": None, "orphans": None,
                "model_name": MODEL_NAME,
                "error_message": run_error_message[:1000] # Limit error message length
            }
            db.collection(DTG_HISTORY_COLLECTION).document(firestore_run_id).set(error_data)
            print(f"[{time.strftime('%H:%M:%S')}] Saved error record {firestore_run_id} to Firestore.")
            st.toast("💾 Error details saved to history.", icon="💾")
        except Exception as db_err:
             print(f"[{time.strftime('%H:%M:%S')}] CRITICAL: Failed to save error record to Firestore: {db_err}")
             st.warning("Could not save error details to history.", icon="⚠️")

    except Exception as e:
        # Catch any other unexpected errors
        run_error_message = f"An unexpected critical error occurred: {e}"
        st.session_state.dtg_error = run_error_message
        print(f"[{time.strftime('%H:%M:%S')}] Unexpected critical error: {run_error_message}")
        traceback.print_exc()
        # Attempt to save failure record (similar to above)
        try:
            status_placeholder.info("💾 Saving error details to history...")
            error_data = {"filename": st.session_state.get('dtg_pdf_name', 'Unknown'), "analysis_timestamp": current_analysis_timestamp, "gcs_pdf_path": None, "graph_data": None, "cycles": None, "orphans": None, "model_name": MODEL_NAME, "error_message": run_error_message[:1000]}
            db.collection(DTG_HISTORY_COLLECTION).document(firestore_run_id).set(error_data)
            print(f"[{time.strftime('%H:%M:%S')}] Saved error record {firestore_run_id} to Firestore.")
            st.toast("💾 Error details saved to history.", icon="💾")
        except Exception as db_err:
             print(f"[{time.strftime('%H:%M:%S')}] CRITICAL: Failed to save error record to Firestore: {db_err}")
             st.warning("Could not save error details to history.", icon="⚠️")

    # --- Cleanup Phase ---
    finally:
        print(f"[{time.strftime('%H:%M:%S')}] Entering cleanup phase...")
        # Clear placeholders only after potential saving messages
        status_placeholder.info("🧹 Cleaning up temporary resources...")
        live_response_placeholder.empty()

        # 1. Delete Temporary Google Cloud File (used for generation)
        if uploaded_file_ref and hasattr(uploaded_file_ref, 'name'):
            try:
                print(f"[{time.strftime('%H:%M:%S')}] Deleting temporary cloud file: {uploaded_file_ref.name}...")
                delete_start = time.time(); genai.delete_file(name=uploaded_file_ref.name); delete_end = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] Temp cloud file deleted. Duration: {delete_end - delete_start:.2f}s")
                # Don't toast success here, it might overwrite final status message
            except Exception as delete_err:
                print(f"[{time.strftime('%H:%M:%S')}] WARNING: Failed to delete temp cloud file '{uploaded_file_ref.name}': {delete_err}")
                st.sidebar.warning(f"Temp cloud cleanup issue: {delete_err}", icon="⚠️")
        elif gemini_file_upload_successful:
             print(f"[{time.strftime('%H:%M:%S')}] WARNING: Temp cloud file reference lost before cleanup.")
             st.sidebar.warning("Could not auto-delete temp cloud file (ref lost).", icon="⚠️")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Skipping temp cloud file deletion.")

        # 2. Delete Local Temporary File
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                print(f"[{time.strftime('%H:%M:%S')}] Deleting local temporary file: {temp_file_path}...")
                os.remove(temp_file_path); print(f"[{time.strftime('%H:%M:%S')}] Local temp file deleted.")
            except Exception as local_delete_err:
                print(f"[{time.strftime('%H:%M:%S')}] WARNING: Failed to delete local temp file '{temp_file_path}': {local_delete_err}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Skipping local temp file deletion.")

        # Ensure raw response is saved even if error happens late
        if full_response_text and not st.session_state.dtg_raw_ai_response:
             st.session_state.dtg_raw_ai_response = full_response_text

        st.session_state.dtg_processing = False
        status_placeholder.empty() # Clear the status message area
        process_end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] ===== Generation Process Ended =====")
        print(f"[{time.strftime('%H:%M:%S')}] Total processing duration (incl. save/cleanup): {process_end_time - process_start_time:.2f} seconds.")
        print(f"[{time.strftime('%H:%M:%S')}] Triggering rerun.")
        time.sleep(1) # Allow user to see final toasts/warnings/status
        st.rerun()

elif st.session_state.dtg_graph_data:
    # --- Display Results (Works for both live run and loaded history) ---
    display_name = st.session_state.dtg_history_filename if st.session_state.dtg_viewing_history else st.session_state.dtg_pdf_name
    st.subheader(f"📊 Interactive Graph & Analysis for '{display_name}'")
    G = st.session_state.dtg_nx_graph
    filter_term = st.session_state.dtg_filter_term
    highlight_node = st.session_state.dtg_highlight_node

    # (Graph filtering, node/edge preparation logic remains the same)
    nodes_to_display_names = set(G.nodes()) if G else set()
    if filter_term:
        try: filter_regex = re.compile(filter_term, re.IGNORECASE); nodes_to_display_names = {n for n in G.nodes() if filter_regex.search(n)}
        except re.error: st.warning("Invalid filter regex.", icon="⚠️"); nodes_to_display_names = set(G.nodes()) if G else set()

    highlight_neighbors_predecessors, highlight_neighbors_successors = set(), set()
    if highlight_node and G: highlight_neighbors_predecessors, highlight_neighbors_successors = get_neighbors(G, highlight_node)

    agraph_nodes, agraph_edges, agraph_edges_tuples = [], [], []
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
                 agraph_edges_tuples.append((u, v)); agraph_edges.append(Edge(source=u, target=v, color="#CCCCCC"))

    # (Agraph config remains the same)
    is_physics = st.session_state.dtg_layout == 'Physics'
    config = Config(width='100%', height=700, directed=True, physics=is_physics, hierarchical=not is_physics, highlightColor=HIGHLIGHT_COLOR, collapsible=False, node={'labelProperty':'label', 'size': 15},
        physics_config={'barnesHut': {'gravitationalConstant': -12000, 'centralGravity': 0.15, 'springLength': 200, 'springConstant': 0.06, 'damping': 0.1, 'avoidOverlap': 0.15}, 'minVelocity': 0.75} if is_physics else {},
        layout={'hierarchical': {'enabled': (not is_physics), 'sortMethod': 'directed', 'levelSeparation': 180, 'nodeSpacing': 150}} if not is_physics else {},
        interaction={'navigationButtons': True, 'keyboard': True, 'tooltipDelay': 300, 'hover': True})

    # (Graph display area layout remains the same)
    graph_col, info_col = st.columns([3, 1])
    with graph_col:
        st.caption("Graph View: Click/drag to pan, scroll/pinch to zoom. Select node in sidebar to highlight.")
        if not agraph_nodes and filter_term: st.warning(f"No nodes match filter: '{filter_term}'")
        elif not agraph_nodes: st.warning("No graph data to display (check AI response or filters).")
        else:
             try: agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)
             except Exception as e: st.error(f"Graph render error: {e}"); print(traceback.format_exc())

    with info_col:
        st.subheader("Details & Analysis")
        st.markdown("**Selected Term:**")
        if highlight_node: st.info(f"`{highlight_node}`")
        else: st.info("_Select node in sidebar_")
        st.caption("_Full definition text is not extracted._")
        st.markdown("---"); st.markdown("**Graph Analysis:**")
        # Display cycles and orphans using the data loaded into session state
        if st.session_state.dtg_cycles is not None:
             if st.session_state.dtg_cycles:
                  with st.expander(f"🚨 {len(st.session_state.dtg_cycles)} Circular Definition(s)", expanded=False):
                       for i, c in enumerate(st.session_state.dtg_cycles): st.markdown(f"- Cycle {i+1}: `{' → '.join(c)} → {c[0]}`")
             else: st.caption("✅ No circular definitions detected.")
        else: st.caption("Cycle analysis not available.")
        if st.session_state.dtg_orphans is not None:
             if st.session_state.dtg_orphans:
                  with st.expander(f"⚠️ {len(st.session_state.dtg_orphans)} Orphan Term(s)", expanded=False):
                       st.markdown(f"`{', '.join(sorted(st.session_state.dtg_orphans))}`")
                       st.caption("_Defined but not linked to/from any other defined term._")
             else: st.caption("✅ All defined terms linked.")
        else: st.caption("Orphan analysis not available.")
    st.divider()

    # (Export section remains the same)
    dot_lines = ["digraph G {"]; node_style_map = {n.id: f'[color="{n.color}", fontcolor="#000000"]' for n in agraph_nodes}
    for node_id in sorted(list(displayed_node_ids)):
        style = node_style_map.get(node_id, "")
        quoted_id = f'"{node_id}"' if re.search(r'\s|[^a-zA-Z0-9_]', node_id) else node_id
        dot_lines.append(f'  {quoted_id} {style};')
    for u, v in sorted(agraph_edges_tuples):
        quoted_u = f'"{u}"' if re.search(r'\s|[^a-zA-Z0-9_]', u) else u
        quoted_v = f'"{v}"' if re.search(r'\s|[^a-zA-Z0-9_]', v) else v
        dot_lines.append(f'  {quoted_u} -> {quoted_v};')
    dot_lines.append("}")
    generated_dot_code = "\n".join(dot_lines)
    st.subheader("Export Graph"); export_cols = st.columns(4); safe_filename_base = re.sub(r'[^\w\-]+', '_', display_name or "graph") # Use display_name
    with export_cols[0]: export_cols[0].download_button("📥 DOT (.dot)", generated_dot_code, f"{safe_filename_base}_graph.dot", "text/vnd.graphviz", use_container_width=True)
    with export_cols[1]:
         try: png_bytes = graphviz.Source(generated_dot_code, format='png').pipe(); export_cols[1].download_button("🖼️ PNG (.png)", png_bytes, f"{safe_filename_base}_graph.png", "image/png", use_container_width=True)
         except graphviz.backend.execute.ExecutableNotFound: export_cols[1].warning("Graphviz missing?", icon="⚠️", help="Graphviz executable not found. PNG/SVG export disabled.")
         except Exception as e: export_cols[1].warning(f"PNG ERR: {e}", icon="⚠️")
    with export_cols[2]:
         try: svg_bytes = graphviz.Source(generated_dot_code, format='svg').pipe(); export_cols[2].download_button("📐 SVG (.svg)", svg_bytes, f"{safe_filename_base}_graph.svg", "image/svg+xml", use_container_width=True)
         except graphviz.backend.execute.ExecutableNotFound: export_cols[2].warning("Graphviz missing?", icon="⚠️", help="Graphviz executable not found. PNG/SVG export disabled.")
         except Exception as e: export_cols[2].warning(f"SVG ERR: {e}", icon="⚠️")
    with export_cols[3]:
        if G:
            try:
                 df_deps = pd.DataFrame([{"Source Term": u, "Depends On (Target Term)": v} for u, v in agraph_edges_tuples])
                 if st.session_state.dtg_orphans:
                      orphan_df = pd.DataFrame([{"Source Term": orphan, "Depends On (Target Term)": "(Orphan)"} for orphan in sorted(st.session_state.dtg_orphans) if orphan in displayed_node_ids])
                      df_deps = pd.concat([df_deps, orphan_df], ignore_index=True)
                 df_deps = df_deps.sort_values(by=["Source Term", "Depends On (Target Term)"])
                 csv_output = df_deps.to_csv(index=False).encode('utf-8')
                 export_cols[3].download_button("📋 Deps (.csv)", csv_output, f"{safe_filename_base}_dependencies.csv", "text/csv", use_container_width=True, help="Exports source->target dependencies for the current filtered view.")
            except Exception as e: export_cols[3].warning(f"CSV ERR: {e}", icon="⚠️")
    with st.expander("View Generated DOT Code (for current view)"): st.code(generated_dot_code, language='dot')


elif st.session_state.dtg_error:
    # --- Error Display ---
    st.error(f"❌ Failed: {st.session_state.dtg_error}")
    if st.session_state.dtg_raw_ai_response:
        with st.expander("View Full Raw AI Response (for debugging)", expanded=False):
             st.text_area("Raw Response", st.session_state.dtg_raw_ai_response, height=400, disabled=True, label_visibility="collapsed")

elif not st.session_state.dtg_pdf_bytes and not st.session_state.dtg_viewing_history:
    # --- Initial State Message ---
    st.info("⬆️ Upload a document (PDF) using the sidebar or load from history to get started.")
elif not st.session_state.dtg_processing and not st.session_state.dtg_viewing_history:
    # --- Ready State Message ---
    st.info("⬆️ Ready to generate. Click the 'Generate & Analyze Graph' button in the sidebar.")
# No specific message needed if just viewing history and graph isn't displayed yet (should be covered by graph display logic)


# Footer
st.sidebar.markdown("---"); st.sidebar.markdown("Developed with Streamlit & Google Gemini")