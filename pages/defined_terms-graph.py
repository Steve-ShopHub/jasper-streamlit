# pages/defined_terms_graph.py
# --- COMPLETE FILE vX.Y+7 (Results JSON in GCS) ---

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
MODEL_NAME = "gemini-2.5-pro-preview-03-25" # Specific model requested
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
GCS_JSON_FOLDER_DTG = "dtg_results_json" # Folder within GCS bucket for results JSON
API_TIMEOUT = 900 # Seconds for Gemini API call

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
        # Try to find JSON within ```json ... ``` fences first
        match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE | re.DOTALL)
        if match:
            json_text = match.group(1).strip(); print(f"[{time.strftime('%H:%M:%S')}] Found JSON block via markdown fence.")
        else:
            # If no fence, try regex for a starting '{' or '[' followed by anything, ending in '}' or ']'
            json_start_match = re.search(r"^\s*(\{.*\}|\[.*\])\s*$", response_text, re.DOTALL)
            if json_start_match:
                json_text = json_start_match.group(0).strip(); print(f"[{time.strftime('%H:%M:%S')}] Found potential JSON block via full-string regex.")
            # Fallback: Check if the raw text starts/ends like JSON after stripping whitespace
            elif response_text.strip().startswith(("{", "[")) and response_text.strip().endswith(("}", "]")):
                json_text = response_text.strip(); print(f"[{time.strftime('%H:%M:%S')}] Assuming raw stripped response is JSON.")
            else:
                 error_msg = f"Response does not appear to contain a JSON object/array within ```json...``` or as the primary content. Raw text snippet: {response_text[:500]}..."; print(f"[{time.strftime('%H:%M:%S')}] Parsing failed: {error_msg}"); return None, error_msg

        if not json_text:
            error_msg = "Could not extract JSON content from the response."; print(f"[{time.strftime('%H:%M:%S')}] Parsing failed: {error_msg}"); return None, error_msg

        print(f"[{time.strftime('%H:%M:%S')}] Attempting json.loads() on extracted text (length: {len(json_text)})...");
        data = json.loads(json_text); print(f"[{time.strftime('%H:%M:%S')}] json.loads() successful. Validating structure...")

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

# --- Define Default State Structure ---
# Moved outside the function so reset_dtg_state can access keys
STATE_DEFAULTS = {
    'dtg_pdf_bytes': None, 'dtg_pdf_name': None,
    'dtg_processing': False, 'dtg_error': None,
    'dtg_graph_data': None, # Will be loaded from JSON if viewing history
    'dtg_nx_graph': None,
    'dtg_cycles': None, # Will be loaded from JSON if viewing history
    'dtg_orphans': None, # Will be loaded from JSON if viewing history
    'dtg_filter_term': "", 'dtg_highlight_node': None, 'dtg_layout': 'Physics',
    'dtg_raw_ai_response': None,
    'dtg_load_history_id': None, # Trigger for loading history
    'dtg_viewing_history': False, # Flag for history mode
    'dtg_history_filename': None, # Filename from history
    'dtg_history_timestamp': None, # Timestamp from history
    'dtg_history_model': None, # Model from history
    'dtg_history_gcs_pdf_path': None, # Store GCS path when viewing history
    'dtg_history_gcs_json_path': None, # Store GCS JSON path when viewing history
    # 'api_key': None, # API key handled separately
    'run_key': 0, # Used to ensure unique widget keys on rerun
}

# --- Initialize Session State ---
def initialize_dtg_state():
    """Initializes session state variables if they don't exist."""
    current_api_key = st.session_state.get('api_key', None) # Preserve API key
    for key, default_value in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(default_value) if isinstance(default_value, (dict, list, set, defaultdict)) else default_value
    # Ensure API key exists, even if None initially
    if 'api_key' not in st.session_state:
        st.session_state.api_key = current_api_key

def reset_dtg_state(preserve_api_key=True):
    """Resets session state, optionally preserving API key."""
    current_api_key = st.session_state.get('api_key', None) if preserve_api_key else None
    keys_to_reset = list(STATE_DEFAULTS.keys())
    if 'api_key' in keys_to_reset: keys_to_reset.remove('api_key')

    print(f"[{time.strftime('%H:%M:%S')}] Resetting state keys: {keys_to_reset}")
    for key in keys_to_reset:
        if key in st.session_state:
            try:
                del st.session_state[key]
            except Exception as e:
                print(f"Warning: Could not delete state key '{key}': {e}")

    initialize_dtg_state() # Re-initialize with defaults
    st.session_state.api_key = current_api_key
    print(f"[{time.strftime('%H:%M:%S')}] State reset complete. API Key preserved: {preserve_api_key}")

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
# MODIFIED upload_to_gcs to accept content_type
def upload_to_gcs(bucket_name, source_bytes, destination_blob_name, content_type='application/pdf', status_placeholder=None):
    """Uploads bytes to GCS bucket with specified content type."""
    file_type_desc = "PDF" if content_type == 'application/pdf' else "JSON results"
    if status_placeholder: status_placeholder.info(f"☁️ Uploading {file_type_desc} to GCS for history...")
    print(f"[{time.strftime('%H:%M:%S')}] Uploading {len(source_bytes)} bytes ({content_type}) to gs://{bucket_name}/{destination_blob_name}")
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(source_bytes, content_type=content_type, timeout=180) # Increased timeout
        gcs_path = f"gs://{bucket_name}/{destination_blob_name}"
        print(f"[{time.strftime('%H:%M:%S')}] Successfully uploaded {file_type_desc} to {gcs_path}")
        if status_placeholder: status_placeholder.info(f"☁️ {file_type_desc} saved to {gcs_path}")
        return gcs_path
    except Exception as e:
        error_msg = f"❌ GCS Upload Failed for {destination_blob_name} ({content_type}): {e}"
        if status_placeholder: status_placeholder.error(error_msg)
        print(f"[{time.strftime('%H:%M:%S')}] {error_msg}")
        traceback.print_exc()
        raise # Re-raise to indicate saving failed

# download_from_gcs remains the same (it just downloads bytes)
def download_from_gcs(bucket_name, source_blob_name):
    """Downloads a blob from the bucket. Returns bytes or None."""
    print(f"[{time.strftime('%H:%M:%S')}] Downloading from gs://{bucket_name}/{source_blob_name}")
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        download_bytes = blob.download_as_bytes(timeout=180)
        print(f"[{time.strftime('%H:%M:%S')}] Successfully downloaded {len(download_bytes)} bytes.")
        return download_bytes
    except NotFound:
        error_msg = f"❌ Error: File not found in GCS at gs://{bucket_name}/{source_blob_name}"
        st.error(error_msg)
        print(f"[{time.strftime('%H:%M:%S')}] {error_msg}")
        return None
    except Exception as e:
        error_msg = f"❌ Failed to download from GCS (gs://{bucket_name}/{source_blob_name}): {e}"
        st.error(error_msg)
        print(f"[{time.strftime('%H:%M:%S')}] {error_msg}")
        traceback.print_exc()
        return None

# --- Load History Logic ---
# Modified to load results from GCS JSON
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
            gcs_json_path = run_data.get("gcs_results_json_path") # Get JSON path
            hist_filename = run_data.get("filename", "N/A")
            hist_timestamp_obj = run_data.get("analysis_timestamp")
            hist_model = run_data.get("model_name", "Unknown")
            # Graph data, cycles, orphans are now loaded from the JSON path

            if not gcs_pdf_path:
                 st.error("❌ History record is missing the PDF file path (gcs_pdf_path). Cannot load.")
            elif not gcs_json_path: # Check for JSON path
                 st.error("❌ History record is missing the results JSON file path (gcs_results_json_path). Cannot load.")
            else:
                # Determine bucket and blob name from GCS PDF path
                hist_bucket_name = GCS_BUCKET_NAME # Default assumption
                hist_pdf_blob_name = None
                if gcs_pdf_path.startswith("gs://"):
                    try:
                        path_parts = gcs_pdf_path[5:].split("/", 1)
                        hist_bucket_name = path_parts[0] # Allow overriding bucket if specified
                        hist_pdf_blob_name = path_parts[1]
                    except IndexError:
                         st.error(f"❌ Invalid GCS PDF path format: {gcs_pdf_path}")
                         hist_pdf_blob_name = None # Prevent further processing
                else:
                     hist_pdf_blob_name = gcs_pdf_path # Assume relative path

                 # Determine bucket and blob name from GCS JSON path
                hist_json_blob_name = None
                if gcs_json_path.startswith("gs://"):
                    try:
                        path_parts = gcs_json_path[5:].split("/", 1)
                        # We assume JSON is in the same bucket as PDF for simplicity here,
                        # but path could override it if needed.
                        hist_json_blob_name = path_parts[1]
                    except IndexError:
                        st.error(f"❌ Invalid GCS JSON path format: {gcs_json_path}")
                        hist_json_blob_name = None # Prevent further processing
                else:
                     hist_json_blob_name = gcs_json_path # Assume relative path

                if hist_pdf_blob_name and hist_json_blob_name:
                    pdf_bytes_from_hist = None
                    results_data_from_hist = None
                    json_parse_error = None

                    # Download both PDF and JSON
                    with st.spinner(f"Downloading PDF & results from GCS..."):
                        pdf_bytes_from_hist = download_from_gcs(hist_bucket_name, hist_pdf_blob_name)
                        if pdf_bytes_from_hist:
                             json_bytes_from_hist = download_from_gcs(hist_bucket_name, hist_json_blob_name)
                             if json_bytes_from_hist:
                                 try:
                                     results_data_from_hist = json.loads(json_bytes_from_hist.decode('utf-8'))
                                     print(f"[{time.strftime('%H:%M:%S')}] Successfully parsed results JSON from GCS.")
                                 except json.JSONDecodeError as json_err:
                                     json_parse_error = f"Failed to parse results JSON from GCS: {json_err}"
                                     print(f"[{time.strftime('%H:%M:%S')}] {json_parse_error}")
                                 except Exception as parse_err:
                                     json_parse_error = f"Error processing results JSON from GCS: {parse_err}"
                                     print(f"[{time.strftime('%H:%M:%S')}] {json_parse_error}")
                             else:
                                 st.error("Failed to download the results JSON associated with this history entry.")

                    if pdf_bytes_from_hist and results_data_from_hist:
                        print(f"[{time.strftime('%H:%M:%S')}] PDF and results downloaded & parsed. Populating state from history.")
                        # Reset state before loading historical data
                        reset_dtg_state(preserve_api_key=True)

                        # Extract data from the loaded JSON
                        hist_graph_data = results_data_from_hist.get("graph_data")
                        hist_cycles = results_data_from_hist.get("cycles", [])
                        hist_orphans = results_data_from_hist.get("orphans", [])

                        if not hist_graph_data:
                             st.error("❌ Downloaded results JSON is missing the required 'graph_data' key.")
                             # Reset state if critical data is missing after download
                             reset_dtg_state(preserve_api_key=True)
                        else:
                            # Load historical data into session state
                            st.session_state.dtg_pdf_bytes = pdf_bytes_from_hist
                            st.session_state.dtg_pdf_name = hist_filename
                            st.session_state.dtg_graph_data = hist_graph_data
                            st.session_state.dtg_cycles = hist_cycles
                            st.session_state.dtg_orphans = hist_orphans
                            st.session_state.dtg_viewing_history = True
                            st.session_state.dtg_history_filename = hist_filename
                            st.session_state.dtg_history_model = hist_model # Store model used
                            st.session_state.dtg_history_gcs_pdf_path = gcs_pdf_path # Store paths for reference
                            st.session_state.dtg_history_gcs_json_path = gcs_json_path

                            # Format timestamp safely
                            try:
                                 if isinstance(hist_timestamp_obj, datetime):
                                     local_ts = hist_timestamp_obj.astimezone()
                                     hist_ts_str = local_ts.strftime('%Y-%m-%d %H:%M:%S %Z')
                                 elif isinstance(hist_timestamp_obj, str): hist_ts_str = hist_timestamp_obj
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
                        # Download or parse failed, error shown by download/parse steps
                        if json_parse_error: st.error(json_parse_error)
                        elif not pdf_bytes_from_hist: st.error("Failed to download the PDF associated with this history entry.")
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
        # Optionally display GCS paths for debugging
        with st.expander("Show GCS Paths (History)", expanded=False):
             st.caption(f"PDF: `{st.session_state.get('dtg_history_gcs_pdf_path', 'N/A')}`")
             st.caption(f"Results JSON: `{st.session_state.get('dtg_history_gcs_json_path', 'N/A')}`")
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
         st.switch_page("defined_terms_graph.py")
     except Exception as e:
          st.sidebar.error(f"Could not navigate to history page: {e}")


# --- Main Area ---
# Modified processing block to save results JSON to GCS
if st.session_state.dtg_processing:
    status_placeholder = st.empty()
    full_response_text = ""
    process_start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] ===== Starting Generation Process =====")

    temp_file_path = None
    uploaded_file_ref = None
    gemini_file_upload_successful = False
    persistent_gcs_pdf_path = None # GCS path for PDF history saving
    persistent_gcs_json_path = None # GCS path for results JSON history saving
    firestore_run_id = str(uuid.uuid4())
    run_error_message = None
    current_analysis_timestamp = datetime.now(timezone.utc)
    response = None # Initialize response variable
    current_graph_data = None # Store results temporarily before saving
    current_cycles = None
    current_orphans = None


    # Wrap the entire processing in a try...finally to ensure cleanup
    try:
        with st.spinner(f"⚙️ Analyzing '{st.session_state.dtg_pdf_name}'..."):
            # ... (API Config, Temp File Creation, Gemini Upload - remain the same) ...
            status_placeholder.info("🔑 Configuring API...")
            print(f"[{time.strftime('%H:%M:%S')}] Configuring GenAI client...")
            try:
                genai.configure(api_key=st.session_state.api_key)
            except Exception as config_err:
                 run_error_message = f"Failed to configure API key: {config_err}"
                 print(f"[{time.strftime('%H:%M:%S')}] CONFIG ERROR: {run_error_message}") # Log config error
                 raise ValueError(run_error_message)

            pdf_bytes = st.session_state.dtg_pdf_bytes
            pdf_name = st.session_state.dtg_pdf_name
            if not pdf_bytes or not pdf_name:
                run_error_message = "Cannot proceed without uploaded PDF data."
                print(f"[{time.strftime('%H:%M:%S')}] PRE-CHECK ERROR: {run_error_message}") # Log pre-check error
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
                print(f"[{time.strftime('%H:%M:%S')}] TEMP FILE ERROR: {run_error_message}") # Log temp file error
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
                        print(f"[{time.strftime('%H:%M:%S')}] GEMINI UPLOAD ERROR (FATAL): {run_error_message}") # Log fatal upload error
                        traceback.print_exc(); raise ValueError(run_error_message)
                    elif attempt < MAX_UPLOAD_RETRIES:
                        print(f"[{time.strftime('%H:%M:%S')}] Retrying Gemini upload in {UPLOAD_RETRY_DELAY}s...")
                        status_placeholder.warning(f"☁️ Gemini upload attempt {attempt+1} failed, retrying...")
                        time.sleep(UPLOAD_RETRY_DELAY)
                    else: # Final attempt failed
                        run_error_message = f"Failed to upload file to Gemini after {MAX_UPLOAD_RETRIES + 1} attempts: {upload_err}"
                        print(f"[{time.strftime('%H:%M:%S')}] GEMINI UPLOAD ERROR (FINAL): {run_error_message}") # Log final upload error
                        traceback.print_exc(); raise ValueError(run_error_message)

            if not gemini_file_upload_successful or not uploaded_file_ref:
                 run_error_message = run_error_message or "Gemini file upload process failed unexpectedly."
                 print(f"[{time.strftime('%H:%M:%S')}] GEMINI UPLOAD STATE ERROR: {run_error_message}") # Log state error
                 raise ValueError(run_error_message)

            # --- Prompt & API Call (Non-Streaming) ---
            status_placeholder.info("🧠 Preparing analysis prompt...")
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

            status_placeholder.info("📞 Calling Gemini API (waiting for full response)...")
            print(f"[{time.strftime('%H:%M:%S')}] Sending request to Gemini API with file URI: {uploaded_file_ref.uri}")
            api_call_start_time = time.time()

            try:
                print(f"[{time.strftime('%H:%M:%S')}] Attempting model.generate_content (non-streaming)...")
                response = model.generate_content(
                    contents=[uploaded_file_ref, prompt_instructions], # Pass file ref and prompt
                    generation_config=generation_config, safety_settings=safety_settings,
                    request_options={'timeout': API_TIMEOUT}, stream=False
                )
                api_call_end_time = time.time(); print(f"[{time.strftime('%H:%M:%S')}] Gemini API call completed. Duration: {api_call_end_time - api_call_start_time:.2f}s.")
                if response is None: raise ValueError("Gemini API call returned None unexpectedly.")
                print(f"[{time.strftime('%H:%M:%S')}] Response object received: Type={type(response)}")
            except google.api_core.exceptions.GoogleAPIError as api_err:
                 api_call_end_time = time.time(); run_error_message = f"Gemini API call failed: {type(api_err).__name__}: {api_err}"; print(f"[{time.strftime('%H:%M:%S')}] GEMINI API ERROR: {run_error_message}"); print(f"[{time.strftime('%H:%M:%S')}] API call duration before error: {api_call_end_time - api_call_start_time:.2f}s."); traceback.print_exc(); raise ValueError(run_error_message) from api_err
            except Exception as call_err:
                 api_call_end_time = time.time(); run_error_message = f"Unexpected error during Gemini API call: {type(call_err).__name__}: {call_err}"; print(f"[{time.strftime('%H:%M:%S')}] UNEXPECTED API CALL ERROR: {run_error_message}"); print(f"[{time.strftime('%H:%M:%S')}] API call duration before error: {api_call_end_time - api_call_start_time:.2f}s."); traceback.print_exc(); raise ValueError(run_error_message) from call_err

            # --- Process Full Response ---
            status_placeholder.info("📄 Processing Gemini's full response...")
            print(f"[{time.strftime('%H:%M:%S')}] Attempting to access response.text...")
            try:
                # ... (Debugging print statements for response parts - can be kept or removed) ...
                full_response_text = response.text
                st.session_state.dtg_raw_ai_response = full_response_text # Save raw response
                print(f"[{time.strftime('%H:%M:%S')}] Successfully accessed response.text. Length: {len(full_response_text)} chars.")
            except ValueError as e: # Catch safety blocks etc.
                 # ... (Error handling for accessing response.text remains the same) ...
                 print(f"[{time.strftime('%H:%M:%S')}] ValueError accessing response.text: {e}")
                 block_reason = "Unknown"; feedback_str = "N/A"
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback: feedback_str = str(response.prompt_feedback); block_reason = feedback_str[:150]
                 run_error_message = f"Failed to get text from Gemini response (ValueError, likely safety block). Feedback: {block_reason}"; print(f"[{time.strftime('%H:%M:%S')}] RESPONSE TEXT ERROR (ValueError): {run_error_message}"); traceback.print_exc(); raise ValueError(run_error_message) from e
            except AttributeError as ae: # Catch unexpected response structure
                 # ... (Error handling for accessing response.text remains the same) ...
                 print(f"[{time.strftime('%H:%M:%S')}] AttributeError accessing response.text: {ae}"); feedback_info = "N/A"
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback: feedback_info = str(response.prompt_feedback)[:150]
                 run_error_message = f"Gemini response object missing 'text' attribute. Feedback: {feedback_info}"; print(f"[{time.strftime('%H:%M:%S')}] RESPONSE TEXT ERROR (AttributeError): {run_error_message}"); traceback.print_exc(); raise ValueError(run_error_message) from ae
            except Exception as resp_err: # Catch other errors
                 # ... (Error handling for accessing response.text remains the same) ...
                 print(f"[{time.strftime('%H:%M:%S')}] Unexpected error accessing response.text: {type(resp_err).__name__}: {resp_err}")
                 run_error_message = f"An unexpected error occurred while accessing the Gemini response text: {resp_err}"; print(f"[{time.strftime('%H:%M:%S')}] RESPONSE TEXT ERROR (Exception): {run_error_message}"); traceback.print_exc(); raise ValueError(run_error_message) from resp_err

            # --- Response Parsing & Graph Building ---
            if not full_response_text.strip(): run_error_message = "AI returned an empty response."; print(f"[{time.strftime('%H:%M:%S')}] PARSING ERROR: {run_error_message}"); raise ValueError(run_error_message)

            graph_data, error_msg = parse_ai_response(full_response_text)
            if error_msg: run_error_message = f"Failed to parse AI response: {error_msg}"; raise ValueError(run_error_message)
            current_graph_data = graph_data # Store locally first
            st.toast("Term names & links extracted!", icon="📊")
            status_placeholder.info("⚙️ Analyzing graph structure...")

            print(f"[{time.strftime('%H:%M:%S')}] Building NetworkX graph...")
            graph_build_start_time = time.time()
            # Build graph from locally stored data
            temp_nx_graph = build_networkx_graph(current_graph_data)
            graph_build_end_time = time.time()

            if temp_nx_graph is None: run_error_message = "Failed to build graph from parsed data."; print(f"[{time.strftime('%H:%M:%S')}] GRAPH BUILD ERROR: {run_error_message}"); raise ValueError(run_error_message)
            print(f"[{time.strftime('%H:%M:%S')}] Graph built. Nodes: {len(temp_nx_graph.nodes())}, Edges: {len(temp_nx_graph.edges())}. Duration: {graph_build_end_time - graph_build_start_time:.2f}s")

            print(f"[{time.strftime('%H:%M:%S')}] Finding cycles & orphans...")
            analysis_start_time = time.time()
            # Find cycles/orphans from locally built graph
            current_cycles = find_cycles(temp_nx_graph)
            current_orphans = find_orphans(temp_nx_graph)
            analysis_end_time = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Found {len(current_cycles) if current_cycles is not None else 'N/A'} cycles.")
            print(f"[{time.strftime('%H:%M:%S')}] Found {len(current_orphans) if current_orphans is not None else 'N/A'} orphans.")
            print(f"[{time.strftime('%H:%M:%S')}] Graph analysis duration: {analysis_end_time - analysis_start_time:.2f}s")
            st.toast("Graph analysis complete.", icon="🔬")
            status_placeholder.info("💾 Saving analysis results to history...")

            # --- Upload ORIGINAL PDF to GCS for History ---
            gcs_pdf_blob_name = f"{GCS_PDF_FOLDER_DTG}/{current_analysis_timestamp.strftime('%Y%m%d')}/{firestore_run_id}_{safe_original_filename}"
            try:
                # Pass 'application/pdf' content type explicitly
                persistent_gcs_pdf_path = upload_to_gcs(GCS_BUCKET_NAME, pdf_bytes, gcs_pdf_blob_name, content_type='application/pdf', status_placeholder=status_placeholder)
            except Exception as gcs_pdf_upload_err:
                 print(f"[{time.strftime('%H:%M:%S')}] GCS PDF UPLOAD FAILED: {gcs_pdf_upload_err}")
                 st.warning(f"☁️ GCS PDF Upload Failed: {gcs_pdf_upload_err}. Cannot save history.", icon="⚠️")
                 run_error_message = f"GCS PDF Upload Failed: {gcs_pdf_upload_err}" # Record error
                 raise # Stop processing as history save depends on PDF upload

            # --- Upload RESULTS JSON to GCS for History ---
            if persistent_gcs_pdf_path: # Only proceed if PDF upload succeeded
                gcs_json_blob_name = f"{GCS_JSON_FOLDER_DTG}/{current_analysis_timestamp.strftime('%Y%m%d')}/{firestore_run_id}_results.json"
                results_data_to_save = {
                    "graph_data": current_graph_data,
                    "cycles": current_cycles if current_cycles is not None else [],
                    "orphans": current_orphans if current_orphans is not None else []
                }
                try:
                    results_json_bytes = json.dumps(results_data_to_save, indent=2).encode('utf-8')
                    # Pass 'application/json' content type
                    persistent_gcs_json_path = upload_to_gcs(GCS_BUCKET_NAME, results_json_bytes, gcs_json_blob_name, content_type='application/json', status_placeholder=status_placeholder)
                except Exception as gcs_json_upload_err:
                    print(f"[{time.strftime('%H:%M:%S')}] GCS JSON UPLOAD FAILED: {gcs_json_upload_err}")
                    st.error(f"❌ Failed to upload results JSON to GCS: {gcs_json_upload_err}. History save failed.")
                    run_error_message = f"GCS JSON Upload Failed: {gcs_json_upload_err}" # Record error
                    # Don't raise here, allow cleanup, but Firestore save will be skipped

            # --- Save run metadata to Firestore ---
            if persistent_gcs_pdf_path and persistent_gcs_json_path:
                data_to_save_firestore = {
                    "filename": pdf_name,
                    "analysis_timestamp": current_analysis_timestamp, # Use UTC timestamp object
                    "gcs_pdf_path": persistent_gcs_pdf_path,
                    "gcs_results_json_path": persistent_gcs_json_path, # Store path to JSON
                    # DO NOT store large graph_data, cycles, orphans here anymore
                    "model_name": MODEL_NAME,
                    "error_message": None # Explicitly null for success
                }
                try:
                    db.collection(DTG_HISTORY_COLLECTION).document(firestore_run_id).set(data_to_save_firestore)
                    print(f"[{time.strftime('%H:%M:%S')}] Successfully saved run metadata {firestore_run_id} to Firestore.")
                    status_placeholder.success("✅ Analysis complete and saved to history!")
                    st.toast("💾 Analysis saved to history.", icon="✅")
                    # Analysis succeeded, populate session state for immediate display
                    st.session_state.dtg_graph_data = current_graph_data
                    st.session_state.dtg_cycles = current_cycles
                    st.session_state.dtg_orphans = current_orphans
                    st.session_state.dtg_nx_graph = temp_nx_graph

                except Exception as db_err:
                    st.error(f"❌ Failed to save results metadata to Firestore: {db_err}")
                    print(f"[{time.strftime('%H:%M:%S')}] FIRESTORE SAVE ERROR: {db_err}")
                    traceback.print_exc()
                    run_error_message = f"Firestore Save Error: {db_err}" # Record error
                    # Don't set dtg_error here, as analysis itself succeeded, but saving failed

            elif run_error_message is None: # If JSON upload failed but analysis was otherwise ok
                 status_placeholder.warning("✅ Analysis complete, but not saved to history due to results JSON upload error.")


    # --- Exception Handling (Outer Block) ---
    except (ValueError, types.StopCandidateException, google.api_core.exceptions.GoogleAPIError) as e:
        # Handles errors raised explicitly or caught API errors
        if not run_error_message: run_error_message = f"Analysis Error: {type(e).__name__}: {e}"
        st.session_state.dtg_error = run_error_message
        print(f"[{time.strftime('%H:%M:%S')}] ANALYSIS FAILED (Outer Except Block 1): {run_error_message}")
        if not isinstance(e, ValueError): traceback.print_exc()
        # Attempt to save failure record (without large data)
        try:
            status_placeholder.warning("💾 Saving error details to history...")
            error_data = {
                "filename": st.session_state.get('dtg_pdf_name', 'Unknown'),
                "analysis_timestamp": current_analysis_timestamp,
                "gcs_pdf_path": persistent_gcs_pdf_path, # Save PDF path if it was uploaded
                "gcs_results_json_path": None, # No JSON path on failure
                "model_name": MODEL_NAME,
                "error_message": run_error_message[:1000]
            }
            db.collection(DTG_HISTORY_COLLECTION).document(firestore_run_id).set(error_data)
            print(f"[{time.strftime('%H:%M:%S')}] Saved error record {firestore_run_id} to Firestore.")
            st.toast("💾 Error details saved to history.", icon="💾")
        except Exception as db_err:
             print(f"[{time.strftime('%H:%M:%S')}] CRITICAL: Failed to save error record to Firestore: {db_err}")
             st.warning("Could not save error details to history.", icon="⚠️")

    except Exception as e:
        # Catch any other unexpected errors
        if not run_error_message: run_error_message = f"An unexpected critical error occurred: {type(e).__name__}: {e}"
        st.session_state.dtg_error = run_error_message
        print(f"[{time.strftime('%H:%M:%S')}] UNEXPECTED CRITICAL ERROR (Outer Except Block 2): {run_error_message}")
        traceback.print_exc()
        # Attempt to save failure record
        try:
            status_placeholder.warning("💾 Saving error details to history...")
            error_data = {
                "filename": st.session_state.get('dtg_pdf_name', 'Unknown'),
                "analysis_timestamp": current_analysis_timestamp,
                "gcs_pdf_path": persistent_gcs_pdf_path, # Save PDF path if available
                "gcs_results_json_path": None,
                "model_name": MODEL_NAME, "error_message": run_error_message[:1000]
            }
            db.collection(DTG_HISTORY_COLLECTION).document(firestore_run_id).set(error_data)
            print(f"[{time.strftime('%H:%M:%S')}] Saved error record {firestore_run_id} to Firestore.")
            st.toast("💾 Error details saved to history.", icon="💾")
        except Exception as db_err:
             print(f"[{time.strftime('%H:%M:%S')}] CRITICAL: Failed to save error record to Firestore: {db_err}")
             st.warning("Could not save error details to history.", icon="⚠️")

    # --- Cleanup Phase ---
    finally:
        print(f"[{time.strftime('%H:%M:%S')}] Entering cleanup phase...")
        # Display final status message before clearing placeholder
        final_save_failed = (persistent_gcs_pdf_path and not persistent_gcs_json_path) or (run_error_message and "save" in run_error_message.lower())
        if st.session_state.dtg_error:
             status_placeholder.error(f"❌ Failed: {st.session_state.dtg_error}")
        elif final_save_failed:
             status_placeholder.warning("✅ Analysis complete, but failed to save results to cloud history.")
        elif st.session_state.dtg_graph_data: # Check if graph data was populated (indicates success)
             status_placeholder.success("✅ Analysis complete and saved to history!") # Reiterate success if needed
        # else: No explicit message if processing failed very early

        print(f"[{time.strftime('%H:%M:%S')}] Cleaning up temporary resources...")
        # ... (Cleanup logic for Gemini temp file and local temp file remains the same) ...
        if uploaded_file_ref and hasattr(uploaded_file_ref, 'name'):
            try:
                print(f"[{time.strftime('%H:%M:%S')}] Deleting temporary cloud file: {uploaded_file_ref.name}...")
                delete_start = time.time(); genai.delete_file(name=uploaded_file_ref.name); delete_end = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] Temp cloud file deleted. Duration: {delete_end - delete_start:.2f}s")
            except Exception as delete_err:
                print(f"[{time.strftime('%H:%M:%S')}] WARNING: Failed to delete temp cloud file '{uploaded_file_ref.name}': {delete_err}")
                st.sidebar.warning(f"Temp cloud cleanup issue: {delete_err}", icon="⚠️")
        elif gemini_file_upload_successful: print(f"[{time.strftime('%H:%M:%S')}] WARNING: Temp cloud file reference available but might not have been deleted.")
        else: print(f"[{time.strftime('%H:%M:%S')}] Skipping temp cloud file deletion (was not uploaded successfully).")

        if temp_file_path and os.path.exists(temp_file_path):
            try:
                print(f"[{time.strftime('%H:%M:%S')}] Deleting local temporary file: {temp_file_path}...")
                os.remove(temp_file_path); print(f"[{time.strftime('%H:%M:%S')}] Local temp file deleted.")
            except Exception as local_delete_err: print(f"[{time.strftime('%H:%M:%S')}] WARNING: Failed to delete local temp file '{temp_file_path}': {local_delete_err}")
        else: print(f"[{time.strftime('%H:%M:%S')}] Skipping local temp file deletion (path: {temp_file_path}).")

        if full_response_text and not st.session_state.dtg_raw_ai_response: st.session_state.dtg_raw_ai_response = full_response_text

        st.session_state.dtg_processing = False
        process_end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] ===== Generation Process Ended =====")
        print(f"[{time.strftime('%H:%M:%S')}] Total processing duration (incl. save/cleanup): {process_end_time - process_start_time:.2f} seconds.")
        print(f"[{time.strftime('%H:%M:%S')}] Triggering rerun after final status display.")
        time.sleep(2.0 if st.session_state.dtg_error or final_save_failed else 1.0)
        status_placeholder.empty() # Clear status *before* rerun
        st.rerun()

# --- Display Results Area (No changes needed here) ---
# This part works correctly whether the data was generated live or loaded from history (via GCS JSON)
elif st.session_state.dtg_graph_data:
    # ... (Graph display logic remains exactly the same as previous version) ...
    display_name = st.session_state.dtg_history_filename if st.session_state.dtg_viewing_history else st.session_state.dtg_pdf_name
    st.subheader(f"📊 Interactive Graph & Analysis for '{display_name}'")
    G = st.session_state.dtg_nx_graph
    filter_term = st.session_state.dtg_filter_term
    highlight_node = st.session_state.dtg_highlight_node

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
            safe_node_id_label = node_id # Use original name for label
            agraph_nodes.append(Node(id=node_id, label=safe_node_id_label, color=node_color, size=node_size, font={'color': "#000000"}))
        for u, v in G.edges():
            if u in displayed_node_ids and v in displayed_node_ids:
                 agraph_edges_tuples.append((u, v)); agraph_edges.append(Edge(source=u, target=v, color="#CCCCCC"))

    is_physics = st.session_state.dtg_layout == 'Physics'
    config = Config(width='100%', height=700, directed=True, physics=is_physics, hierarchical=not is_physics, highlightColor=HIGHLIGHT_COLOR, collapsible=False, node={'labelProperty':'label', 'size': 15},
        physics_config={'barnesHut': {'gravitationalConstant': -12000, 'centralGravity': 0.15, 'springLength': 200, 'springConstant': 0.06, 'damping': 0.1, 'avoidOverlap': 0.15}, 'minVelocity': 0.75} if is_physics else {},
        layout={'hierarchical': {'enabled': (not is_physics), 'sortMethod': 'directed', 'levelSeparation': 180, 'nodeSpacing': 150}} if not is_physics else {},
        interaction={'navigationButtons': True, 'keyboard': True, 'tooltipDelay': 300, 'hover': True})

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
        style = node_style_map.get(node_id, ""); quoted_id = f'"{node_id}"' if re.search(r'\s|[^a-zA-Z0-9_]', node_id) else node_id
        dot_lines.append(f'  {quoted_id} {style};')
    for u, v in sorted(agraph_edges_tuples):
        quoted_u = f'"{u}"' if re.search(r'\s|[^a-zA-Z0-9_]', u) else u; quoted_v = f'"{v}"' if re.search(r'\s|[^a-zA-Z0-9_]', v) else v
        dot_lines.append(f'  {quoted_u} -> {quoted_v};')
    dot_lines.append("}")
    generated_dot_code = "\n".join(dot_lines)
    st.subheader("Export Graph"); export_cols = st.columns(4); safe_filename_base = re.sub(r'[^\w\-]+', '_', display_name or "graph")
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
                      visible_orphans = [orphan for orphan in st.session_state.dtg_orphans if orphan in displayed_node_ids]
                      if visible_orphans: orphan_df = pd.DataFrame([{"Source Term": orphan, "Depends On (Target Term)": "(Orphan)"} for orphan in sorted(visible_orphans)]); df_deps = pd.concat([df_deps, orphan_df], ignore_index=True)
                 if not df_deps.empty:
                      df_deps = df_deps.sort_values(by=["Source Term", "Depends On (Target Term)"]); csv_output = df_deps.to_csv(index=False).encode('utf-8')
                      export_cols[3].download_button("📋 Deps (.csv)", csv_output, f"{safe_filename_base}_dependencies.csv", "text/csv", use_container_width=True, help="Exports source->target dependencies for the current filtered view.")
                 else: export_cols[3].button("📋 Deps (.csv)", disabled=True, help="No dependencies/orphans to export in current view.", use_container_width=True)
            except Exception as e: export_cols[3].warning(f"CSV ERR: {e}", icon="⚠️")
        else: export_cols[3].button("📋 Deps (.csv)", disabled=True, help="Graph data not available for CSV export.", use_container_width=True)
    with st.expander("View Generated DOT Code (for current view)"): st.code(generated_dot_code, language='dot')


# --- Error Display / Initial State Messages ---
elif st.session_state.dtg_error:
    # Error is displayed by the finally block, but this remains as a fallback
    if st.session_state.dtg_raw_ai_response:
        with st.expander("View Full Raw AI Response (for debugging)", expanded=False):
             st.text_area("Raw Response", st.session_state.dtg_raw_ai_response, height=400, disabled=True, label_visibility="collapsed")

elif not st.session_state.dtg_pdf_bytes and not st.session_state.dtg_viewing_history:
    # --- Initial State Message ---
    st.info("⬆️ Upload a document (PDF) using the sidebar or load from history to get started.")
elif not st.session_state.dtg_processing and not st.session_state.dtg_viewing_history:
    # --- Ready State Message ---
    st.info("⬆️ Ready to generate. Click the 'Generate & Analyze Graph' button in the sidebar.")


# Footer
st.sidebar.markdown("---"); st.sidebar.markdown("Developed with Streamlit & Google Gemini")