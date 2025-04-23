# pages/history_dtg.py
# --- History Page for Defined Terms Grapher (Loads JSON from GCS) ---

import streamlit as st
import google.cloud.firestore
import google.oauth2.service_account
import traceback # For logging errors
from datetime import datetime, timezone # Use timezone aware
import re # For quoting DOT node IDs
import json # For parsing downloaded JSON
import time # For delays

# --- Configuration ---
DTG_HISTORY_COLLECTION = "dtg_runs" # Firestore collection for DTG history
MAX_PREVIEW_ITEMS = 10 # Limit for lists in preview
# Assume same GCS bucket and credentials as main app
try:
    GCS_BUCKET_NAME = st.secrets["gcs_config"]["bucket_name"]
except KeyError:
    st.error("GCS bucket name configuration (`secrets.toml`) missing.")
    st.stop()

st.set_page_config(layout="wide", page_title="DTG Analysis History")
st.title("📜 Defined Terms Grapher - History")

# --- Firestore Client Initialization (moved outside main try-except) ---
db = None
try:
    if "firestore" not in st.secrets:
        st.error("Firestore credentials (`secrets.toml`) not found or incomplete. Please configure secrets.")
        st.stop()

    key_dict = st.secrets["firestore"]
    creds = google.oauth2.service_account.Credentials.from_service_account_info(key_dict)
    db = google.cloud.firestore.Client(credentials=creds)
except Exception as e:
    st.error(f"Failed to initialize Firestore Client: {e}")
    print(f"Firestore Init Error: {e}\n{traceback.format_exc()}")
    st.stop()


# --- Helper Functions ---
def format_timestamp(ts):
    """Safely format Firestore timestamp or string"""
    if isinstance(ts, datetime):
        try:
            local_ts = ts.astimezone() # Attempt to make it timezone-aware
            return local_ts.strftime('%Y-%m-%d %H:%M:%S %Z')
        except Exception:
            return ts.strftime('%Y-%m-%d %H:%M:%S') # Fallback
    elif isinstance(ts, google.cloud.firestore.SERVER_TIMESTAMP.__class__):
         return "Timestamp pending"
    elif isinstance(ts, str): return ts
    elif ts is None: return "N/A"
    return str(ts)

def display_list_preview(items_list, title, max_items=MAX_PREVIEW_ITEMS):
    """Displays a preview of a list (e.g., orphans, cycles)."""
    if items_list is None: # Handle case where data might be explicitly None
         st.caption(f"**{title}:** Data not available or not loaded.")
         return

    if not isinstance(items_list, list):
         st.caption(f"**{title}:** Invalid data format (expected list).")
         return

    item_count = len(items_list)
    if item_count > 0:
        st.caption(f"**{title} ({item_count}):**")
        preview_items = items_list[:max_items]
        # Display cycles or simple lists differently
        if title.lower().startswith("cycle"):
             for i, c in enumerate(preview_items):
                 # Ensure cycle elements are strings for join
                 cycle_str = ' → '.join(map(str, c)) if c else ""
                 first_elem_str = str(c[0]) if c else ""
                 st.code(f"Cycle {i+1}: {cycle_str} → {first_elem_str}", language="text")
        else:
             st.code('\n'.join(map(str, preview_items)), language="text") # Display simple lists vertically

        if item_count > max_items:
            st.caption(f"... and {item_count - max_items} more.")
    else:
        st.caption(f"**{title}:** None found or recorded.")


# --- Main History Loading Logic ---
if db: # Proceed only if Firestore client is initialized
    try:
        # Query the database for DTG runs
        runs_ref = db.collection(DTG_HISTORY_COLLECTION).order_by(
            "analysis_timestamp", direction=google.cloud.firestore.Query.DESCENDING
        ).limit(50) # Limit to latest 50 runs
        docs = runs_ref.stream()

        st.write("Showing latest Defined Terms Grapher analysis runs:")

        doc_list = list(docs) # Convert stream iterator to list

        if not doc_list:
             st.info(f"No analysis history found in the '{DTG_HISTORY_COLLECTION}' collection.")
        else:
            for doc in doc_list:
                run_data = doc.to_dict()
                doc_id = doc.id

                # Extract relevant data fields from Firestore
                filename = run_data.get("filename", "N/A")
                timestamp = run_data.get("analysis_timestamp")
                gcs_pdf_path = run_data.get("gcs_pdf_path")
                gcs_json_path = run_data.get("gcs_results_json_path") # Path to results JSON
                error_message = run_data.get("error_message")

                # Initialize preview data - loaded later if needed
                num_terms = "N/A"
                num_edges = "N/A"
                num_cycles = "N/A"
                num_orphans = "N/A"
                cycles_preview = None
                orphans_preview = None

                # Load results from JSON only if needed for preview (or full load)
                # For now, we only need counts for the title if available easily
                # Let's simplify the title - counts will be shown inside
                ts_str = format_timestamp(timestamp)

                # Construct expander title
                status_icon = "❌" if error_message else "📊"
                expander_title = f"{status_icon} {filename} | 🕒 {ts_str}"
                if not gcs_pdf_path and not error_message:
                    expander_title += " (⚠️ PDF Link Missing)"
                elif not gcs_json_path and not error_message:
                     expander_title += " (⚠️ Results JSON Link Missing)"

                with st.expander(expander_title):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.caption(f"Run ID: `{doc_id}`")
                        if error_message:
                             st.error(f"Analysis Failed: {error_message}")
                        # else:
                        #      st.caption(f"Terms: {num_terms}, Edges: {num_edges}, Cycles: {num_cycles}, Orphans: {num_orphans}") # Defer counts

                        if not gcs_pdf_path and not error_message:
                            st.warning("Original PDF file link is missing. Cannot reload analysis.", icon="⚠️")
                        elif gcs_pdf_path:
                            st.caption(f"PDF Location: `{gcs_pdf_path}`")

                        if not gcs_json_path and not error_message:
                             st.warning("Results JSON file link is missing. Cannot reload or preview analysis.", icon="⚠️")
                        elif gcs_json_path:
                             st.caption(f"Results Location: `{gcs_json_path}`")


                    with col2:
                        # Use a unique key prefix for the load button session state variable
                        load_button_key = f"load_dtg_{doc_id}"
                        # Disable load if PDF or JSON path is missing or if the run failed
                        load_disabled = not gcs_pdf_path or not gcs_json_path or bool(error_message)
                        load_help = "Load this analysis graph and PDF." if not load_disabled else \
                                    ("Cannot load (PDF link missing)." if not gcs_pdf_path else \
                                    ("Cannot load (Results JSON link missing)." if not gcs_json_path else \
                                    "Cannot load (Analysis failed)."))

                        if st.button("🔄 Load this Graph", key=load_button_key, disabled=load_disabled, help=load_help, use_container_width=True):
                            # Set a specific session state key for the DTG app to detect
                            st.session_state['dtg_load_history_id'] = doc_id
                            st.switch_page("pages/defined_terms_graph.py") # Switch back to the DTG page

                    # --- Preview Section ---
                    if not error_message and gcs_json_path:
                        st.markdown("---")
                        # Display analysis previews (requires loading JSON)
                        preview_placeholder = st.empty() # Placeholder for loading message/results
                        if st.button("🔍 Show Previews", key=f"preview_{doc_id}"):
                            with preview_placeholder.container():
                                with st.spinner("Loading analysis results from GCS..."):
                                     hist_bucket_name_preview = GCS_BUCKET_NAME
                                     hist_json_blob_name_preview = None
                                     if gcs_json_path.startswith("gs://"):
                                         try:
                                             path_parts = gcs_json_path[5:].split("/", 1)
                                             hist_bucket_name_preview = path_parts[0]
                                             hist_json_blob_name_preview = path_parts[1]
                                         except IndexError: st.error(f"Invalid JSON Path: {gcs_json_path}")
                                     else: hist_json_blob_name_preview = gcs_json_path

                                     results_data = None
                                     if hist_json_blob_name_preview:
                                         # Reuse the download function from main app (import if needed, or redefine)
                                         # For simplicity, let's assume it's available via import or similar context
                                         # Or just copy the GCS download part here:
                                         try:
                                             # Assume storage_client is initialized if db is
                                             bucket = storage_client.bucket(hist_bucket_name_preview)
                                             blob = bucket.blob(hist_json_blob_name_preview)
                                             json_bytes = blob.download_as_bytes(timeout=60)
                                             results_data = json.loads(json_bytes.decode('utf-8'))
                                         except Exception as download_err:
                                             st.error(f"Failed to download/parse results JSON for preview: {download_err}")

                                if results_data:
                                     graph_data_preview = results_data.get("graph_data")
                                     cycles_preview = results_data.get("cycles") # Load full lists
                                     orphans_preview = results_data.get("orphans")

                                     if graph_data_preview:
                                         num_terms = len(graph_data_preview.get("terms", [])) if isinstance(graph_data_preview.get("terms"), list) else '?'
                                         num_edges = len(graph_data_preview.get("edges", [])) if isinstance(graph_data_preview.get("edges"), list) else '?'
                                         st.caption(f"Terms: {num_terms}, Edges: {num_edges}")
                                     else:
                                         st.warning("Graph data missing in results file.")

                                     preview_col1, preview_col2 = st.columns(2)
                                     with preview_col1:
                                         display_list_preview(orphans_preview, "Orphan Terms Preview")
                                     with preview_col2:
                                         display_list_preview(cycles_preview, "Circular Definitions Preview")
                                else:
                                     st.warning("Could not load results data for preview.")


    except KeyError as e:
         st.error(f"Firestore credentials key '{e}' not found in `secrets.toml`. Please configure secrets.")
    except Exception as e:
        st.error(f"An error occurred while loading history from the database: {e}")
        st.error("Check Firestore connection, permissions, and collection name ('dtg_runs').")
        print(f"DTG History Load Error: {e}\n{traceback.format_exc()}")

    st.caption("End of history list.")
else:
    st.error("Firestore client could not be initialized. Cannot load history.")