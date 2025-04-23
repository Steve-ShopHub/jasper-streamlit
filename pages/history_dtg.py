# pages/history_dtg.py
# --- History Page for Defined Terms Grapher ---

import streamlit as st
import google.cloud.firestore
import google.oauth2.service_account
import traceback # For logging errors
from datetime import datetime # For timestamp formatting
import re # For quoting DOT node IDs

# --- Configuration ---
DTG_HISTORY_COLLECTION = "dtg_runs" # Firestore collection for DTG history
MAX_PREVIEW_ITEMS = 10 # Limit for lists in preview

st.set_page_config(layout="wide", page_title="DTG Analysis History")
st.title("📜 Defined Terms Grapher - History")

# --- Helper Functions ---
def format_timestamp(ts):
    """Safely format Firestore timestamp or string"""
    if isinstance(ts, datetime):
        # Format timestamp in a user-friendly way, maybe including timezone if available
        try:
            # Attempt to make it timezone-aware if possible (adjust as needed)
            local_ts = ts.astimezone()
            return local_ts.strftime('%Y-%m-%d %H:%M:%S %Z')
        except Exception:
            return ts.strftime('%Y-%m-%d %H:%M:%S') # Fallback to naive format
    elif isinstance(ts, google.cloud.firestore.SERVER_TIMESTAMP.__class__):
         return "Timestamp pending" # Should resolve after write
    elif isinstance(ts, str):
        return ts # Already a string
    elif ts is None:
        return "N/A"
    return str(ts) # Fallback for other unexpected types

def display_list_preview(items_list, title, max_items=MAX_PREVIEW_ITEMS):
    """Displays a preview of a list (e.g., orphans, cycles)."""
    if items_list:
        st.caption(f"**{title} ({len(items_list)}):**")
        preview_items = items_list[:max_items]
        # Display cycles or simple lists differently
        if title.lower().startswith("cycle"):
             for i, c in enumerate(preview_items):
                 st.code(f"Cycle {i+1}: {' → '.join(c)} → {c[0]}", language="text")
        else:
             st.code('\n'.join(map(str, preview_items)), language="text") # Display simple lists vertically in code block

        if len(items_list) > max_items:
            st.caption(f"... and {len(items_list) - max_items} more.")
    else:
        st.caption(f"**{title}:** None found or recorded.")


# --- Main History Loading Logic ---
try:
    # Initialize Firestore client
    if "firestore" not in st.secrets:
        st.error("Firestore credentials (`secrets.toml`) not found or incomplete. Please configure secrets.")
        st.stop()

    key_dict = st.secrets["firestore"]
    creds = google.oauth2.service_account.Credentials.from_service_account_info(key_dict)
    db = google.cloud.firestore.Client(credentials=creds)

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

            # Extract relevant data fields (handle missing keys gracefully)
            filename = run_data.get("filename", "N/A")
            timestamp = run_data.get("analysis_timestamp") # Might be Firestore Timestamp or string
            graph_data = run_data.get("graph_data", {}) # The dict with 'terms' and 'edges'
            gcs_pdf_path = run_data.get("gcs_pdf_path") # Assuming PDF stored in GCS
            cycles = run_data.get("cycles", [])
            orphans = run_data.get("orphans", [])
            error_message = run_data.get("error_message") # Check if an error was saved

            num_terms = len(graph_data.get("terms", [])) if isinstance(graph_data.get("terms"), list) else 0
            num_edges = len(graph_data.get("edges", [])) if isinstance(graph_data.get("edges"), list) else 0
            num_cycles = len(cycles) if isinstance(cycles, list) else 0
            num_orphans = len(orphans) if isinstance(orphans, list) else 0

            ts_str = format_timestamp(timestamp) # Use helper function

            # Construct expander title
            status_icon = "❌" if error_message else "📊"
            expander_title = f"{status_icon} {filename} | {num_terms} Terms, {num_edges} Edges | 🕒 {ts_str}"
            if not gcs_pdf_path and not error_message:
                expander_title += " (⚠️ PDF Link Missing)"

            with st.expander(expander_title):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.caption(f"Run ID: `{doc_id}`")
                    if error_message:
                         st.error(f"Analysis Failed: {error_message}")
                    else:
                         st.caption(f"Terms: {num_terms}, Edges: {num_edges}, Cycles: {num_cycles}, Orphans: {num_orphans}")

                    if not gcs_pdf_path and not error_message:
                        st.warning("Original PDF file link is missing. Cannot reload analysis.", icon="⚠️")
                    elif gcs_pdf_path:
                        st.caption(f"PDF Location: `{gcs_pdf_path}`") # Show GCS path if available


                with col2:
                    # Use a unique key prefix for the load button session state variable
                    load_button_key = f"load_dtg_{doc_id}"
                    # Disable load if PDF path is missing or if the run failed
                    load_disabled = not gcs_pdf_path or bool(error_message)
                    load_help = "Load this analysis graph and PDF." if not load_disabled else \
                                ("Cannot load (PDF link missing)." if not gcs_pdf_path else "Cannot load (Analysis failed).")

                    if st.button("🔄 Load this Graph", key=load_button_key, disabled=load_disabled, help=load_help, use_container_width=True):
                        # Set a specific session state key for the DTG app to detect
                        st.session_state['dtg_load_history_id'] = doc_id
                        # Clear any potential leftover state from other history loads if necessary (optional)
                        # if 'load_history_id' in st.session_state: del st.session_state['load_history_id']
                        st.switch_page("pages/defined_terms_graph.py") # Switch back to the DTG page

                if not error_message:
                    st.markdown("---")
                    # Display analysis previews
                    preview_col1, preview_col2 = st.columns(2)
                    with preview_col1:
                         display_list_preview(orphans, "Orphan Terms Preview")
                    with preview_col2:
                         display_list_preview(cycles, "Circular Definitions Preview")
                else:
                     # Optionally show raw graph data if analysis failed but data exists
                     if graph_data:
                          with st.expander("View Partial Graph Data (from failed run)"):
                              st.json(graph_data)


except KeyError as e:
     st.error(f"Firestore credentials key '{e}' not found in `secrets.toml`. Please configure secrets.")
except Exception as e:
    st.error(f"An error occurred while loading history from the database: {e}")
    st.error("Check Firestore connection, permissions, and collection name ('dtg_runs').")
    print(f"DTG History Load Error: {e}\n{traceback.format_exc()}")

st.caption("End of history list.")