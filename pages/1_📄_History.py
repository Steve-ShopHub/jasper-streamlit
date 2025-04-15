# 1_📄_History.py
# --- COMPLETE FILE (v2 - Improved Dataframe Preview) ---

import streamlit as st
import google.cloud.firestore
import google.oauth2.service_account
import pandas as pd
import traceback # For logging errors
from datetime import datetime # For timestamp formatting

# --- Define Column Order for Preview (Subset/Adaptation of Excel Order) ---
# Match desired preview columns, using names consistent with app.py's Excel export if possible
PREVIEW_COLUMN_ORDER = [
    "Question Number",
    "Question Category",
    "Question",
    "Answer",
    "Answer Justification",
    "Evidence References" # Consolidated evidence for preview
]

st.set_page_config(layout="wide", page_title="Analysis History")
st.title("📜 Analysis History")

def format_timestamp(ts):
    """Safely format Firestore timestamp or string"""
    if isinstance(ts, datetime):
        return ts.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(ts, google.cloud.firestore.SERVER_TIMESTAMP.__class__):
         # Handle server timestamps if they somehow appear directly
         # This is less common for retrieved data; usually converted to datetime
         return "Timestamp pending"
    return str(ts) # Fallback for unexpected types

def prepare_results_for_preview(results_list):
    """Processes the raw results list to create a preview-friendly list of dicts."""
    preview_data = []
    if not isinstance(results_list, list):
        return [] # Return empty if input is not a list

    for item in results_list:
        if not isinstance(item, dict):
            continue # Skip malformed items

        # Extract evidence references
        evidence_refs = []
        evidence_list = item.get("Evidence", [])
        if isinstance(evidence_list, list):
            for ev in evidence_list:
                if isinstance(ev, dict):
                    ref = ev.get("Clause Reference", "N/A")
                    evidence_refs.append(str(ref)) # Ensure string conversion

        preview_item = {
            "Question Number": item.get("Question Number"),
            "Question Category": item.get("Question Category", "Uncategorized"),
            "Question": item.get("Question", "N/A"),
            "Answer": item.get("Answer", "N/A"),
            "Answer Justification": item.get("Answer Justification", "N/A"),
            "Evidence References": "; ".join(evidence_refs) if evidence_refs else "N/A"
        }
        preview_data.append(preview_item)

    # Sort by Question Number for consistent preview order
    try:
        preview_data.sort(key=lambda x: x.get('Question Number') if isinstance(x.get('Question Number'), (int, float)) else float('inf'))
    except Exception as sort_err:
        st.warning(f"Could not sort history preview by question number: {sort_err}")

    return preview_data


try:
    # Initialize Firestore client
    if "firestore" not in st.secrets:
        st.error("Firestore credentials (`secrets.toml`) not found or incomplete. Please configure secrets.")
        st.stop()

    key_dict = st.secrets["firestore"]
    creds = google.oauth2.service_account.Credentials.from_service_account_info(key_dict)
    db = google.cloud.firestore.Client(credentials=creds)

    # Query the database
    runs_ref = db.collection("analysis_runs").order_by("analysis_timestamp", direction=google.cloud.firestore.Query.DESCENDING).limit(50)
    docs = runs_ref.stream()

    st.write("Showing latest analysis runs:")

    doc_list = list(docs)

    if not doc_list:
         st.info("No analysis history found in the database.")
    else:
        for doc in doc_list:
            run_data = doc.to_dict()
            doc_id = doc.id
            filename = run_data.get("filename", "N/A")
            timestamp = run_data.get("analysis_timestamp") # Might be Firestore Timestamp
            results_raw = run_data.get("results", []) # Raw results list
            gcs_path = run_data.get("gcs_pdf_path")

            ts_str = format_timestamp(timestamp) # Use helper function

            expander_title = f"📄 {filename} ({ts_str})"
            if not gcs_path:
                expander_title += " (⚠️ PDF Missing)"

            with st.expander(expander_title):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.caption(f"Document ID: `{doc_id}`")
                    num_results = len(results_raw) if isinstance(results_raw, list) else 0
                    st.caption(f"Number of result items: {num_results}")
                    if not gcs_path:
                        st.warning("Original PDF file link is missing. Cannot reload analysis.", icon="⚠️")

                with col2:
                    load_button_key = f"load_{doc_id}"
                    load_disabled = not gcs_path
                    load_help = "Load this analysis (results and PDF)." if not load_disabled else "Cannot load (PDF link missing)."

                    if st.button("🔄 Load this Analysis", key=load_button_key, disabled=load_disabled, help=load_help, use_container_width=True):
                        st.session_state['load_history_id'] = doc_id
                        st.switch_page("app.py")

                st.markdown("---")
                # Display results preview using the processed data
                if results_raw:
                     try:
                         # Prepare data specifically for the preview DataFrame
                         preview_list = prepare_results_for_preview(results_raw)

                         if preview_list:
                             df_preview = pd.DataFrame(preview_list)

                             # Ensure columns exist before selecting/reordering
                             final_preview_cols = [col for col in PREVIEW_COLUMN_ORDER if col in df_preview.columns]
                             df_preview = df_preview[final_preview_cols] # Reorder/select columns

                             st.dataframe(df_preview, height=300, use_container_width=True) # Adjust height as needed
                         else:
                             st.info("No processable results found in this record for preview.")

                     except Exception as df_err:
                         st.warning(f"Could not display results preview: {df_err}")
                         # st.json(results_raw) # Fallback to raw JSON if preview fails critically
                else:
                     st.info("No analysis results found in this record.")


except KeyError as e:
     st.error(f"Firestore credentials key '{e}' not found in `secrets.toml`. Please configure secrets.")
except Exception as e:
    st.error(f"An error occurred while loading history from the database: {e}")
    print(f"DB Load Error: {e}\n{traceback.format_exc()}")