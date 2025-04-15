import streamlit as st
import google.cloud.firestore
import google.oauth2.service_account
import pandas as pd
import traceback # For logging errors

st.set_page_config(layout="wide", page_title="Analysis History")
st.title("📜 Analysis History")

try:
    # Initialize Firestore client (reuse logic or put in a helper if needed elsewhere)
    # Ensure secrets are loaded correctly
    if "firestore" not in st.secrets:
        st.error("Firestore credentials (`secrets.toml`) not found or incomplete. Please configure secrets.")
        st.stop()

    key_dict = st.secrets["firestore"]
    creds = google.oauth2.service_account.Credentials.from_service_account_info(key_dict)
    db = google.cloud.firestore.Client(credentials=creds)

    # Query the database
    runs_ref = db.collection("analysis_runs").order_by("analysis_timestamp", direction=google.cloud.firestore.Query.DESCENDING).limit(50) # Limit results initially
    docs = runs_ref.stream()

    st.write("Showing latest analysis runs:")

    doc_list = list(docs) # Convert iterator to list to check if empty

    if not doc_list:
         st.info("No analysis history found in the database.")
    else:
        for doc in doc_list:
            run_data = doc.to_dict()
            doc_id = doc.id # Get the document ID
            filename = run_data.get("filename", "N/A")
            timestamp = run_data.get("analysis_timestamp", "N/A")
            results = run_data.get("results", [])
            gcs_path = run_data.get("gcs_pdf_path") # Check if PDF path exists

            ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp)

            expander_title = f"📄 {filename} ({ts_str})"
            if not gcs_path:
                expander_title += " (⚠️ PDF Missing)"

            with st.expander(expander_title):
                col1, col2 = st.columns([3, 1]) # Layout for info and button

                with col1:
                    st.caption(f"Document ID: `{doc_id}`")
                    num_results = len(results)
                    st.caption(f"Number of result items: {num_results}")
                    if not gcs_path:
                        st.warning("Original PDF file link is missing from this record. Cannot reload analysis.", icon="⚠️")

                with col2:
                    # Button to load this history item
                    load_button_key = f"load_{doc_id}"
                    # Disable button if PDF path is missing
                    load_disabled = not gcs_path
                    load_help = "Load this analysis (results and PDF) into the main view." if not load_disabled else "Cannot load analysis because the PDF link is missing."

                    if st.button("🔄 Load this Analysis", key=load_button_key, disabled=load_disabled, help=load_help, use_container_width=True):
                        st.session_state['load_history_id'] = doc_id # Set the ID to load in session state
                        st.switch_page("app.py") # Switch back to the main app page

                st.markdown("---")
                # Display results preview (optional, can be intensive if large)
                if results:
                     try:
                         # Display a sample or summary instead of full dataframe if needed
                         df_preview = pd.DataFrame(results)
                         st.dataframe(df_preview, height=200) # Limit height
                     except Exception as df_err:
                         st.warning(f"Could not display results preview: {df_err}")
                         # st.json(results) # Fallback to JSON view if dataframe fails
                else:
                     st.info("No analysis results found in this record.")


except KeyError as e:
     st.error(f"Firestore credentials key '{e}' not found in `secrets.toml`. Please configure secrets.")
except Exception as e:
    st.error(f"An error occurred while loading history from the database: {e}")
    print(f"DB Load Error: {e}\n{traceback.format_exc()}")