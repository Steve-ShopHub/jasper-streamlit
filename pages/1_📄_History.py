# Example in pages/1_📄_History.py
import streamlit as st
import google.cloud.firestore
import google.oauth2.service_account
import pandas as pd

st.set_page_config(layout="wide", page_title="Analysis History")
st.title("📜 Analysis History")

try:
    # Initialize Firestore client (reuse logic or put in a helper)
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
            filename = run_data.get("filename", "N/A")
            timestamp = run_data.get("analysis_timestamp", "N/A")
            results = run_data.get("results", [])

            ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp)

            with st.expander(f"📄 {filename} ({ts_str})"):
                st.dataframe(pd.DataFrame(results)) # Display results in a table
                # Or display in a more custom way if needed
                # st.json(results)

except KeyError:
     st.error("Firestore credentials (`secrets.toml`) not found or incomplete. Please configure secrets.")
except Exception as e:
    st.error(f"Error loading history from database: {e}")
    print(f"DB Load Error: {e}\n{traceback.format_exc()}")