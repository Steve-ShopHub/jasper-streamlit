# pages/defined_terms_graph.py
import streamlit as st
import google.generativeai as genai
from google.generativeai import types
import fitz  # PyMuPDF for PDF text extraction
import re
import os
import traceback
import time

# --- Configuration ---
MODEL_NAME = "gemini-1.5-pro-latest"
PAGE_TITLE = "Defined Terms Relationship Grapher"
PAGE_ICON = "🔗"

# --- Set Page Config ---
st.set_page_config(layout="wide", page_title=PAGE_TITLE, page_icon=PAGE_ICON)

# --- Inject custom CSS (Optional - Copy from app.py if needed) ---
# st.markdown("""<style>...</style>""", unsafe_allow_html=True)

# --- Helper Function for Text Extraction ---
@st.cache_data(show_spinner="Extracting text from PDF...")
def extract_text_from_pdf(pdf_bytes):
    """Extracts text from PDF bytes using PyMuPDF."""
    if not pdf_bytes:
        return None, "No PDF file provided."
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text", sort=True) # sort=True helps with reading order
            text += "\n--- Page Break --- \n" # Add page breaks for context
        doc.close()
        if not text.strip():
            return None, "Could not extract any text from the PDF."
        return text, None
    except Exception as e:
        error_msg = f"Error extracting text: {e}"
        print(traceback.format_exc())
        return None, error_msg

# --- Helper Function to Extract DOT Code ---
def extract_dot_code(raw_response_text):
    """Extracts DOT code, potentially removing markdown fences."""
    if not raw_response_text:
        return None, "AI response was empty."

    # Check for markdown code fence (dot or graphviz)
    match = re.search(r"```(?:dot|graphviz)?\s*([\s\S]*?)\s*```", raw_response_text, re.IGNORECASE | re.DOTALL)
    if match:
        dot_code = match.group(1).strip()
        return dot_code, None # Found fenced code
    else:
        # No fence found, assume the whole response might be DOT code,
        # but check if it starts like a digraph.
        if raw_response_text.strip().lower().startswith("digraph"):
            return raw_response_text.strip(), None
        else:
            # If it doesn't look like DOT, return error and the raw text for debugging
            return None, f"Could not find DOT code fences (```dot ... ```) and response doesn't start with 'digraph'. Raw response:\n\n{raw_response_text[:500]}..."

# --- Initialize Session State for this page ---
def initialize_dtg_state():
    if 'dtg_pdf_bytes' not in st.session_state:
        st.session_state.dtg_pdf_bytes = None
    if 'dtg_pdf_name' not in st.session_state:
        st.session_state.dtg_pdf_name = None
    if 'dtg_extracted_text' not in st.session_state:
        st.session_state.dtg_extracted_text = None
    if 'dtg_dot_code' not in st.session_state:
        st.session_state.dtg_dot_code = None
    if 'dtg_processing' not in st.session_state:
        st.session_state.dtg_processing = False
    if 'dtg_error' not in st.session_state:
        st.session_state.dtg_error = None
    # Use the main app's API key if available, otherwise None
    if 'api_key' not in st.session_state:
         st.session_state.api_key = None # Ensure it exists

initialize_dtg_state()

# --- Streamlit UI ---

# --- Header ---
# Borrowing logo logic if available, otherwise just title
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Get parent directory
LOGO_FILE = "jasper-logo-1.png"
LOGO_PATH = os.path.join(APP_DIR, LOGO_FILE)

header_cols = st.columns([1, 5])
with header_cols[0]:
    if os.path.exists(LOGO_PATH):
        try:
            from PIL import Image # Import here if not already global
            logo = Image.open(LOGO_PATH)
            st.image(logo, width=80, caption=None, output_format='PNG', alt="Logo")
        except Exception as img_err:
            st.warning(f"Could not load logo: {img_err}")

with header_cols[1]:
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.caption("Upload a document to visualize the relationships between its defined terms.")

st.divider()

# --- Sidebar Controls ---
st.sidebar.title("Controls")
st.sidebar.markdown("---")

# API Key Input (Reuse from main app's state or request here)
api_key_input = st.sidebar.text_input(
    "Google AI Gemini API Key*",
    type="password",
    key="api_key_sidebar_dtg", # Use a distinct key if needed
    value=st.session_state.get("api_key", ""), # Default to value from main app state if set
    help="Your Gemini API key. Needed to generate the graph.",
)
if api_key_input and api_key_input != st.session_state.api_key:
    st.session_state.api_key = api_key_input # Update the shared state

if not st.session_state.api_key:
    st.sidebar.warning("API Key required to generate graph.", icon="🔑")

# File Uploader
st.sidebar.markdown("### 1. Upload Document")
uploaded_file_obj = st.sidebar.file_uploader(
    "Upload Document (PDF recommended)*",
    type=["pdf", "txt"], # Allow text files too
    key="dtg_pdf_uploader"
)

# Process uploaded file
if uploaded_file_obj is not None:
    # Check if it's a new file
    if uploaded_file_obj.name != st.session_state.get('dtg_pdf_name'):
        st.session_state.dtg_pdf_bytes = uploaded_file_obj.getvalue()
        st.session_state.dtg_pdf_name = uploaded_file_obj.name
        st.session_state.dtg_extracted_text = None # Clear old text
        st.session_state.dtg_dot_code = None      # Clear old results
        st.session_state.dtg_error = None         # Clear old errors
        st.session_state.dtg_processing = False
        st.toast(f"📄 File '{st.session_state.dtg_pdf_name}' loaded.", icon="✅")
        # Immediately try text extraction for PDF
        if uploaded_file_obj.type == "application/pdf":
             extracted_text, error_msg = extract_text_from_pdf(st.session_state.dtg_pdf_bytes)
             if error_msg:
                 st.session_state.dtg_error = f"Failed to extract text from PDF: {error_msg}"
             else:
                 st.session_state.dtg_extracted_text = extracted_text
                 st.toast("Text extracted from PDF.", icon="📝")
        elif uploaded_file_obj.type == "text/plain":
             # Assume UTF-8 encoding for text files
             try:
                  st.session_state.dtg_extracted_text = st.session_state.dtg_pdf_bytes.decode('utf-8')
                  st.toast("Text loaded from TXT file.", icon="📝")
             except Exception as e:
                  st.session_state.dtg_error = f"Failed to read text file: {e}"
        else: # Should not happen with type restriction, but just in case
             st.session_state.dtg_error = f"Unsupported file type: {uploaded_file_obj.type}"

        st.rerun() # Update UI after processing upload

# Display error if extraction failed during upload
if st.session_state.dtg_error and not st.session_state.dtg_processing:
    st.error(st.session_state.dtg_error)

# Generation Button
st.sidebar.markdown("### 2. Generate Graph")
can_generate = (st.session_state.api_key is not None and
                st.session_state.dtg_pdf_bytes is not None and
                st.session_state.dtg_extracted_text is not None and # Must have text
                not st.session_state.dtg_processing)

generate_button_tooltip = ""
if st.session_state.dtg_processing: generate_button_tooltip = "Processing..."
elif not st.session_state.api_key: generate_button_tooltip = "Enter API Key"
elif not st.session_state.dtg_pdf_bytes: generate_button_tooltip = "Upload a document"
elif not st.session_state.dtg_extracted_text: generate_button_tooltip = "Could not extract text from document"
else: generate_button_tooltip = "Generate the defined terms graph using Gemini"

if st.sidebar.button("✨ Generate Relationship Graph", key="dtg_generate", disabled=not can_generate, help=generate_button_tooltip, use_container_width=True, type="primary"):
    st.session_state.dtg_processing = True
    st.session_state.dtg_dot_code = None
    st.session_state.dtg_error = None
    st.rerun() # Show spinner and start processing

# --- Main Area: Processing and Display ---
if st.session_state.dtg_processing:
    # Display spinner and status message
    status_placeholder = st.empty()
    with st.spinner(f"⚙️ Analyzing '{st.session_state.dtg_pdf_name}' with Gemini... This may take a moment."):
        status_placeholder.info("🧠 Asking Gemini to identify terms, relationships, and generate DOT graph...")
        try:
            # 1. Configure GenAI
            genai.configure(api_key=st.session_state.api_key)

            # 2. Prepare the Prompt (using the user's detailed instructions)
            #    Combine the extracted text with the specific instructions.
            document_text = st.session_state.dtg_extracted_text
            # Truncate text if it's excessively long? Gemini 1.5 Pro has 1M token context, less likely needed.
            # Consider adding a warning if text is huge, but proceed for now.

            prompt_instructions = f"""
For the provided document text below, consider the defined terms, their definitions, and usages throughout the document, and how these link together with other defined terms used or defined in the document itself.

Produce a directed graph of how these terms link together, producing the output **strictly in "DOT" digraph format** (suitable for graphviz). Only output the DOT code itself, nothing else.

**Instructions for DOT Graph Generation:**
1.  **Type:** Use `digraph G {{ ... }}`.
2.  **Nodes:** Use the exact defined terms as node names (e.g., `"Defined Term"`).
3.  **Edges:** Create a directed edge ` "Term A" -> "Term B"; ` if "Term B" is used within the definition of "Term A".
4.  **Cycles:** Cycles are allowed where present in the document's definitions.
5.  **Unlinked Defined Terms:** Where a defined term is present and defined in the document but its definition does not use other defined terms, AND it is not used in the definition of other terms within the document, include it as a standalone node (no incoming or outgoing edges based on definition links).
6.  **Externally Defined Terms (Blue Nodes):** Where there is a defined term *used* within a definition, but that term itself does *not* appear to be defined *within this document*, format the node for that term as ` "External Term" [color=blue, fontcolor=blue]; `. This requires inferring if a definition is missing in the provided text.
7.  **Implied Formatting:** Allow graphviz to handle layout algorithms and default node/edge styles unless specified above. Do not add unnecessary formatting commands.
8.  **Output:** Ensure the final output is ONLY the valid DOT code, starting with `digraph` and ending with `}}`. Do not include any explanatory text before or after the code block.

**Document Text:**
--- Start Document Text ---
{document_text}
--- End Document Text ---

**Final Output (DOT Code Only):**
"""

            # 3. Set up the Model and Call API
            # Note: Schema/JSON mode isn't suitable here as we want direct DOT output.
            model = genai.GenerativeModel(MODEL_NAME)
            generation_config = types.GenerationConfig(
                # We cannot enforce DOT schema directly, rely on prompt instructions.
                # Set temperature slightly lower for more deterministic structure following.
                temperature=0.1,
                # Increase timeout for potentially long documents/analysis
            )
            safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]

            status_placeholder.info("📞 Calling Gemini API...")
            response = model.generate_content(
                contents=prompt_instructions,
                generation_config=generation_config,
                safety_settings=safety_settings,
                request_options={'timeout': 600} # 10 min timeout
            )

            status_placeholder.info("📄 Processing Gemini response...")
            # 4. Process Response
            generated_text = response.text
            dot_code, error_msg = extract_dot_code(generated_text)

            if error_msg:
                st.session_state.dtg_error = error_msg
                st.session_state.dtg_dot_code = None # Ensure no stale code
            else:
                st.session_state.dtg_dot_code = dot_code
                st.session_state.dtg_error = None
                st.toast("Graph generated successfully!", icon="🎉")

        except types.StopCandidateException as sce:
            st.session_state.dtg_error = f"Generation Stopped Error: {sce}. The response might be incomplete or blocked."
            print(traceback.format_exc())
        except google.api_core.exceptions.GoogleAPIError as api_err:
            st.session_state.dtg_error = f"Google API Error: {api_err}. Please check your API key, quota, and permissions."
            print(traceback.format_exc())
        except Exception as e:
            st.session_state.dtg_error = f"An unexpected error occurred: {e}"
            print(traceback.format_exc())
        finally:
            st.session_state.dtg_processing = False
            status_placeholder.empty() # Clear status message
            st.rerun() # Rerun to show results or error

elif st.session_state.dtg_dot_code:
    # --- Display Results ---
    st.subheader(f"📊 Generated Relationship Graph for '{st.session_state.dtg_pdf_name}'")

    # Display the DOT code in an expander
    with st.expander("View Generated DOT Code"):
        st.code(st.session_state.dtg_dot_code, language='dot')

    # Display the graph using Graphviz
    st.markdown("---")
    st.markdown("**Rendered Graph:**")
    try:
        # Add note about Graphviz dependency for local run
        st.caption("Rendering using Graphviz. Ensure Graphviz is installed if running locally (e.g., `sudo apt install graphviz` or `brew install graphviz`).")
        st.graphviz_chart(st.session_state.dtg_dot_code)
        st.success("Graph rendered successfully.")
    except Exception as graphviz_err:
        st.error(f"⚠️ Failed to render graph using `st.graphviz_chart`. Error: {graphviz_err}")
        st.error("Please ensure Graphviz is correctly installed and accessible in your system's PATH.")
        st.warning("You can still copy the DOT code above and use an external Graphviz tool (like an online editor) to visualize it.")
        print(traceback.format_exc())

elif st.session_state.dtg_error:
    # Display error if processing failed
    st.error(f"❌ Failed to generate graph: {st.session_state.dtg_error}")

elif not st.session_state.dtg_pdf_bytes:
    # Initial state message
    st.info("⬆️ Upload a document (PDF or TXT) using the sidebar to get started.")

else:
    # PDF is uploaded, text extracted (or failed), but generation not triggered/completed
    st.info("⬆️ Ready to generate. Click the 'Generate Relationship Graph' button in the sidebar.")


# --- Footer/Links (Optional) ---
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with Streamlit & Google Gemini")