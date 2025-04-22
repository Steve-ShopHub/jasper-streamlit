# pages/defined_terms_graph.py
import streamlit as st
import google.generativeai as genai
from google.generativeai import types
import fitz  # PyMuPDF for PDF text extraction
import re
import os
import traceback
import time
import io # For download button

# --- NEW IMPORTS ---
import graphviz # Python graphviz library for parsing DOT
from streamlit_agraph import agraph, Node, Edge, Config
# --- END NEW IMPORTS ---


# --- Configuration ---
MODEL_NAME = "gemini-1.5-pro-latest"
PAGE_TITLE = "Defined Terms Relationship Grapher"
PAGE_ICON = "🔗"

# --- Set Page Config ---
st.set_page_config(layout="wide", page_title=PAGE_TITLE, page_icon=PAGE_ICON)

# --- Inject custom CSS (Optional - Copy from app.py if needed) ---
# Example for centering the graph container if needed, adjust as necessary
st.markdown("""
<style>
    /* Optional: Style the container holding the agraph component */
    .agraph-container {
        border: 1px solid #ddd; /* Add a border */
        border-radius: 5px;
        height: 600px; /* Set a fixed height */
        width: 100%; /* Make it responsive width-wise */
        overflow: hidden; /* Hide internal scrollbars if VisJS handles it */
        margin-bottom: 1rem; /* Add space below */
    }
    /* Ensure Streamlit containers don't add excessive padding */
     div[data-testid="stVerticalBlock"] > div[style*="gap: 1rem;"] {
        gap: 0.5rem !important;
     }
</style>
""", unsafe_allow_html=True)


# --- Helper Function for Text Extraction (Unchanged) ---
@st.cache_data(show_spinner="Extracting text from PDF...")
def extract_text_from_pdf(pdf_bytes):
    # ... (keep the existing function code) ...
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

# --- Helper Function to Extract DOT Code (Unchanged) ---
def extract_dot_code(raw_response_text):
    # ... (keep the existing function code) ...
    if not raw_response_text:
        return None, "AI response was empty."
    match = re.search(r"```(?:dot|graphviz)?\s*([\s\S]*?)\s*```", raw_response_text, re.IGNORECASE | re.DOTALL)
    if match:
        dot_code = match.group(1).strip()
        return dot_code, None # Found fenced code
    else:
        if raw_response_text.strip().lower().startswith("digraph"):
            return raw_response_text.strip(), None
        else:
            return None, f"Could not find DOT code fences (```dot ... ```) and response doesn't start with 'digraph'. Raw response:\n\n{raw_response_text[:500]}..."

# --- Initialize Session State for this page (Unchanged) ---
def initialize_dtg_state():
    # ... (keep the existing function code) ...
    if 'dtg_pdf_bytes' not in st.session_state: st.session_state.dtg_pdf_bytes = None
    if 'dtg_pdf_name' not in st.session_state: st.session_state.dtg_pdf_name = None
    if 'dtg_extracted_text' not in st.session_state: st.session_state.dtg_extracted_text = None
    if 'dtg_dot_code' not in st.session_state: st.session_state.dtg_dot_code = None
    if 'dtg_processing' not in st.session_state: st.session_state.dtg_processing = False
    if 'dtg_error' not in st.session_state: st.session_state.dtg_error = None
    if 'api_key' not in st.session_state: st.session_state.api_key = None

initialize_dtg_state()

# --- Streamlit UI ---

# --- Header (Unchanged) ---
# ... (keep the existing header code) ...
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


# --- Sidebar Controls (Unchanged) ---
st.sidebar.title("Controls")
st.sidebar.markdown("---")
# API Key Input
api_key_input = st.sidebar.text_input(
    "Google AI Gemini API Key*", type="password", key="api_key_sidebar_dtg",
    value=st.session_state.get("api_key", ""),
    help="Your Gemini API key. Needed to generate the graph.",
)
if api_key_input and api_key_input != st.session_state.api_key:
    st.session_state.api_key = api_key_input
if not st.session_state.api_key:
    st.sidebar.warning("API Key required to generate graph.", icon="🔑")
# File Uploader
st.sidebar.markdown("### 1. Upload Document")
uploaded_file_obj = st.sidebar.file_uploader(
    "Upload Document (PDF recommended)*", type=["pdf", "txt"], key="dtg_pdf_uploader"
)
# Process uploaded file logic (Unchanged)
if uploaded_file_obj is not None:
    if uploaded_file_obj.name != st.session_state.get('dtg_pdf_name'):
        st.session_state.dtg_pdf_bytes = uploaded_file_obj.getvalue()
        st.session_state.dtg_pdf_name = uploaded_file_obj.name
        st.session_state.dtg_extracted_text = None; st.session_state.dtg_dot_code = None
        st.session_state.dtg_error = None; st.session_state.dtg_processing = False
        st.toast(f"📄 File '{st.session_state.dtg_pdf_name}' loaded.", icon="✅")
        if uploaded_file_obj.type == "application/pdf":
             extracted_text, error_msg = extract_text_from_pdf(st.session_state.dtg_pdf_bytes)
             if error_msg: st.session_state.dtg_error = f"Failed to extract text from PDF: {error_msg}"
             else: st.session_state.dtg_extracted_text = extracted_text; st.toast("Text extracted from PDF.", icon="📝")
        elif uploaded_file_obj.type == "text/plain":
             try: st.session_state.dtg_extracted_text = st.session_state.dtg_pdf_bytes.decode('utf-8'); st.toast("Text loaded from TXT file.", icon="📝")
             except Exception as e: st.session_state.dtg_error = f"Failed to read text file: {e}"
        else: st.session_state.dtg_error = f"Unsupported file type: {uploaded_file_obj.type}"
        st.rerun()
if st.session_state.dtg_error and not st.session_state.dtg_processing: st.error(st.session_state.dtg_error)
# Generation Button
st.sidebar.markdown("### 2. Generate Graph")
can_generate = (st.session_state.api_key is not None and
                st.session_state.dtg_pdf_bytes is not None and
                st.session_state.dtg_extracted_text is not None and
                not st.session_state.dtg_processing)
# ... (generate_button_tooltip logic unchanged) ...
generate_button_tooltip = ""
if st.session_state.dtg_processing: generate_button_tooltip = "Processing..."
elif not st.session_state.api_key: generate_button_tooltip = "Enter API Key"
elif not st.session_state.dtg_pdf_bytes: generate_button_tooltip = "Upload a document"
elif not st.session_state.dtg_extracted_text: generate_button_tooltip = "Could not extract text from document"
else: generate_button_tooltip = "Generate the defined terms graph using Gemini"

if st.sidebar.button("✨ Generate Relationship Graph", key="dtg_generate", disabled=not can_generate, help=generate_button_tooltip, use_container_width=True, type="primary"):
    st.session_state.dtg_processing = True; st.session_state.dtg_dot_code = None
    st.session_state.dtg_error = None; st.rerun()
# --- END Sidebar ---


# --- Main Area: Processing and Display ---
if st.session_state.dtg_processing:
    # --- Processing Logic (Largely Unchanged, still generates DOT code) ---
    status_placeholder = st.empty()
    with st.spinner(f"⚙️ Analyzing '{st.session_state.dtg_pdf_name}' with Gemini... This may take a moment."):
        status_placeholder.info("🧠 Asking Gemini to identify terms, relationships, and generate DOT graph...")
        try:
            # Configure GenAI
            genai.configure(api_key=st.session_state.api_key)
            # Prepare Prompt
            document_text = st.session_state.dtg_extracted_text
            prompt_instructions = f"""
For the provided document text below, consider the defined terms, their definitions, and usages throughout the document, and how these link together with other defined terms used or defined in the document itself.

Produce a directed graph of how these terms link together, producing the output **strictly in "DOT" digraph format** (suitable for graphviz). Only output the DOT code itself, nothing else.

**Instructions for DOT Graph Generation:**
1.  **Type:** Use `digraph G {{ ... }}`.
2.  **Nodes:** Use the exact defined terms as node names (e.g., `"Defined Term"`). Ensure node names are properly quoted if they contain spaces or special characters.
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
            # Set up Model and Call API
            model = genai.GenerativeModel(MODEL_NAME)
            generation_config = types.GenerationConfig(temperature=0.1)
            safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
            status_placeholder.info("📞 Calling Gemini API...")
            response = model.generate_content(
                contents=prompt_instructions, generation_config=generation_config,
                safety_settings=safety_settings, request_options={'timeout': 600}
            )
            status_placeholder.info("📄 Processing Gemini response...")
            # Process Response
            generated_text = response.text
            dot_code, error_msg = extract_dot_code(generated_text)
            if error_msg:
                st.session_state.dtg_error = error_msg
                st.session_state.dtg_dot_code = None
            else:
                st.session_state.dtg_dot_code = dot_code
                st.session_state.dtg_error = None
                st.toast("Graph generated successfully!", icon="🎉")
        # ... (keep existing exception handling) ...
        except types.StopCandidateException as sce: st.session_state.dtg_error = f"Generation Stopped Error: {sce}. Response might be incomplete/blocked."; print(traceback.format_exc())
        except google.api_core.exceptions.GoogleAPIError as api_err: st.session_state.dtg_error = f"Google API Error: {api_err}. Check API key/quota/permissions."; print(traceback.format_exc())
        except Exception as e: st.session_state.dtg_error = f"An unexpected error occurred: {e}"; print(traceback.format_exc())
        finally:
            st.session_state.dtg_processing = False
            status_placeholder.empty()
            st.rerun()

elif st.session_state.dtg_dot_code:
    # --- Display Results ---
    st.subheader(f"📊 Interactive Relationship Graph for '{st.session_state.dtg_pdf_name}'")
    st.caption("Click and drag to pan, use mouse wheel or pinch to zoom.")

    # --- NEW: Parse DOT and Prepare for Agraph ---
    nodes = []
    edges = []
    try:
        # Use graphviz library to parse the DOT string
        # Using graph_attr, node_attr, edge_attr defaults from graphviz if not overridden in DOT
        parsed_graph = graphviz.Source(st.session_state.dtg_dot_code, format='dot') # Just parse, don't render yet

        # Iterate through nodes defined in the DOT source
        # Note: ._node method might be internal, but often used. If breaks, need alternative parsing.
        # A more robust way might involve rendering to DOT output and parsing *that*, but let's try this.
        # We need access to attributes defined *in* the DOT source.
        # Re-parsing the source string might be more direct if Source object hides attributes.
        # Let's try a regex approach *on the dot code* just for node attributes we care about (color).

        node_colors = {}
        # Simple regex to find nodes with color=blue attribute
        # Handles quotes around node names and potential whitespace variations
        # Example: "External Term" [color=blue, fontcolor=blue];
        # Or:      "External Term" [ color = blue , fontcolor=blue ] ;
        node_attr_pattern = re.compile(r'^\s*("?.*?"?)\s+\[(.*?)\]\s*;', re.MULTILINE)
        color_pattern = re.compile(r'(?:color|fontcolor)\s*=\s*blue')

        for match in node_attr_pattern.finditer(st.session_state.dtg_dot_code):
            node_name = match.group(1).strip('"') # Get name without quotes
            attributes = match.group(2)
            if color_pattern.search(attributes):
                 node_colors[node_name] = "blue" # Store that this node should be blue

        # Now use the parsed graph object to get nodes and edges reliably
        # Need to render to dot language output to easily iterate nodes/edges
        dot_source = parsed_graph.source
        # Basic parsing - assumes simple "A" -> "B"; structure and nodes defined implicitly or explicitly
        # This parsing is simplified and might break on complex DOT files.
        # A full DOT parser library might be needed for complex cases.
        lines = dot_source.strip().split('\n')
        all_nodes = set()
        edge_list = []
        node_pattern_simple = re.compile(r'^\s*("?.*?"?)\s*(?:\[.*\])?\s*;?\s*$') # Find node definitions/mentions
        edge_pattern = re.compile(r'^\s*("?.*?"?)\s*->\s*("?.*?"?)\s*(?:\[.*\])?\s*;?\s*$') # Find edges

        for line in lines:
            line = line.strip()
            if not line or line.startswith(('//', '#', 'digraph', 'graph', '{', '}')):
                continue

            edge_match = edge_pattern.match(line)
            if edge_match:
                source_node = edge_match.group(1).strip('"')
                target_node = edge_match.group(2).strip('"')
                all_nodes.add(source_node)
                all_nodes.add(target_node)
                edge_list.append((source_node, target_node))
                continue # Go to next line

            node_match = node_attr_pattern.match(line) # Check for nodes with attributes first
            if node_match:
                 node_name = node_match.group(1).strip('"')
                 all_nodes.add(node_name)
                 continue # Go to next line

            # Check for simple node mentions (without explicit attributes on the line)
            # This helps catch nodes defined implicitly by edges or standalone nodes like "MyTerm";
            node_match_simple = node_pattern_simple.match(line)
            if node_match_simple:
                 node_name = node_match_simple.group(1).strip('"')
                 # Add only if it's not likely an edge definition remnant (basic check)
                 if '->' not in node_name:
                      all_nodes.add(node_name)


        # Create Agraph Nodes
        for node_id in all_nodes:
            color = node_colors.get(node_id, "#ACDBC9") # Default color, use blue if found
            label_color = "#FFFFFF" if color == "blue" else "#000000" # White text on blue nodes?
            nodes.append(Node(id=node_id,
                              label=node_id, # Label is the node name itself
                              # size=25, # Default size
                              color=color,
                              font={'color': label_color}
                              ))

        # Create Agraph Edges
        for src, tgt in edge_list:
            edges.append(Edge(source=src,
                              target=tgt,
                              # type="CURVE_SMOOTH", # Optional edge styling
                              color="#DDDDDD" # Light gray edges
                              ))

        # --- Configure Agraph ---
        # Explore options: https://visjs.github.io/vis-network/docs/network/
        config = Config(width='100%', # Use full width of container
                        height=600, # Set fixed height
                        directed=True,
                        physics=True, # Enable physics simulation for layout initially
                        hierarchical=False, # Turn off hierarchical layout if physics is on
                        # nodeHighlightBehavior=True, # Enable hover effect
                        highlightColor='#F7A7A6',
                        collapsible=False, # Not typically needed for this type
                        node={'labelProperty':'label'}, # Use 'label' field for display
                        # Example layout tuning (optional):
                        # layout={'improvedLayout': True},
                        # physics options (if enabled): https://visjs.github.io/vis-network/docs/network/physics.html
                        physics_config={
                            'barnesHut': {
                                'gravitationalConstant': -8000,
                                'springConstant': 0.04,
                                'springLength': 150 # Increase preferred distance
                            },
                            'minVelocity': 0.75
                        },
                        interaction={'navigationButtons': True, 'keyboard': True, 'tooltipDelay': 300} # Add nav buttons, keyboard support
                       )

        # --- Display Interactive Graph ---
        st.markdown('<div class="agraph-container">', unsafe_allow_html=True) # Apply CSS class
        return_value = agraph(nodes=nodes, edges=edges, config=config)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Add Download Buttons ---
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            # Download DOT code
            st.download_button(
                label="📥 Download DOT Code (.dot)",
                data=st.session_state.dtg_dot_code,
                file_name=f"{st.session_state.dtg_pdf_name}_graph.dot",
                mime="text/vnd.graphviz",
                use_container_width=True
            )
        with col2:
             # Download as PNG using graphviz library rendering
             try:
                  png_bytes = parsed_graph.pipe(format='png')
                  st.download_button(
                      label="🖼️ Download as Image (.png)",
                      data=png_bytes,
                      file_name=f"{st.session_state.dtg_pdf_name}_graph.png",
                      mime="image/png",
                      use_container_width=True
                  )
             except Exception as render_err:
                  st.warning(f"Could not render PNG for download: {render_err}. Ensure Graphviz executable is in PATH.", icon="⚠️")


    except Exception as parse_render_err:
        st.error(f"⚠️ Failed to parse DOT code or render interactive graph. Error: {parse_render_err}")
        st.warning("The DOT code might be invalid, or there was an issue with the visualization library.")
        print(traceback.format_exc())
        # Display DOT code anyway for debugging
        with st.expander("View Generated DOT Code (failed to parse/render)", expanded=True):
             st.code(st.session_state.dtg_dot_code, language='dot')


elif st.session_state.dtg_error:
    # Display error if processing failed (Unchanged)
    st.error(f"❌ Failed to generate graph: {st.session_state.dtg_error}")

elif not st.session_state.dtg_pdf_bytes:
    # Initial state message (Unchanged)
    st.info("⬆️ Upload a document (PDF or TXT) using the sidebar to get started.")

else:
    # Ready state message (Unchanged)
    st.info("⬆️ Ready to generate. Click the 'Generate Relationship Graph' button in the sidebar.")


# --- Footer/Links (Optional) ---
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with Streamlit & Google Gemini")