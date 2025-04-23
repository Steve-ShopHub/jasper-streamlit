# pages/defined_terms_graph.py
# --- COMPLETE FILE vX.Y+2 (Adding Enhanced Console Logging) ---

import streamlit as st
import google.generativeai as genai
from google.generativeai import types
import fitz  # PyMuPDF for PDF text extraction
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

# --- Configuration ---
MODEL_NAME = "gemini-2.5-pro-preview-03-25" # Keeping model as requested
PAGE_TITLE = "Defined Terms Relationship Grapher"
PAGE_ICON = "🔗"
DEFAULT_NODE_COLOR = "#ACDBC9" # Light greenish-teal
HIGHLIGHT_COLOR = "#FFA07A" # Light Salmon for selected node
NEIGHBOR_COLOR = "#ADD8E6" # Light Blue for neighbors

# --- Set Page Config ---
st.set_page_config(layout="wide", page_title=PAGE_TITLE, page_icon=PAGE_ICON)

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
</style>
""", unsafe_allow_html=True)


# --- Helper Function for Text Extraction (Corrected) ---
# (Remains the same)
@st.cache_data(show_spinner="Extracting text from PDF...")
def extract_text_from_pdf(pdf_bytes):
    if not pdf_bytes:
        return None, "No PDF file provided."
    doc = None
    try:
        print(f"[{time.strftime('%H:%M:%S')}] Starting PDF text extraction...")
        start_time = time.time()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text", sort=True)
            if page_text:
                text += page_text
                text += "\n\n--- Page Break --- \n\n"
        end_time = time.time()
        if not text.strip():
            print(f"[{time.strftime('%H:%M:%S')}] PDF text extraction finished in {end_time - start_time:.2f}s. No text found.")
            return None, "Could not extract any text from the PDF."
        print(f"[{time.strftime('%H:%M:%S')}] PDF text extraction finished in {end_time - start_time:.2f}s. Extracted {len(text)} characters.")
        return text, None
    except Exception as e:
        error_msg = f"Error extracting text: {e}"
        print(f"[{time.strftime('%H:%M:%S')}] PDF extraction error: {error_msg}")
        traceback.print_exc()
        return None, error_msg
    finally:
        if doc:
            try: doc.close()
            except Exception as close_err:
                 print(f"[{time.strftime('%H:%M:%S')}] Warning: Error closing PDF document in finally block: {close_err}")
                 pass

# --- Helper Function to Parse AI JSON Response (MODIFIED - added logging) ---
def parse_ai_response(response_text):
    """Parses the AI's JSON response for term names and edges (no definitions)."""
    print(f"[{time.strftime('%H:%M:%S')}] Starting AI response parsing...")
    parsing_start_time = time.time()
    if not response_text or not response_text.strip():
         error_msg = "AI response content is empty."
         print(f"[{time.strftime('%H:%M:%S')}] Parsing failed: {error_msg}")
         return None, error_msg

    try:
        # Attempt to find JSON block
        json_start_match = re.search(r"(\{.*\}|\[.*\])", response_text, re.DOTALL)
        json_text = ""
        if json_start_match:
            json_text = json_start_match.group(0).strip()
            print(f"[{time.strftime('%H:%M:%S')}] Found potential JSON block via regex.")
        else:
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE | re.DOTALL)
            if match:
                json_text = match.group(1).strip()
                print(f"[{time.strftime('%H:%M:%S')}] Found potential JSON block via markdown fence.")
            elif response_text.strip().startswith(("{", "[")):
                json_text = response_text.strip()
                print(f"[{time.strftime('%H:%M:%S')}] Assuming raw response is JSON.")
            else:
                 error_msg = f"Response does not appear to contain a JSON object/array. Raw text snippet: {response_text[:500]}..."
                 print(f"[{time.strftime('%H:%M:%S')}] Parsing failed: {error_msg}")
                 return None, error_msg

        if not json_text:
            error_msg = "Could not extract JSON content from the response."
            print(f"[{time.strftime('%H:%M:%S')}] Parsing failed: {error_msg}")
            return None, error_msg

        print(f"[{time.strftime('%H:%M:%S')}] Attempting json.loads() on extracted text (length: {len(json_text)})...")
        data = json.loads(json_text)
        print(f"[{time.strftime('%H:%M:%S')}] json.loads() successful. Validating structure...")

        # Validate structure
        if not isinstance(data, dict): return None, "Extracted content is not a JSON object."
        if "terms" not in data or "edges" not in data: return None, "Extracted JSON missing required 'terms' or 'edges' keys."
        if not isinstance(data["terms"], list) or not isinstance(data["edges"], list): return None, "'terms' or 'edges' are not lists."

        # Validate terms
        validated_terms = []
        term_names = set()
        for item in data["terms"]:
            if isinstance(item, dict) and "name" in item and isinstance(item["name"], str):
                term_name = item["name"].strip()
                if term_name and term_name not in term_names:
                    validated_terms.append({"name": term_name})
                    term_names.add(term_name)

        # Validate edges
        validated_edges = []
        for edge in data["edges"]:
            if isinstance(edge, dict) and "source" in edge and "target" in edge and isinstance(edge["source"], str) and isinstance(edge["target"], str):
                 source = edge["source"].strip(); target = edge["target"].strip()
                 if source and target and source in term_names and target in term_names:
                    validated_edges.append({"source": source, "target": target})

        if not validated_terms: return None, "Extracted JSON contained no valid terms after validation."

        validated_data = {"terms": validated_terms, "edges": validated_edges}
        parsing_end_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Parsing successful. Found {len(validated_terms)} terms, {len(validated_edges)} edges. Duration: {parsing_end_time - parsing_start_time:.2f}s")
        return validated_data, None

    except json.JSONDecodeError as json_err:
        parsing_end_time = time.time()
        error_pos = json_err.pos
        context_window = 50
        start = max(0, error_pos - context_window)
        end = min(len(json_text), error_pos + context_window) # Use json_text here
        error_snippet = json_text[start:end] if json_text else response_text[start:end] # Fallback to response_text if json_text is empty
        error_snippet_display = repr(error_snippet)
        error_msg = (f"Failed to decode AI JSON response: {json_err}. "
                     f"Error near character {error_pos}. "
                     f"Snippet around error: ...{error_snippet_display}...")
        print(f"[{time.strftime('%H:%M:%S')}] Parsing failed (JSONDecodeError). Duration: {parsing_end_time - parsing_start_time:.2f}s. Error: {error_msg}")
        # traceback.print_exc() # Optional: print full traceback for JSON errors too
        return None, error_msg
    except Exception as e:
        parsing_end_time = time.time()
        error_msg = f"Error parsing AI response structure: {e}"
        print(f"[{time.strftime('%H:%M:%S')}] Parsing failed (Exception). Duration: {parsing_end_time - parsing_start_time:.2f}s. Error: {error_msg}")
        traceback.print_exc()
        return None, error_msg


# --- Initialize Session State (Expanded) ---
# (Remains the same)
def initialize_dtg_state():
    defaults = {
        'dtg_pdf_bytes': None, 'dtg_pdf_name': None, 'dtg_extracted_text': None,
        'dtg_processing': False, 'dtg_error': None, 'dtg_graph_data': None,
        'dtg_nx_graph': None, 'dtg_cycles': None, 'dtg_orphans': None,
        'dtg_filter_term': "", 'dtg_highlight_node': None, 'dtg_layout': 'Physics',
        'dtg_raw_ai_response': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
    if 'api_key' not in st.session_state: st.session_state.api_key = None

initialize_dtg_state()

# --- Graph Analysis Functions ---
# (build_networkx_graph, find_cycles, find_orphans, get_neighbors remain the same - logging added in main processing block)
def build_networkx_graph(graph_data):
    if not graph_data or 'terms' not in graph_data or 'edges' not in graph_data: return None
    G = nx.DiGraph()
    for term_data in graph_data['terms']: G.add_node(term_data['name'])
    for edge_data in graph_data['edges']:
        if G.has_node(edge_data['source']) and G.has_node(edge_data['target']):
             G.add_edge(edge_data['source'], edge_data['target'])
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


# --- Streamlit UI ---

# --- Header ---
# (Remains the same)
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_FILE = "jasper-logo-1.png"
LOGO_PATH = os.path.join(APP_DIR, LOGO_FILE)
header_cols = st.columns([1, 5])
with header_cols[0]:
    if os.path.exists(LOGO_PATH):
        try: st.image(Image.open(LOGO_PATH), width=80)
        except: st.warning("Logo load error.")
with header_cols[1]:
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.caption("Upload document, generate interactive graph of defined terms (names only), analyze relationships.")
st.divider()

# --- Sidebar Controls ---
# (Remains the same)
st.sidebar.title("Controls")
st.sidebar.markdown("---")
api_key_input = st.sidebar.text_input("Google AI Gemini API Key*", type="password", key="api_key_sidebar_dtg", value=st.session_state.get("api_key", ""), help="Your Gemini API key.")
if api_key_input and api_key_input != st.session_state.api_key: st.session_state.api_key = api_key_input
if not st.session_state.api_key: st.sidebar.warning("API Key required.", icon="🔑")

st.sidebar.markdown("### 1. Upload Document")
uploaded_file_obj = st.sidebar.file_uploader("Upload Document (PDF recommended)*", type=["pdf", "txt"], key="dtg_pdf_uploader")

if uploaded_file_obj is not None:
    new_bytes = uploaded_file_obj.getvalue()
    if new_bytes != st.session_state.get('dtg_pdf_bytes') or uploaded_file_obj.name != st.session_state.get('dtg_pdf_name'):
        st.session_state.dtg_pdf_bytes = new_bytes
        st.session_state.dtg_pdf_name = uploaded_file_obj.name
        st.session_state.dtg_extracted_text = None; st.session_state.dtg_error = None
        st.session_state.dtg_processing = False
        st.session_state.dtg_graph_data = None; st.session_state.dtg_nx_graph = None
        st.session_state.dtg_cycles = None; st.session_state.dtg_orphans = None
        st.session_state.dtg_filter_term = ""; st.session_state.dtg_highlight_node = None
        st.session_state.dtg_raw_ai_response = None
        st.toast(f"📄 File '{st.session_state.dtg_pdf_name}' loaded.", icon="✅")
        # Trigger text extraction (will use cache if possible)
        print(f"[{time.strftime('%H:%M:%S')}] New file uploaded: {st.session_state.dtg_pdf_name}. Triggering text extraction.")
        if uploaded_file_obj.type == "application/pdf":
            extracted_text, error_msg = extract_text_from_pdf(st.session_state.dtg_pdf_bytes)
        elif uploaded_file_obj.type == "text/plain":
            try: extracted_text, error_msg = st.session_state.dtg_pdf_bytes.decode('utf-8'), None
            except Exception as e: extracted_text, error_msg = None, f"Failed to read text file: {e}"
        else: extracted_text, error_msg = None, f"Unsupported file type: {uploaded_file_obj.type}"

        if error_msg:
            st.session_state.dtg_error = error_msg
            st.session_state.dtg_extracted_text = None
        else:
            st.session_state.dtg_extracted_text = extracted_text
            st.session_state.dtg_error = None
            st.toast("Text extracted.", icon="📝")
        st.rerun()

if st.session_state.dtg_error and not st.session_state.dtg_processing and not st.session_state.dtg_graph_data:
     st.error(st.session_state.dtg_error)

st.sidebar.markdown("### 2. Generate & Analyze")
can_generate = (st.session_state.api_key and st.session_state.dtg_pdf_bytes and st.session_state.dtg_extracted_text and not st.session_state.dtg_processing)
generate_button_tooltip = ""
if st.session_state.dtg_processing: generate_button_tooltip = "Processing..."
elif not st.session_state.api_key: generate_button_tooltip = "Enter API Key"
elif not st.session_state.dtg_pdf_bytes: generate_button_tooltip = "Upload a document"
elif not st.session_state.dtg_extracted_text: generate_button_tooltip = "Could not extract text"
else: generate_button_tooltip = "Generate graph (names only)"

if st.sidebar.button("✨ Generate & Analyze Graph", key="dtg_generate", disabled=not can_generate, help=generate_button_tooltip, use_container_width=True, type="primary"):
    print(f"[{time.strftime('%H:%M:%S')}] 'Generate & Analyze Graph' button clicked.")
    st.session_state.dtg_processing = True
    st.session_state.dtg_graph_data = None; st.session_state.dtg_nx_graph = None
    st.session_state.dtg_cycles = None; st.session_state.dtg_orphans = None
    st.session_state.dtg_error = None; st.session_state.dtg_filter_term = ""
    st.session_state.dtg_highlight_node = None; st.session_state.dtg_raw_ai_response = None
    st.rerun()

if st.session_state.dtg_graph_data:
    # (Graph Interaction Controls remain the same)
    st.sidebar.markdown("---"); st.sidebar.markdown("### 3. Graph Interaction")
    st.session_state.dtg_filter_term = st.sidebar.text_input("Filter Nodes (by name)", value=st.session_state.dtg_filter_term, placeholder="Type term...", key="dtg_filter_input").strip()
    available_nodes = ["--- Select Node ---"]; current_highlight_index = 0
    if st.session_state.dtg_nx_graph:
         nodes_to_consider = list(st.session_state.dtg_nx_graph.nodes())
         if st.session_state.dtg_filter_term:
              try: filter_regex = re.compile(st.session_state.dtg_filter_term, re.IGNORECASE); nodes_to_consider = [n for n in nodes_to_consider if filter_regex.search(n)]
              except re.error: st.sidebar.warning("Invalid regex", icon="⚠️"); nodes_to_consider = []
         available_nodes.extend(sorted(nodes_to_consider))
         if st.session_state.dtg_highlight_node and st.session_state.dtg_highlight_node in available_nodes:
              current_highlight_index = available_nodes.index(st.session_state.dtg_highlight_node)
         else: st.session_state.dtg_highlight_node = None

    highlight_key = f"highlight_select_{st.session_state.dtg_filter_term}"
    st.session_state.dtg_highlight_node = st.sidebar.selectbox("Highlight Node & Neighbors", options=available_nodes, index=current_highlight_index, key=highlight_key)
    if st.session_state.dtg_highlight_node == "--- Select Node ---": st.session_state.dtg_highlight_node = None
    st.session_state.dtg_layout = st.sidebar.radio("Graph Layout", options=['Physics', 'Hierarchical'], index=0 if st.session_state.dtg_layout == 'Physics' else 1, key="dtg_layout_radio")


# --- Main Area ---
if st.session_state.dtg_processing:
    status_placeholder = st.empty()
    full_response_text = "" # Initialize outside the try block
    process_start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] ===== Starting Generation Process =====")
    with st.spinner(f"⚙️ Analyzing '{st.session_state.dtg_pdf_name}'..."):
        status_placeholder.info("🧠 Asking Gemini to extract term names and relationships...")
        try:
            print(f"[{time.strftime('%H:%M:%S')}] Configuring GenAI client...")
            genai.configure(api_key=st.session_state.api_key)
            document_text = st.session_state.dtg_extracted_text
            if not document_text:
                st.session_state.dtg_error = "Cannot proceed without extracted text."
                print(f"[{time.strftime('%H:%M:%S')}] Error: Extracted text is missing.")
                # Need to stop processing here
                st.session_state.dtg_processing = False
                st.rerun() # Rerun to show error
                st.stop()  # Stop script execution

            # Prompt (Remains the same)
            prompt_instructions = f"""
Your task is to analyze ONLY the 'Definitions' section (typically Section 1 or similar) of the provided legal document text below. The goal is to identify all formally defined terms and map the interdependencies *only* between these terms based on their definitions.

**Output Format:** Produce a single JSON object with two keys: "terms" and "edges".
1.  `"terms"`: A list of JSON objects. Each object must have ONLY:
    *   `"name"`: The exact defined term (string), properly handling quotes if they were part of the definition marker. **DO NOT include the definition text itself.**
2.  `"edges"`: A list of JSON objects. Each object represents a directed link and must have:
    *   `"source"`: The name of the defined term whose definition uses another term (string, must match a name in the "terms" list).
    *   `"target"`: The name of the defined term used within the source term's definition (string, must match a name in the "terms" list).

**Instructions for Extraction:**

*   **Focus:** Strictly analyze the section containing explicit definitions (e.g., terms in quotes followed by "means..."). Ignore other sections.
*   **Identify Defined Terms:** Only include terms that are formally defined within this 'Definitions' section (e.g., `"Term Name" means...`). Include all such terms found in the "terms" list, providing only their `"name"`.
*   **Omit Definitions:** CRITICAL - **Do NOT** include the full definition text in the output JSON. Only provide the term's name.
*   **Identify Edges (Links):** Even though you are not outputting the definition text, you MUST read the definition for each formally defined term ("Term A"). If that definition text explicitly uses another term ("Term B") that is *also* formally defined in the same section, create an edge object from "Term A" (source) to "Term B" (target).
*   **Exclusions (CRITICAL): Do NOT include data in the "terms" or "edges" lists relating to:** Clause numbers, Section numbers, Schedule numbers, specific dates, amounts, percentages, references to external laws/acts/directives (unless the act itself is the primary term being defined), party names (unless explicitly defined as a term), or acronyms (unless formally defined). Only include formally defined terms (by name) and their direct definition-based links to other formally defined terms.
*   **Completeness:** Ensure all formally defined terms from the relevant section are included by name in the "terms" list. Ensure all valid definition-based links between these terms are included in the "edges" list.

**Document Text (Definitions Section Focus):**
--- Start Document Text ---
{document_text}
--- End Document Text ---

**Final Output (Valid JSON Object Only - NO DEFINITION TEXT):**
"""
            model = genai.GenerativeModel(MODEL_NAME)
            generation_config = types.GenerationConfig(temperature=0.1, top_k=0.05)
            safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]

            status_placeholder.info("📞 Calling Gemini API (streaming response)...")
            print(f"[{time.strftime('%H:%M:%S')}] Preparing API call. Prompt length (approx): {len(prompt_instructions)} chars.")
            print(f"[{time.strftime('%H:%M:%S')}] Sending request to Gemini API...")
            api_call_start_time = time.time()
            response = model.generate_content(
                contents=prompt_instructions,
                generation_config=generation_config,
                safety_settings=safety_settings,
                request_options={'timeout': 600}, # 10 minute timeout for API call itself
                stream=True
            )

            status_placeholder.info("⏳ Receiving streamed response from Gemini...")
            print(f"[{time.strftime('%H:%M:%S')}] API call initiated. Waiting for stream...")
            chunk_count = 0
            stream_start_time = None # Track when first chunk arrives

            for chunk in response:
                if stream_start_time is None:
                    stream_start_time = time.time()
                    print(f"[{time.strftime('%H:%M:%S')}] First chunk received! Time to first chunk: {stream_start_time - api_call_start_time:.2f}s")

                try:
                    if chunk.text:
                        full_response_text += chunk.text
                        chunk_count += 1
                        # Log every N chunks or every X seconds to avoid flooding console
                        if chunk_count % 20 == 0:
                             current_time = time.time()
                             print(f"[{time.strftime('%H:%M:%S')}] Received chunk {chunk_count}. Total length so far: {len(full_response_text)}. Stream duration: {current_time - stream_start_time:.2f}s")
                             status_placeholder.info(f"⏳ Receiving streamed response from Gemini... (received {chunk_count} chunks)")
                    # else:
                    #     print(f"[{time.strftime('%H:%M:%S')}] Received chunk {chunk_count+1} with no text content (might be metadata/finish reason).")

                except ValueError as ve:
                    print(f"[{time.strftime('%H:%M:%S')}] Warning: Skipping a non-text chunk or potential error in stream: {ve}")
                    print(f"Stream chunk value error details: {chunk}")
                    continue

            api_call_end_time = time.time()
            st.session_state.dtg_raw_ai_response = full_response_text # Store the full raw response
            print(f"[{time.strftime('%H:%M:%S')}] Stream finished. Received {chunk_count} text chunks.")
            print(f"[{time.strftime('%H:%M:%S')}] Total raw response length: {len(full_response_text)} chars.")
            if stream_start_time: print(f"[{time.strftime('%H:%M:%S')}] Full stream duration: {api_call_end_time - stream_start_time:.2f} seconds.")
            print(f"[{time.strftime('%H:%M:%S')}] Total API call duration (including stream wait): {api_call_end_time - api_call_start_time:.2f} seconds.")

            status_placeholder.info("📄 Processing Gemini's full response...")
            if not full_response_text.strip():
                 st.session_state.dtg_error = "AI returned an empty response."
                 print(f"[{time.strftime('%H:%M:%S')}] Error: AI response was empty.")
                 graph_data = None
            else:
                 # Parsing (logging is inside the function)
                 graph_data, error_msg = parse_ai_response(full_response_text)
                 if error_msg:
                     st.session_state.dtg_error = error_msg
                     # Error already logged inside parse_ai_response
                 else:
                    st.session_state.dtg_graph_data = graph_data; st.session_state.dtg_error = None; st.toast("Term names & links extracted!", icon="📊")
                    status_placeholder.info("⚙️ Analyzing graph structure...")

                    # Graph Building
                    print(f"[{time.strftime('%H:%M:%S')}] Building NetworkX graph...")
                    graph_build_start_time = time.time()
                    st.session_state.dtg_nx_graph = build_networkx_graph(graph_data)
                    graph_build_end_time = time.time()
                    if st.session_state.dtg_nx_graph:
                        print(f"[{time.strftime('%H:%M:%S')}] Graph built. Nodes: {len(st.session_state.dtg_nx_graph.nodes())}, Edges: {len(st.session_state.dtg_nx_graph.edges())}. Duration: {graph_build_end_time - graph_build_start_time:.2f}s")

                        # Graph Analysis
                        print(f"[{time.strftime('%H:%M:%S')}] Finding cycles...")
                        analysis_start_time = time.time()
                        st.session_state.dtg_cycles = find_cycles(st.session_state.dtg_nx_graph)
                        print(f"[{time.strftime('%H:%M:%S')}] Found {len(st.session_state.dtg_cycles) if st.session_state.dtg_cycles is not None else 'N/A'} cycles.")
                        print(f"[{time.strftime('%H:%M:%S')}] Finding orphans...")
                        st.session_state.dtg_orphans = find_orphans(st.session_state.dtg_nx_graph)
                        analysis_end_time = time.time()
                        print(f"[{time.strftime('%H:%M:%S')}] Found {len(st.session_state.dtg_orphans) if st.session_state.dtg_orphans is not None else 'N/A'} orphans.")
                        print(f"[{time.strftime('%H:%M:%S')}] Graph analysis duration: {analysis_end_time - analysis_start_time:.2f}s")
                        st.toast("Graph analysis complete.", icon="🔬")
                    else:
                        st.warning("Could not build internal graph for analysis.")
                        print(f"[{time.strftime('%H:%M:%S')}] Warning: Failed to build NetworkX graph. Duration: {graph_build_end_time - graph_build_start_time:.2f}s")

        # --- Exception Handling ---
        except types.StopCandidateException as sce:
            st.session_state.dtg_error = f"Generation Stopped Unexpectedly: {sce}. Response might be incomplete or blocked."
            print(f"[{time.strftime('%H:%M:%S')}] StopCandidateException: {sce}")
            traceback.print_exc()
        except google.api_core.exceptions.GoogleAPIError as api_err:
            st.session_state.dtg_error = f"Google API Error: {api_err}. Check key/quota/permissions/network."
            print(f"[{time.strftime('%H:%M:%S')}] GoogleAPIError: {api_err}")
            traceback.print_exc()
        except Exception as e:
            st.session_state.dtg_error = f"Processing Error: {e}"
            print(f"[{time.strftime('%H:%M:%S')}] General Exception during processing:")
            traceback.print_exc()
        finally:
            # Ensure raw response is saved even if error happens late
            if full_response_text and not st.session_state.dtg_raw_ai_response:
                 st.session_state.dtg_raw_ai_response = full_response_text
            st.session_state.dtg_processing = False
            status_placeholder.empty()
            process_end_time = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] ===== Generation Process Ended =====")
            print(f"[{time.strftime('%H:%M:%S')}] Total processing duration: {process_end_time - process_start_time:.2f} seconds.")
            print(f"[{time.strftime('%H:%M:%S')}] Triggering rerun.")
            st.rerun()

elif st.session_state.dtg_graph_data:
    # --- Display Results ---
    # (Graph display, analysis display, export remain the same)
    st.subheader(f"📊 Interactive Graph & Analysis for '{st.session_state.dtg_pdf_name}'")
    G = st.session_state.dtg_nx_graph
    filter_term = st.session_state.dtg_filter_term; highlight_node = st.session_state.dtg_highlight_node

    # Filter nodes/edges
    nodes_to_display_names = set(G.nodes()) if G else set()
    if filter_term:
        try: filter_regex = re.compile(filter_term, re.IGNORECASE); nodes_to_display_names = {n for n in G.nodes() if filter_regex.search(n)}
        except re.error: st.warning("Invalid filter regex.", icon="⚠️"); nodes_to_display_names = set(G.nodes()) if G else set()

    highlight_neighbors_predecessors, highlight_neighbors_successors = set(), set()
    if highlight_node and G: highlight_neighbors_predecessors, highlight_neighbors_successors = get_neighbors(G, highlight_node)

    # Prepare Agraph Nodes & Edges
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
                 agraph_edges_tuples.append((u, v))
                 agraph_edges.append(Edge(source=u, target=v, color="#CCCCCC"))

    # Configure Agraph
    is_physics = st.session_state.dtg_layout == 'Physics'
    config = Config(width='100%', height=700, directed=True, physics=is_physics, hierarchical=not is_physics, highlightColor=HIGHLIGHT_COLOR, collapsible=False, node={'labelProperty':'label', 'size': 15},
        physics_config={'barnesHut': {'gravitationalConstant': -10000, 'centralGravity': 0.1, 'springLength': 180, 'springConstant': 0.05, 'damping': 0.09, 'avoidOverlap': 0.1}, 'minVelocity': 0.75} if is_physics else {},
        layout={'hierarchical': {'enabled': (not is_physics), 'sortMethod': 'directed', 'levelSeparation': 150, 'nodeSpacing': 120}} if not is_physics else {},
        interaction={'navigationButtons': True, 'keyboard': True, 'tooltipDelay': 300, 'hover': True})

    # Display Area
    graph_col, info_col = st.columns([3, 1])
    with graph_col:
        st.caption("Graph View: Click/drag to pan, scroll/pinch to zoom. Select node in sidebar to highlight.")
        if not agraph_nodes and filter_term: st.warning(f"No nodes match filter: '{filter_term}'")
        elif not agraph_nodes: st.warning("No graph data to display.")
        elif agraph_nodes and agraph_edges is not None:
             try: agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)
             except Exception as e: st.error(f"Graph render error: {e}"); print(traceback.format_exc())
        elif agraph_nodes:
             try: agraph(nodes=agraph_nodes, edges=[], config=config)
             except Exception as e: st.error(f"Graph render error (nodes only): {e}"); print(traceback.format_exc())
    with info_col:
        st.subheader("Details & Analysis")
        st.markdown("**Selected Term:**")
        if highlight_node: st.info(f"`{highlight_node}`")
        else: st.info("_Select node in sidebar_")
        st.caption("_Full definition text is not extracted._")
        st.markdown("---"); st.markdown("**Graph Analysis:**")
        if st.session_state.dtg_cycles is not None:
             if st.session_state.dtg_cycles:
                  with st.expander(f"🚨 {len(st.session_state.dtg_cycles)} Circular Definitions", expanded=False):
                       for i, c in enumerate(st.session_state.dtg_cycles): st.markdown(f"- Cycle {i+1}: `{' → '.join(c)} → {c[0]}`")
             else: st.caption("✅ No circular definitions detected.")
        if st.session_state.dtg_orphans is not None:
             if st.session_state.dtg_orphans:
                  with st.expander(f"⚠️ {len(st.session_state.dtg_orphans)} Orphan Terms", expanded=False):
                       st.markdown(f"`{', '.join(st.session_state.dtg_orphans)}`")
                       st.caption("_Defined but not linked._")
             else: st.caption("✅ All defined terms linked.")
    st.divider()

    # Generate DOT Code
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

    # Download Buttons
    st.subheader("Export Graph"); export_cols = st.columns(4); safe_filename_base = re.sub(r'[^\w\-]+', '_', st.session_state.dtg_pdf_name or "graph")
    with export_cols[0]: export_cols[0].download_button("📥 DOT (.dot)", generated_dot_code, f"{safe_filename_base}_graph.dot", "text/vnd.graphviz", use_container_width=True)
    with export_cols[1]:
         try: png_bytes = graphviz.Source(generated_dot_code).pipe(format='png'); export_cols[1].download_button("🖼️ PNG (.png)", png_bytes, f"{safe_filename_base}_graph.png", "image/png", use_container_width=True)
         except Exception as e: export_cols[1].warning(f"PNG ERR: {e}", icon="⚠️")
    with export_cols[2]:
         try: svg_bytes = graphviz.Source(generated_dot_code).pipe(format='svg'); export_cols[2].download_button("📐 SVG (.svg)", svg_bytes, f"{safe_filename_base}_graph.svg", "image/svg+xml", use_container_width=True)
         except Exception as e: export_cols[2].warning(f"SVG ERR: {e}", icon="⚠️")
    with export_cols[3]:
        if G:
            try:
                 df_deps = pd.DataFrame([{"Source Term": u, "Depends On (Target Term)": v} for u, v in agraph_edges_tuples])
                 csv_output = df_deps.to_csv(index=False).encode('utf-8')
                 export_cols[3].download_button("📋 Deps (.csv)", csv_output, f"{safe_filename_base}_dependencies.csv", "text/csv", use_container_width=True)
            except Exception as e: export_cols[3].warning(f"CSV ERR: {e}", icon="⚠️")
    with st.expander("View Generated DOT Code (for current view)"): st.code(generated_dot_code, language='dot')

elif st.session_state.dtg_error:
    # (Error display with raw response remains the same)
    st.error(f"❌ Failed: {st.session_state.dtg_error}")
    if st.session_state.dtg_raw_ai_response:
        with st.expander("View Full Raw AI Response (for debugging)", expanded=False):
             st.text_area("Raw Response", st.session_state.dtg_raw_ai_response, height=400, disabled=True, label_visibility="collapsed")
    # else: st.info("No raw AI response was captured.") # Redundant if error always shown

elif not st.session_state.dtg_pdf_bytes: st.info("⬆️ Upload a document (PDF or TXT) using the sidebar to get started.")
else: st.info("⬆️ Ready to generate. Click the 'Generate & Analyze Graph' button in the sidebar.")


# Footer
st.sidebar.markdown("---"); st.sidebar.markdown("Developed with Streamlit & Google Gemini")