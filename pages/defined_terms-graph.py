# pages/defined_terms_graph.py
# --- COMPLETE FILE vX.Y (Integrating all features and fixes) ---

import streamlit as st
import google.generativeai as genai
from google.generativeai import types
import fitz  # PyMuPDF for PDF text extraction
import re
import os
import traceback
import time
import io # For download button
import json
import graphviz # Python graphviz library for parsing DOT and rendering
import networkx as nx # For graph analysis (cycles, orphans, neighbors)
import pandas as pd # For CSV export
from streamlit_agraph import agraph, Node, Edge, Config
from PIL import Image # For Logo import
from collections import defaultdict

# --- Configuration ---
MODEL_NAME = "gemini-1.5-pro-latest" # Or your preferred model
PAGE_TITLE = "Defined Terms Relationship Grapher"
PAGE_ICON = "🔗"
DEFAULT_NODE_COLOR = "#ACDBC9" # Light greenish-teal
HIGHLIGHT_COLOR = "#FFA07A" # Light Salmon for selected node
NEIGHBOR_COLOR = "#ADD8E6" # Light Blue for neighbors

# --- Set Page Config ---
st.set_page_config(layout="wide", page_title=PAGE_TITLE, page_icon=PAGE_ICON)

# --- Optional CSS ---
st.markdown("""
<style>
    /* Ensure Streamlit containers don't add excessive padding */
     div[data-testid="stVerticalBlock"] > div[style*="gap: 1rem;"] {
        gap: 0.5rem !important;
     }
    /* Style for the definition display area */
    .definition-box {
        background-color: #f0f2f6;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
        height: 150px; /* Adjust height as needed */
        overflow-y: auto; /* Add scroll if definition is long */
        font-size: 0.9em;
        white-space: pre-wrap; /* Ensure line breaks in definitions are respected */
        word-wrap: break-word; /* Break long words */
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Function for Text Extraction (Corrected) ---
@st.cache_data(show_spinner="Extracting text from PDF...")
def extract_text_from_pdf(pdf_bytes):
    if not pdf_bytes:
        return None, "No PDF file provided."
    doc = None # Initialize doc outside try block for finally clause
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        # Correctly indented loop inside the try block
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Use get_text("text", sort=True) for better reading order
            page_text = page.get_text("text", sort=True)
            if page_text: # Avoid adding breaks if page had no text
                text += page_text
                text += "\n\n--- Page Break --- \n\n" # Add clearer page breaks

        if not text.strip():
            # No need to close doc here, finally will handle it
            return None, "Could not extract any text from the PDF."

        # Return successfully extracted text
        return text, None
    except Exception as e:
        # Correctly placed except block
        error_msg = f"Error extracting text: {e}"
        print(traceback.format_exc())
        return None, error_msg
    finally:
        # Ensure the document is closed even if errors occur
        if doc: # Check if doc was successfully opened
            try:
                doc.close()
            except Exception as close_err:
                 # Log or ignore errors during closing
                 print(f"Warning: Error closing PDF document in finally block: {close_err}")
                 pass

# --- Helper Function to Parse AI JSON Response ---
def parse_ai_response(response_text):
    """Parses the AI's JSON response for terms, definitions, and edges."""
    try:
        # Attempt to handle potential markdown fences
        match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE | re.DOTALL)
        if match:
            json_text = match.group(1).strip()
        else:
            # Check if the entire response looks like a JSON object/array before stripping
            if response_text.strip().startswith(("{", "[")):
                 json_text = response_text.strip()
            else: # If it's plain text or something else unexpected
                 return None, f"Response does not appear to be JSON. Raw text snippet: {response_text[:500]}..."


        if not json_text:
            return None, "AI response content is empty after stripping."

        data = json.loads(json_text)

        # Validate basic structure
        if not isinstance(data, dict):
            return None, "AI response is not a JSON object."
        if "terms" not in data or "edges" not in data:
            return None, "AI response missing required 'terms' or 'edges' keys."
        if not isinstance(data["terms"], list) or not isinstance(data["edges"], list):
            return None, "'terms' or 'edges' are not lists in the AI response."

        # Further validation
        validated_terms = []
        term_names = set()
        for item in data["terms"]:
            if isinstance(item, dict) and "name" in item and "definition" in item and isinstance(item["name"], str) and isinstance(item["definition"], str):
                term_name = item["name"].strip() # Trim whitespace
                if term_name and term_name not in term_names: # Ensure name is not empty
                    validated_terms.append({"name": term_name, "definition": item["definition"]})
                    term_names.add(term_name)
            # else: st.warning(f"Skipping malformed term item: {item}") # Optional warning

        validated_edges = []
        for edge in data["edges"]:
            if isinstance(edge, dict) and "source" in edge and "target" in edge and isinstance(edge["source"], str) and isinstance(edge["target"], str):
                 source = edge["source"].strip()
                 target = edge["target"].strip()
                 # Ensure source and target are actual defined terms found and not empty
                 if source and target and source in term_names and target in term_names:
                    validated_edges.append({"source": source, "target": target})
            # else: st.warning(f"Skipping malformed edge item: {edge}") # Optional warning

        if not validated_terms:
             return None, "AI response contained no valid terms after validation."

        validated_data = {
            "terms": validated_terms,
            "edges": validated_edges
        }
        return validated_data, None

    except json.JSONDecodeError as json_err:
        return None, f"Failed to decode AI JSON response: {json_err}. Raw text snippet: {response_text[:500]}..."
    except Exception as e:
        return None, f"Error parsing AI response structure: {e}"


# --- Initialize Session State (Expanded) ---
def initialize_dtg_state():
    defaults = {
        'dtg_pdf_bytes': None,
        'dtg_pdf_name': None,
        'dtg_extracted_text': None,
        'dtg_processing': False,
        'dtg_error': None,
        'dtg_graph_data': None, # Will store {"terms": [...], "edges": [...]}
        'dtg_nx_graph': None,   # Will store the networkx graph object
        'dtg_cycles': None,     # List of cycles found
        'dtg_orphans': None,    # List of orphan nodes
        'dtg_filter_term': "",  # Text input for filtering
        'dtg_highlight_node': None, # Node selected for highlight/definition
        'dtg_layout': 'Physics', # Default layout
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            # Deep copy for mutable defaults if necessary, though these are mostly primitives or None
            st.session_state[key] = value
    # Ensure API key exists
    if 'api_key' not in st.session_state:
         st.session_state.api_key = None

initialize_dtg_state()

# --- Graph Analysis Functions ---
def build_networkx_graph(graph_data):
    """Builds a NetworkX DiGraph from parsed AI data."""
    if not graph_data or 'terms' not in graph_data or 'edges' not in graph_data:
        return None
    G = nx.DiGraph()
    # Add nodes first
    for term_data in graph_data['terms']:
        G.add_node(term_data['name'], definition=term_data['definition'])
    # Add edges, ensuring nodes exist (though they should from previous step)
    for edge_data in graph_data['edges']:
        if G.has_node(edge_data['source']) and G.has_node(edge_data['target']):
             G.add_edge(edge_data['source'], edge_data['target'])
    return G

def find_cycles(G):
    """Finds simple cycles in a NetworkX DiGraph."""
    if G is None: return None
    try:
        # simple_cycles detects elementary cycles
        return list(nx.simple_cycles(G))
    except Exception as e:
        print(f"Error finding cycles: {e}")
        return None # Return None or empty list on error

def find_orphans(G):
    """Finds nodes with in-degree and out-degree of 0."""
    if G is None: return None
    orphans_directed = [
        node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 0
    ]
    return orphans_directed

def get_neighbors(G, node_id):
    """Gets predecessors (pointing to node) and successors (node points to)."""
    if G is None or node_id not in G:
        return set(), set()
    # NetworkX provides these methods directly
    predecessors = set(G.predecessors(node_id))
    successors = set(G.successors(node_id))
    return predecessors, successors

# --- Streamlit UI ---

# --- Header ---
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_FILE = "jasper-logo-1.png" # Make sure this filename matches your logo file
LOGO_PATH = os.path.join(APP_DIR, LOGO_FILE)
header_cols = st.columns([1, 5])
with header_cols[0]:
    if os.path.exists(LOGO_PATH):
        try:
            logo = Image.open(LOGO_PATH)
            st.image(logo, width=80, caption=None, output_format='PNG') # Removed alt
        except FileNotFoundError:
             st.warning(f"Logo file not found: {LOGO_PATH}") # More specific error
        except Exception as img_err:
            st.warning(f"Could not load logo: {img_err}")
with header_cols[1]:
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.caption("Upload a document, generate an interactive graph of defined terms, and analyze relationships.")
st.divider()

# --- Sidebar Controls ---
st.sidebar.title("Controls")
st.sidebar.markdown("---")
# API Key Input
api_key_input = st.sidebar.text_input(
    "Google AI Gemini API Key*", type="password", key="api_key_sidebar_dtg",
    value=st.session_state.get("api_key", ""), help="Your Gemini API key."
)
if api_key_input and api_key_input != st.session_state.api_key:
    st.session_state.api_key = api_key_input
if not st.session_state.api_key:
    st.sidebar.warning("API Key required.", icon="🔑")
# File Uploader
st.sidebar.markdown("### 1. Upload Document")
uploaded_file_obj = st.sidebar.file_uploader(
    "Upload Document (PDF recommended)*", type=["pdf", "txt"], key="dtg_pdf_uploader"
)
# Process upload
if uploaded_file_obj is not None:
    # Check if it's a new file based on bytes AND name to handle re-uploads
    new_bytes = uploaded_file_obj.getvalue()
    if new_bytes != st.session_state.get('dtg_pdf_bytes') or uploaded_file_obj.name != st.session_state.get('dtg_pdf_name'):
        st.session_state.dtg_pdf_bytes = new_bytes
        st.session_state.dtg_pdf_name = uploaded_file_obj.name
        st.session_state.dtg_extracted_text = None; st.session_state.dtg_error = None
        st.session_state.dtg_processing = False
        # Reset results and analysis state
        st.session_state.dtg_graph_data = None; st.session_state.dtg_nx_graph = None
        st.session_state.dtg_cycles = None; st.session_state.dtg_orphans = None
        st.session_state.dtg_filter_term = ""; st.session_state.dtg_highlight_node = None
        st.toast(f"📄 File '{st.session_state.dtg_pdf_name}' loaded.", icon="✅")
        # Extract text
        if uploaded_file_obj.type == "application/pdf":
            extracted_text, error_msg = extract_text_from_pdf(st.session_state.dtg_pdf_bytes)
        elif uploaded_file_obj.type == "text/plain":
            try:
                extracted_text, error_msg = st.session_state.dtg_pdf_bytes.decode('utf-8'), None
            except Exception as e:
                extracted_text, error_msg = None, f"Failed to read text file: {e}"
        else:
            extracted_text, error_msg = None, f"Unsupported file type: {uploaded_file_obj.type}"

        if error_msg:
            st.session_state.dtg_error = error_msg # Store error to display in main area
            st.session_state.dtg_extracted_text = None # Ensure no stale text
        else:
            st.session_state.dtg_extracted_text = extracted_text
            st.session_state.dtg_error = None # Clear any previous error
            st.toast("Text extracted.", icon="📝")
        st.rerun() # Rerun necessary after state update for upload

if st.session_state.dtg_error and not st.session_state.dtg_processing and not st.session_state.dtg_graph_data:
     st.error(st.session_state.dtg_error) # Show extraction error if it happened before processing

# Generation Button
st.sidebar.markdown("### 2. Generate & Analyze")
can_generate = (st.session_state.api_key and
                st.session_state.dtg_pdf_bytes and
                st.session_state.dtg_extracted_text and
                not st.session_state.dtg_processing)
generate_button_tooltip = ""
if st.session_state.dtg_processing: generate_button_tooltip = "Processing..."
elif not st.session_state.api_key: generate_button_tooltip = "Enter API Key"
elif not st.session_state.dtg_pdf_bytes: generate_button_tooltip = "Upload a document"
elif not st.session_state.dtg_extracted_text: generate_button_tooltip = "Could not extract text from document"
else: generate_button_tooltip = "Generate graph and analyze term relationships using Gemini"
if st.sidebar.button("✨ Generate & Analyze Graph", key="dtg_generate", disabled=not can_generate, help=generate_button_tooltip, use_container_width=True, type="primary"):
    st.session_state.dtg_processing = True
    st.session_state.dtg_graph_data = None; st.session_state.dtg_nx_graph = None
    st.session_state.dtg_cycles = None; st.session_state.dtg_orphans = None
    st.session_state.dtg_error = None; st.session_state.dtg_filter_term = "" # Reset filter
    st.session_state.dtg_highlight_node = None # Reset highlight
    st.rerun()

# Graph Interaction Controls
if st.session_state.dtg_graph_data:
    st.sidebar.markdown("---"); st.sidebar.markdown("### 3. Graph Interaction")
    st.session_state.dtg_filter_term = st.sidebar.text_input("Filter Nodes (by name)", value=st.session_state.dtg_filter_term, placeholder="Type term to filter...", key="dtg_filter_input").strip()
    available_nodes = ["--- Select Node ---"]; current_highlight_index = 0
    if st.session_state.dtg_nx_graph:
         nodes_to_consider = list(st.session_state.dtg_nx_graph.nodes())
         if st.session_state.dtg_filter_term:
              try: filter_regex = re.compile(st.session_state.dtg_filter_term, re.IGNORECASE); nodes_to_consider = [n for n in nodes_to_consider if filter_regex.search(n)]
              except re.error: st.sidebar.warning("Invalid filter regex.", icon="⚠️"); nodes_to_consider = []
         available_nodes.extend(sorted(nodes_to_consider))
         # Ensure highlighted node exists in potentially filtered list
         if st.session_state.dtg_highlight_node and st.session_state.dtg_highlight_node in available_nodes:
              current_highlight_index = available_nodes.index(st.session_state.dtg_highlight_node)
         else:
              st.session_state.dtg_highlight_node = None # Reset highlight if filter removes it

    highlight_key = f"highlight_select_{st.session_state.dtg_filter_term}" # Key changes with filter to reset selection
    st.session_state.dtg_highlight_node = st.sidebar.selectbox(
        "Highlight Node & Neighbors", options=available_nodes, index=current_highlight_index,
        key=highlight_key, help="Select node to highlight it and dependencies."
    )
    if st.session_state.dtg_highlight_node == "--- Select Node ---": st.session_state.dtg_highlight_node = None

    st.session_state.dtg_layout = st.sidebar.radio("Graph Layout", options=['Physics', 'Hierarchical'], index=0 if st.session_state.dtg_layout == 'Physics' else 1, key="dtg_layout_radio", help="Choose layout algorithm.")

# --- Main Area ---
if st.session_state.dtg_processing:
    status_placeholder = st.empty()
    with st.spinner(f"⚙️ Analyzing '{st.session_state.dtg_pdf_name}'..."):
        status_placeholder.info("🧠 Asking Gemini to extract terms, definitions, and relationships...")
        try:
            genai.configure(api_key=st.session_state.api_key); document_text = st.session_state.dtg_extracted_text
            # Revised prompt requesting JSON
            prompt_instructions = f"""
Your task is to analyze ONLY the 'Definitions' section (typically Section 1 or similar) of the provided legal document text below. The goal is to create a structured representation of the interdependencies *only* between the terms explicitly defined within that section.

**Output Format:** Produce a single JSON object with two keys: "terms" and "edges".
1.  `"terms"`: A list of JSON objects. Each object must have:
    *   `"name"`: The exact defined term (string), properly handling quotes if they were part of the definition marker.
    *   `"definition"`: The full definition text for that term (string).
2.  `"edges"`: A list of JSON objects. Each object represents a directed link and must have:
    *   `"source"`: The name of the defined term whose definition uses another term (string, must match a name in the "terms" list).
    *   `"target"`: The name of the defined term used within the source term's definition (string, must match a name in the "terms" list).

**Instructions for Extraction:**

*   **Focus:** Strictly analyze the section containing explicit definitions (e.g., terms in quotes followed by "means..."). Ignore other sections.
*   **Identify Defined Terms:** Only include terms that are formally defined within this 'Definitions' section (e.g., `"Term Name" means...`). Include all such terms found in the "terms" list.
*   **Extract Definitions:** Capture the complete definition text associated with each defined term.
*   **Identify Edges (Links):** Create an edge object from "Term A" (source) to "Term B" (target) ONLY IF the definition text provided *for* "Term A" explicitly uses "Term B", AND "Term B" is *also* one of the formally defined terms identified from the same 'Definitions' section.
*   **Exclusions (CRITICAL): Do NOT include data in the "terms" or "edges" lists relating to:** Clause numbers, Section numbers, Schedule numbers, specific dates, amounts, percentages, references to external laws/acts/directives (unless the act itself is the primary term being defined), party names (unless explicitly defined as a term), or acronyms (unless formally defined). Only include formally defined terms and their direct definition-based links to other formally defined terms.
*   **Completeness:** Ensure all formally defined terms from the relevant section are included in the "terms" list. Ensure all valid definition-based links between these terms are included in the "edges" list.

**Document Text (Definitions Section Focus):**
--- Start Document Text ---
{document_text}
--- End Document Text ---

**Final Output (Valid JSON Object Only):**
"""
            model = genai.GenerativeModel(MODEL_NAME)
            generation_config = types.GenerationConfig(response_mime_type="application/json", temperature=0.1)
            safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
            status_placeholder.info("📞 Calling Gemini API (expecting JSON)...")
            response = model.generate_content(contents=prompt_instructions, generation_config=generation_config, safety_settings=safety_settings, request_options={'timeout': 600})
            status_placeholder.info("📄 Processing Gemini JSON response...")
            graph_data, error_msg = parse_ai_response(response.text)
            if error_msg: st.session_state.dtg_error = error_msg
            else:
                st.session_state.dtg_graph_data = graph_data; st.session_state.dtg_error = None; st.toast("Term data extracted!", icon="📊")
                status_placeholder.info("⚙️ Analyzing graph structure...")
                st.session_state.dtg_nx_graph = build_networkx_graph(graph_data)
                if st.session_state.dtg_nx_graph:
                    st.session_state.dtg_cycles = find_cycles(st.session_state.dtg_nx_graph)
                    st.session_state.dtg_orphans = find_orphans(st.session_state.dtg_nx_graph)
                    st.toast("Graph analysis complete.", icon="🔬")
                else: st.warning("Could not build internal graph for analysis.")
        # --- Exception Handling ---
        except types.StopCandidateException as sce: st.session_state.dtg_error = f"Generation Stopped: {sce}. Response might be incomplete or blocked."; print(traceback.format_exc())
        except google.api_core.exceptions.GoogleAPIError as api_err: st.session_state.dtg_error = f"Google API Error: {api_err}. Check key/quota/permissions."; print(traceback.format_exc())
        except json.JSONDecodeError as json_err: st.session_state.dtg_error = f"Error Decoding AI Response: {json_err}. The AI did not return valid JSON."; print(traceback.format_exc()) # Specific JSON error
        except Exception as e: st.session_state.dtg_error = f"Processing Error: {e}"; print(traceback.format_exc())
        finally: st.session_state.dtg_processing = False; status_placeholder.empty(); st.rerun()

elif st.session_state.dtg_graph_data:
    # --- Display Results ---
    st.subheader(f"📊 Interactive Graph & Analysis for '{st.session_state.dtg_pdf_name}'")
    graph_data = st.session_state.dtg_graph_data; G = st.session_state.dtg_nx_graph
    terms_map = {term['name']: term['definition'] for term in graph_data.get('terms', [])}
    filter_term = st.session_state.dtg_filter_term; highlight_node = st.session_state.dtg_highlight_node

    # Filter nodes/edges
    nodes_to_display_names = set(G.nodes()) if G else set()
    if filter_term:
        try: filter_regex = re.compile(filter_term, re.IGNORECASE); nodes_to_display_names = {n for n in G.nodes() if filter_regex.search(n)}
        except re.error: st.warning("Invalid filter regex.", icon="⚠️"); nodes_to_display_names = set(G.nodes()) if G else set()
    # Determine highlight set
    highlight_neighbors_predecessors = set(); highlight_neighbors_successors = set()
    if highlight_node and G: highlight_neighbors_predecessors, highlight_neighbors_successors = get_neighbors(G, highlight_node)
    # Prepare Agraph Nodes & Edges
    agraph_nodes = []; agraph_edges = []; agraph_edges_tuples = [] # Store tuples for DOT gen
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
                 agraph_edges_tuples.append((u, v)) # Store tuple
                 agraph_edges.append(Edge(source=u, target=v, color="#CCCCCC")) # Create Agraph Edge

    # Configure Agraph
    is_physics = st.session_state.dtg_layout == 'Physics'
    config = Config(width='100%', height=700, directed=True, physics=is_physics, hierarchical=not is_physics,
                    highlightColor=HIGHLIGHT_COLOR, collapsible=False, node={'labelProperty':'label', 'size': 15},
                    physics_config={'barnesHut': {'gravitationalConstant': -10000, 'centralGravity': 0.1, 'springLength': 180, 'springConstant': 0.05, 'damping': 0.09, 'avoidOverlap': 0.1}, 'minVelocity': 0.75} if is_physics else None,
                    layout={'hierarchical': {'enabled': (not is_physics), 'sortMethod': 'directed', 'levelSeparation': 150, 'nodeSpacing': 120}} if not is_physics else None,
                    interaction={'navigationButtons': True, 'keyboard': True, 'tooltipDelay': 300, 'hover': True} )

    # Display Area
    graph_col, info_col = st.columns([3, 1])
    with graph_col:
        st.caption("Graph View: Click/drag to pan, scroll/pinch to zoom. Select node in sidebar to highlight.")
        if not agraph_nodes: st.warning(f"No nodes match filter: '{filter_term}'")
        else: agraph_return = agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)
    with info_col:
        st.subheader("Details & Analysis"); st.markdown("**Selected Definition:**")
        selected_def = terms_map.get(highlight_node, "_Select node in sidebar_")
        st.text_area("Definition Display", value=selected_def, height=150, disabled=True, label_visibility="collapsed", key="def_display_box")
        st.markdown("---"); st.markdown("**Graph Analysis:**")
        if st.session_state.dtg_cycles is not None:
             if st.session_state.dtg_cycles:
                  with st.expander(f"🚨 Found {len(st.session_state.dtg_cycles)} Circular Definitions", expanded=False):
                       for i, cycle in enumerate(st.session_state.dtg_cycles): st.markdown(f"- Cycle {i+1}: `{' → '.join(cycle)} → {cycle[0]}`")
             else: st.caption("✅ No circular definitions detected.")
        if st.session_state.dtg_orphans is not None:
             if st.session_state.dtg_orphans:
                  with st.expander(f"⚠️ Found {len(st.session_state.dtg_orphans)} Orphan Terms", expanded=False):
                       st.markdown(f"`{', '.join(st.session_state.dtg_orphans)}`")
                       st.caption("_Defined but not linked within definition network._")
             else: st.caption("✅ All defined terms linked.")
    st.divider()

    # Generate DOT Code for Download
    dot_lines = ["digraph G {"]; node_style_map = {node.id: f'[color="{node.color}", fontcolor="#000000"]' for node in agraph_nodes}
    for node_id in sorted(list(displayed_node_ids)): # Sort nodes for consistent DOT output
        style = node_style_map.get(node_id, "")
        # Ensure node_id is properly quoted if it contains spaces or special chars
        quoted_node_id = f'"{node_id}"' if re.search(r'\s|[^a-zA-Z0-9_]', node_id) else node_id
        dot_lines.append(f'  {quoted_node_id} {style};')
    for u, v in sorted(agraph_edges_tuples): # Sort edges for consistent DOT output
        # Ensure source and target are properly quoted
        quoted_u = f'"{u}"' if re.search(r'\s|[^a-zA-Z0-9_]', u) else u
        quoted_v = f'"{v}"' if re.search(r'\s|[^a-zA-Z0-9_]', v) else v
        dot_lines.append(f'  {quoted_u} -> {quoted_v};')
    dot_lines.append("}")
    generated_dot_code = "\n".join(dot_lines)

    # Download Buttons
    st.subheader("Export Graph"); export_cols = st.columns(4); safe_filename_base = re.sub(r'[^\w\-]+', '_', st.session_state.dtg_pdf_name or "graph")
    with export_cols[0]: export_cols[0].download_button(label="📥 DOT Code (.dot)", data=generated_dot_code, file_name=f"{safe_filename_base}_graph.dot", mime="text/vnd.graphviz", use_container_width=True)
    with export_cols[1]:
         try: g_render = graphviz.Source(generated_dot_code); png_bytes = g_render.pipe(format='png'); export_cols[1].download_button(label="🖼️ PNG Image (.png)", data=png_bytes, file_name=f"{safe_filename_base}_graph.png", mime="image/png", use_container_width=True)
         except graphviz.backend.execute.ExecutableNotFound: export_cols[1].warning("Graphviz executable not found for PNG render.", icon="⚠️")
         except Exception as render_err: export_cols[1].warning(f"PNG ERR: {render_err}", icon="⚠️")
    with export_cols[2]:
         try: g_render_svg = graphviz.Source(generated_dot_code); svg_bytes = g_render_svg.pipe(format='svg'); export_cols[2].download_button(label="📐 SVG Image (.svg)", data=svg_bytes, file_name=f"{safe_filename_base}_graph.svg", mime="image/svg+xml", use_container_width=True)
         except graphviz.backend.execute.ExecutableNotFound: export_cols[2].warning("Graphviz executable not found for SVG render.", icon="⚠️")
         except Exception as render_err: export_cols[2].warning(f"SVG ERR: {render_err}", icon="⚠️")
    with export_cols[3]:
        if G:
            try:
                 # Use the edges currently displayed (after filtering) for CSV
                 dep_list = [{"Source Term": u, "Depends On (Target Term)": v} for u, v in agraph_edges_tuples]
                 df_deps = pd.DataFrame(dep_list)
                 csv_output = df_deps.to_csv(index=False).encode('utf-8')
                 export_cols[3].download_button(label="📋 Dependencies (.csv)", data=csv_output, file_name=f"{safe_filename_base}_dependencies.csv", mime="text/csv", use_container_width=True)
            except Exception as csv_err: export_cols[3].warning(f"CSV ERR: {csv_err}", icon="⚠️")

    with st.expander("View Generated DOT Code (for current view)"): st.code(generated_dot_code, language='dot')

elif st.session_state.dtg_error: st.error(f"❌ Failed: {st.session_state.dtg_error}")
elif not st.session_state.dtg_pdf_bytes: st.info("⬆️ Upload a document (PDF or TXT) using the sidebar to get started.")
else: st.info("⬆️ Ready to generate. Click the 'Generate & Analyze Graph' button in the sidebar.")

# Footer
st.sidebar.markdown("---"); st.sidebar.markdown("Developed with Streamlit & Google Gemini")