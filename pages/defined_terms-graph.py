# pages/defined_terms_graph.py
import streamlit as st
import google.generativeai as genai
from google.generativeai import types
import fitz  # PyMuPDF for PDF text extraction
import re
import os
import traceback
import time
import io
import json
import graphviz # Python graphviz library for parsing DOT and rendering
import networkx as nx # For graph analysis (cycles, orphans, neighbors)
import pandas as pd # For CSV export
from streamlit_agraph import agraph, Node, Edge, Config
from PIL import Image # For Logo import
from collections import defaultdict

# --- Configuration ---
MODEL_NAME = "gemini-1.5-pro-latest"
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
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Function for Text Extraction (Unchanged) ---
@st.cache_data(show_spinner="Extracting text from PDF...")
def extract_text_from_pdf(pdf_bytes):
    # ... (identical to previous version) ...
    if not pdf_bytes: return None, "No PDF file provided."
    try: doc = fitz.open(stream=pdf_bytes, filetype="pdf"); text = "";
    for page_num in range(len(doc)): page = doc.load_page(page_num); text += page.get_text("text", sort=True); text += "\n--- Page Break --- \n"
    doc.close();
    if not text.strip(): return None, "Could not extract any text from the PDF."
    return text, None
    except Exception as e: error_msg = f"Error extracting text: {e}"; print(traceback.format_exc()); return None, error_msg

# --- Helper Function to Parse AI JSON Response ---
def parse_ai_response(response_text):
    """Parses the AI's JSON response for terms, definitions, and edges."""
    try:
        # Attempt to handle potential markdown fences
        match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.IGNORECASE | re.DOTALL)
        if match:
            json_text = match.group(1).strip()
        else:
            json_text = response_text.strip()

        if not json_text:
            return None, "AI response content is empty."

        data = json.loads(json_text)

        # Validate basic structure
        if not isinstance(data, dict):
            return None, "AI response is not a JSON object."
        if "terms" not in data or "edges" not in data:
            return None, "AI response missing required 'terms' or 'edges' keys."
        if not isinstance(data["terms"], list) or not isinstance(data["edges"], list):
            return None, "'terms' or 'edges' are not lists in the AI response."

        # Further validation (optional but recommended)
        validated_terms = []
        term_names = set()
        for item in data["terms"]:
            if isinstance(item, dict) and "name" in item and "definition" in item:
                term_name = item["name"]
                if term_name not in term_names:
                    validated_terms.append({"name": term_name, "definition": item["definition"]})
                    term_names.add(term_name)
            # else: warning about malformed term item

        validated_edges = []
        for edge in data["edges"]:
            if isinstance(edge, dict) and "source" in edge and "target" in edge:
                 # Ensure source and target are actual defined terms found
                if edge["source"] in term_names and edge["target"] in term_names:
                    validated_edges.append({"source": edge["source"], "target": edge["target"]})
            # else: warning about malformed edge item or edge linking non-defined term

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
    for term_data in graph_data['terms']:
        G.add_node(term_data['name'], definition=term_data['definition'])
    for edge_data in graph_data['edges']:
        G.add_edge(edge_data['source'], edge_data['target'])
    return G

def find_cycles(G):
    """Finds simple cycles in a NetworkX DiGraph."""
    if G is None: return None
    try:
        return list(nx.simple_cycles(G))
    except Exception as e:
        print(f"Error finding cycles: {e}")
        return None

def find_orphans(G):
    """Finds nodes with in-degree and out-degree of 0."""
    if G is None: return None
    orphans = [node for node, degree in G.degree() if degree == 0]
    # The above considers total degree. For directed, we need in & out separately.
    orphans_directed = [
        node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 0
    ]
    return orphans_directed

def get_neighbors(G, node_id):
    """Gets predecessors (pointing to node) and successors (node points to)."""
    if G is None or node_id not in G:
        return set(), set()
    predecessors = set(G.predecessors(node_id))
    successors = set(G.successors(node_id))
    return predecessors, successors

# --- Streamlit UI ---

# --- Header (Fixed Logo Alt Issue) ---
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_FILE = "jasper-logo-1.png"
LOGO_PATH = os.path.join(APP_DIR, LOGO_FILE)
header_cols = st.columns([1, 5])
with header_cols[0]:
    if os.path.exists(LOGO_PATH):
        try:
            logo = Image.open(LOGO_PATH)
            st.image(logo, width=80, caption=None, output_format='PNG') # Removed alt
        except Exception as img_err:
            st.warning(f"Could not load logo: {img_err}")
with header_cols[1]:
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.caption("Upload a document, generate an interactive graph of defined terms, and analyze relationships.")
st.divider()

# --- Sidebar Controls (Expanded) ---
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

# Process uploaded file (reset state on new upload)
if uploaded_file_obj is not None:
    if uploaded_file_obj.name != st.session_state.get('dtg_pdf_name'):
        st.session_state.dtg_pdf_bytes = uploaded_file_obj.getvalue()
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
             if error_msg: st.session_state.dtg_error = f"Failed to extract text from PDF: {error_msg}"
             else: st.session_state.dtg_extracted_text = extracted_text; st.toast("Text extracted from PDF.", icon="📝")
        elif uploaded_file_obj.type == "text/plain":
             try: st.session_state.dtg_extracted_text = st.session_state.dtg_pdf_bytes.decode('utf-8'); st.toast("Text loaded from TXT file.", icon="📝")
             except Exception as e: st.session_state.dtg_error = f"Failed to read text file: {e}"
        else: st.session_state.dtg_error = f"Unsupported file type: {uploaded_file_obj.type}"
        st.rerun()

if st.session_state.dtg_error and not st.session_state.dtg_processing and not st.session_state.dtg_graph_data:
     st.error(st.session_state.dtg_error)

# Generation Button
st.sidebar.markdown("### 2. Generate & Analyze")
can_generate = (st.session_state.api_key and
                st.session_state.dtg_pdf_bytes and
                st.session_state.dtg_extracted_text and
                not st.session_state.dtg_processing)
# ... (tooltip logic) ...
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

# --- Graph Interaction Controls (Show only if graph data exists) ---
if st.session_state.dtg_graph_data:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 3. Graph Interaction")

    # Filter Term Input
    st.session_state.dtg_filter_term = st.sidebar.text_input(
        "Filter Nodes (by name)",
        value=st.session_state.dtg_filter_term,
        placeholder="Type term to filter...",
        key="dtg_filter_input"
    ).strip()

    # Highlight Node Selectbox
    # Get available node names (potentially filtered) for selection
    available_nodes = ["--- Select Node ---"]
    if st.session_state.dtg_nx_graph:
         nodes_to_consider = list(st.session_state.dtg_nx_graph.nodes())
         if st.session_state.dtg_filter_term:
              nodes_to_consider = [n for n in nodes_to_consider if st.session_state.dtg_filter_term.lower() in n.lower()]
         available_nodes.extend(sorted(nodes_to_consider))

    st.session_state.dtg_highlight_node = st.sidebar.selectbox(
         "Highlight Node & Neighbors",
         options=available_nodes,
         index=0 if not st.session_state.dtg_highlight_node or st.session_state.dtg_highlight_node not in available_nodes else available_nodes.index(st.session_state.dtg_highlight_node),
         key="dtg_highlight_select",
         help="Select a node to highlight it and its direct dependencies/dependents."
    )
    if st.session_state.dtg_highlight_node == "--- Select Node ---":
        st.session_state.dtg_highlight_node = None # Treat placeholder as None

    # Layout Selection
    st.session_state.dtg_layout = st.sidebar.radio(
        "Graph Layout",
        options=['Physics', 'Hierarchical'],
        index=0 if st.session_state.dtg_layout == 'Physics' else 1,
        key="dtg_layout_radio",
        help="Choose layout algorithm. Physics is often good for clusters, Hierarchical for dependencies."
    )


# --- Main Area: Processing and Display ---
if st.session_state.dtg_processing:
    # --- Processing Logic (Uses Revised Prompt for JSON) ---
    status_placeholder = st.empty()
    with st.spinner(f"⚙️ Analyzing '{st.session_state.dtg_pdf_name}' with Gemini..."):
        status_placeholder.info("🧠 Asking Gemini to extract terms, definitions, and relationships...")
        try:
            genai.configure(api_key=st.session_state.api_key)
            document_text = st.session_state.dtg_extracted_text

            # Use the revised prompt requesting JSON
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
            # Configure for JSON output
            generation_config = types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1,
            )
            safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]

            status_placeholder.info("📞 Calling Gemini API (expecting JSON)...")
            response = model.generate_content(
                contents=prompt_instructions,
                generation_config=generation_config,
                safety_settings=safety_settings,
                request_options={'timeout': 600}
            )

            status_placeholder.info("📄 Processing Gemini JSON response...")
            graph_data, error_msg = parse_ai_response(response.text)

            if error_msg:
                st.session_state.dtg_error = error_msg
            else:
                st.session_state.dtg_graph_data = graph_data
                st.session_state.dtg_error = None
                st.toast("Term data extracted successfully!", icon="📊")

                # Perform graph analysis immediately after getting data
                status_placeholder.info("⚙️ Analyzing graph structure...")
                st.session_state.dtg_nx_graph = build_networkx_graph(graph_data)
                if st.session_state.dtg_nx_graph:
                    st.session_state.dtg_cycles = find_cycles(st.session_state.dtg_nx_graph)
                    st.session_state.dtg_orphans = find_orphans(st.session_state.dtg_nx_graph)
                    st.toast("Graph analysis complete.", icon="🔬")
                else:
                     st.warning("Could not build internal graph for analysis.")


        # ... (exception handling remains the same) ...
        except types.StopCandidateException as sce: st.session_state.dtg_error = f"Generation Stopped Error: {sce}. Response might be incomplete/blocked."; print(traceback.format_exc())
        except google.api_core.exceptions.GoogleAPIError as api_err: st.session_state.dtg_error = f"Google API Error: {api_err}. Check API key/quota/permissions."; print(traceback.format_exc())
        except Exception as e: st.session_state.dtg_error = f"An unexpected error occurred: {e}"; print(traceback.format_exc())
        finally:
            st.session_state.dtg_processing = False
            status_placeholder.empty()
            st.rerun()

elif st.session_state.dtg_graph_data:
    # --- Display Results ---
    st.subheader(f"📊 Interactive Graph & Analysis for '{st.session_state.dtg_pdf_name}'")

    # Get data and graph object
    graph_data = st.session_state.dtg_graph_data
    G = st.session_state.dtg_nx_graph
    terms_map = {term['name']: term['definition'] for term in graph_data.get('terms', [])}

    # --- Filtering Logic ---
    filter_term = st.session_state.dtg_filter_term
    nodes_to_display = list(G.nodes()) if G else []
    edges_to_display = list(G.edges()) if G else []

    if filter_term:
        # Simple filter: keep nodes containing the filter term (case-insensitive)
        # and only edges connecting two kept nodes.
        filtered_node_names = {n for n in G.nodes() if filter_term.lower() in n.lower()}
        nodes_to_display = [n for n in G.nodes() if n in filtered_node_names]
        edges_to_display = [(u, v) for u, v in G.edges() if u in filtered_node_names and v in filtered_node_names]
        st.caption(f"Filtering nodes containing: '{filter_term}'")


    # --- Highlighting Logic ---
    highlight_node = st.session_state.dtg_highlight_node
    highlight_neighbors_predecessors = set()
    highlight_neighbors_successors = set()
    if highlight_node and G:
        highlight_neighbors_predecessors, highlight_neighbors_successors = get_neighbors(G, highlight_node)


    # --- Prepare Agraph Nodes & Edges ---
    agraph_nodes = []
    agraph_edges = []
    displayed_node_ids = set(nodes_to_display) # Keep track of nodes actually being displayed

    for node_id in nodes_to_display:
        node_color = DEFAULT_NODE_COLOR
        node_size = 15
        # Apply highlighting
        if node_id == highlight_node:
            node_color = HIGHLIGHT_COLOR
            node_size = 25 # Make selected node bigger
        elif node_id in highlight_neighbors_predecessors or node_id in highlight_neighbors_successors:
            node_color = NEIGHBOR_COLOR
            node_size = 20 # Make neighbors slightly bigger

        # Check if the original graph had this marked blue (more robust check needed if AI provided this)
        # For now, assume AI is not providing blue nodes anymore with the new prompt
        # if node_id in originally_blue_nodes: # Replace with actual check if needed
        #    node_color = "blue" # Overwrite highlight if it was supposed to be blue

        agraph_nodes.append(Node(id=node_id,
                                 label=node_id,
                                 color=node_color,
                                 size=node_size,
                                 font={'color': "#000000"} # Always black font
                                 ))

    for u, v in edges_to_display:
         # Only add edges if both nodes are being displayed (handles filtering)
         if u in displayed_node_ids and v in displayed_node_ids:
            agraph_edges.append(Edge(source=u, target=v, color="#CCCCCC"))

    # --- Configure Agraph ---
    is_physics = st.session_state.dtg_layout == 'Physics'
    config = Config(width='100%',
                    height=700,
                    directed=True,
                    physics=is_physics, # Set based on radio button
                    hierarchical=not is_physics, # Enable if physics is off
                    highlightColor=HIGHLIGHT_COLOR,
                    collapsible=False,
                    node={'labelProperty':'label', 'size': 15},
                    # Conditional physics/hierarchical settings
                    physics_config={'barnesHut': {'gravitationalConstant': -10000, 'centralGravity': 0.1, 'springLength': 180, 'springConstant': 0.05, 'damping': 0.09, 'avoidOverlap': 0.1}, 'minVelocity': 0.75} if is_physics else None,
                    layout={'hierarchical': {'enabled': (not is_physics), 'sortMethod': 'directed'}} if not is_physics else None, # Basic hierarchical settings
                    interaction={'navigationButtons': True, 'keyboard': True, 'tooltipDelay': 300, 'hover': True}
                   )

    # --- Display Area ---
    graph_col, info_col = st.columns([3, 1]) # Graph takes more space

    with graph_col:
        st.caption("Graph View: Click/drag to pan, scroll/pinch to zoom. Select node in sidebar to highlight.")
        if not agraph_nodes:
            st.warning("No nodes match the current filter criteria.")
        else:
            agraph_return = agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)
            # Note: Using agraph_return for hover/click can be complex, sticking to sidebar select for now.

    with info_col:
        st.subheader("Details & Analysis")

        # Definition Display
        st.markdown("**Selected Definition:**")
        selected_def = terms_map.get(highlight_node, "_Select a node in the sidebar to see its definition._")
        st.markdown(f'<div class="definition-box">{selected_def}</div>', unsafe_allow_html=True)
        st.markdown("---") # Separator

        # Analysis Display
        st.markdown("**Graph Analysis:**")
        if st.session_state.dtg_cycles is not None:
             if st.session_state.dtg_cycles:
                  with st.expander(f"🚨 Found {len(st.session_state.dtg_cycles)} Circular Definitions", expanded=False):
                       for i, cycle in enumerate(st.session_state.dtg_cycles):
                            st.markdown(f"- Cycle {i+1}: `{' -> '.join(cycle)} -> {cycle[0]}`")
             else:
                  st.caption("✅ No circular definitions detected.")

        if st.session_state.dtg_orphans is not None:
             if st.session_state.dtg_orphans:
                  with st.expander(f"⚠️ Found {len(st.session_state.dtg_orphans)} Orphan Terms (Defined but not linked)", expanded=False):
                       st.markdown(f"`{', '.join(st.session_state.dtg_orphans)}`")
                       st.caption("_These terms were defined but not used in other definitions and their definitions didn't use other defined terms._")
             else:
                  st.caption("✅ All defined terms are linked within the definition network.")

    st.divider()

    # --- Generate DOT Code for Download ---
    # Generate DOT based on the *currently displayed* nodes/edges (respecting filter)
    dot_lines = ["digraph G {"]
    # Add node definitions with potential highlight colors
    node_style_map = {node.id: f'[color="{node.color}", fontcolor="{node.font.get("color", "#000000")}"]' for node in agraph_nodes}
    for node_id in displayed_node_ids:
         style = node_style_map.get(node_id, "")
         dot_lines.append(f'  "{node_id}" {style};')
    # Add edges
    for edge in agraph_edges:
        dot_lines.append(f'  "{edge.source}" -> "{edge.target}";')
    dot_lines.append("}")
    generated_dot_code = "\n".join(dot_lines)

    # --- Download Buttons ---
    st.subheader("Export Graph")
    export_cols = st.columns(4)
    safe_filename_base = re.sub(r'[^\w\-]+', '_', st.session_state.dtg_pdf_name or "graph")

    with export_cols[0]:
        export_cols[0].download_button(
            label="📥 DOT Code (.dot)", data=generated_dot_code,
            file_name=f"{safe_filename_base}_graph.dot", mime="text/vnd.graphviz", use_container_width=True
        )
    with export_cols[1]:
         try:
              g_render = graphviz.Source(generated_dot_code) # Use generated DOT
              png_bytes = g_render.pipe(format='png')
              export_cols[1].download_button(
                  label="🖼️ PNG Image (.png)", data=png_bytes,
                  file_name=f"{safe_filename_base}_graph.png", mime="image/png", use_container_width=True
              )
         except Exception as render_err: export_cols[1].warning(f"PNG ERR: {render_err}", icon="⚠️")
    with export_cols[2]:
         try:
              g_render_svg = graphviz.Source(generated_dot_code) # Use generated DOT
              svg_bytes = g_render_svg.pipe(format='svg')
              export_cols[2].download_button(
                  label="📐 SVG Image (.svg)", data=svg_bytes,
                  file_name=f"{safe_filename_base}_graph.svg", mime="image/svg+xml", use_container_width=True
              )
         except Exception as render_err: export_cols[2].warning(f"SVG ERR: {render_err}", icon="⚠️")
    with export_cols[3]:
        # Export Dependency List (CSV)
        if G:
            try:
                dep_list = [{"Source Term": u, "Depends On (Target Term)": v} for u, v in G.edges()]
                df_deps = pd.DataFrame(dep_list)
                csv_output = df_deps.to_csv(index=False).encode('utf-8')
                export_cols[3].download_button(
                    label="📋 Dependencies (.csv)", data=csv_output,
                    file_name=f"{safe_filename_base}_dependencies.csv", mime="text/csv", use_container_width=True
                )
            except Exception as csv_err:
                export_cols[3].warning(f"CSV ERR: {csv_err}", icon="⚠️")


    # --- DOT Code Expander ---
    with st.expander("View Generated DOT Code (for current view)"):
        st.code(generated_dot_code, language='dot')


elif st.session_state.dtg_error:
    # Error Display
    st.error(f"❌ Failed: {st.session_state.dtg_error}")

elif not st.session_state.dtg_pdf_bytes:
    # Initial state
    st.info("⬆️ Upload a document (PDF or TXT) using the sidebar to get started.")

else:
    # Ready state
    st.info("⬆️ Ready to generate. Click the 'Generate & Analyze Graph' button in the sidebar.")


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with Streamlit & Google Gemini")