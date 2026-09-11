import os
import io
import streamlit as st
import chromadb
from chromadb.config import Settings
from google import genai

# Third-party document parsers
import pypdf
import docx
from bs4 import BeautifulSoup

# Page Configuration
st.set_page_config(
    page_title="EliteBotStudios - ChromaDB RAG Agent Platform", 
    page_icon="🌿", 
    layout="wide"
)

# --- Custom CSS: Full Outer Screen & Background Styling ---
st.markdown("""
<style>
    /* Full Outer App & Margins Background */
    .stApp, [data-testid="stAppViewContainer"], .main, header[data-testid="stHeader"] {
        background: linear-gradient(180deg, #e0f2fe 0%, #f0f7f7 50%, #e8f5e9 100%) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Inner Page Container Padding */
    .main .block-container {
        background-color: transparent;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header Banner - Sky Blue to Emerald Gradient */
    .header-banner {
        background: linear-gradient(135deg, #0f4c81 0%, #1b4332 50%, #2d6a4f 100%);
        padding: 28px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 20px rgba(15, 76, 129, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .header-banner h1 {
        color: #ffffff !important;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .header-banner p {
        color: #e0f2fe;
        margin-top: 8px;
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* Sidebar - Light Mint Accent */
    section[data-testid="stSidebar"] {
        background-color: #d8ebd9 !important;
        border-right: 1px solid #b7e4c7;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        border-radius: 10px;
        padding-left: 20px;
        padding-right: 20px;
        font-weight: 600;
        color: #0f4c81;
        background-color: #e0f2fe;
        border: 1px solid #bae6fd;
        transition: all 0.2s ease-in-out;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2d6a4f !important;
        color: #ffffff !important;
        border-color: #2d6a4f !important;
        box-shadow: 0 4px 10px rgba(45, 106, 79, 0.25);
    }

    /* Primary Action Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #059669 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.2);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369a1 0%, #047857 100%);
        box-shadow: 0 6px 16px rgba(3, 105, 161, 0.3);
        transform: translateY(-1px);
    }

    /* Agent Result Cards */
    .agent-card {
        background-color: #ffffff;
        border-left: 5px solid #0284c7;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        border-top: 1px solid #e0f2fe;
        border-right: 1px solid #e0f2fe;
        border-bottom: 1px solid #e0f2fe;
    }
    .agent-title {
        color: #0f4c81;
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Custom Metric Styling */
    [data-testid="stMetricValue"] {
        color: #059669 !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Application Header Banner
st.markdown("""
<div class="header-banner">
    <h1>🌿 EliteBotStudios Platform</h1>
    <p>Action-Oriented Multi-Agent & ChromaDB Vector RAG Workflow Engine</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Google Gemini API Key:", type="password")
    selected_model = st.selectbox("Select Gemini Model:", ["gemini-2.5-flash", "gemini-2.5-pro"])
    st.divider()
    st.info("💡 **Tip:** Upload your docs in Tab 1, then run multi-agent workflows in Tab 2.")

# Universal File Text Extractor
def extract_text_from_file(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    # 1. PDF Documents
    if filename.endswith(".pdf"):
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n\n"
        return text

    # 2. Word Documents (.docx)
    elif filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])

    # 3. HTML / Web Documents
    elif filename.endswith(".html") or filename.endswith(".htm"):
        soup = BeautifulSoup(file_bytes, "html.parser")
        return soup.get_text(separator="\n\n")

    # 4. Text, Markdown, CSV, JSON, Log files, etc.
    else:
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="ignore")

# Helper class to wrap Gemini Embedding API for ChromaDB
class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input:
            res = self.client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            embeddings.append(res.embeddings[0].values)
        return embeddings

# Initialize ChromaDB Client in Session State
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []

# Main Layout Tabs
tab1, tab2, tab3 = st.tabs([
    "📚 Vector Knowledge Base", 
    "🤖 Multi-Agent Executor", 
    "📊 Workflow Logs"
])

# Tab 1: Knowledge Base Management
with tab1:
    st.subheader("Document Indexing")
    st.caption("Upload files (.pdf, .docx, .txt, .md, .csv, .json, .html) to index them into ChromaDB for semantic retrieval.")
    
    uploaded_files = st.file_uploader(
        "Upload your documents:", 
        type=["txt", "md", "pdf", "docx", "csv", "json", "html", "htm", "log"], 
        accept_multiple_files=True
    )

    if st.button("📥 Index Documents into ChromaDB", use_container_width=True, type="primary"):
        if not api_key:
            st.error("Please enter your Google Gemini API Key in the sidebar.")
        elif not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Processing & embedding documents into ChromaDB..."):
                embed_fn = GeminiEmbeddingFunction(api_key=api_key)
                
                try:
                    st.session_state.chroma_client.delete_collection("elitebot_docs")
                except Exception:
                    pass

                collection = st.session_state.chroma_client.create_collection(
                    name="elitebot_docs",
                    embedding_function=embed_fn
                )

                documents, metadatas, ids = [], [], []
                doc_counter = 0

                for file in uploaded_files:
                    content = extract_text_from_file(file)
                    chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        doc_counter += 1
                        documents.append(chunk)
                        metadatas.append({"source": file.name, "chunk": chunk_idx})
                        ids.append(f"doc_{doc_counter}")

                if documents:
                    collection.add(documents=documents, metadatas=metadatas, ids=ids)
                    st.success(f"Successfully indexed **{len(documents)}** chunks across **{len(uploaded_files)}** document(s)!")

    st.divider()
    try:
        col = st.session_state.chroma_client.get_collection("elitebot_docs")
        st.metric(label="Active ChromaDB Vector Chunks", value=col.count())
    except Exception:
        st.metric(label="Active ChromaDB Vector Chunks", value=0)

# Tab 2: Workflow Executor
with tab2:
    st.subheader("Multi-Agent Execution Engine")
    st.caption("Define the goal and let your team of specialized AI agents solve it.")

    col_left, col_right = st.columns([3, 1])
    with col_left:
        user_goal = st.text_area("Workflow Goal / Problem Statement:", placeholder="Example: Search company policy docs to evaluate customer dispute resolution steps.", height=120)
    with col_right:
        top_k = st.slider("Retrieval Count (k):", min_value=1, max_value=10, value=3)

    if st.button("🚀 Run Multi-Agent Execution", use_container_width=True, type="primary"):
        if not api_key:
            st.error("Please enter your Google Gemini API Key in the sidebar.")
        elif not user_goal.strip():
            st.warning("Please provide a task or goal for the agents.")
        else:
            client = genai.Client(api_key=api_key)
            st.session_state.agent_logs = []

            retrieved_context = "No relevant documents found in ChromaDB."
            try:
                embed_fn = GeminiEmbeddingFunction(api_key=api_key)
                collection = st.session_state.chroma_client.get_collection(
                    name="elitebot_docs",
                    embedding_function=embed_fn
                )
                results = collection.query(query_texts=[user_goal], n_results=top_k)
                if results and results.get("documents") and len(results["documents"][0]) > 0:
                    retrieved_context = " --- ".join(results["documents"][0])
            except ValueError:
                retrieved_context = "No documents found in ChromaDB. Please index documents in Tab 1 first."
            except Exception as e:
                retrieved_context = f"Vector retrieval error: {str(e)}"

            with st.status("Executing Multi-Agent Workflow...", expanded=True) as status:
                st.write("🔍 **Agent 1 (Researcher):** Retrieving & analyzing context...")
                prompt_agent1 = f"You are Agent 1 (Researcher Agent). Context:\n{retrieved_context}\n\nGoal:\n{user_goal}\nProvide structured findings and key facts."
                res1 = client.models.generate_content(model=selected_model, contents=prompt_agent1)
                st.session_state.agent_logs.append(("Agent 1 (Researcher)", res1.text))

                st.write("💡 **Agent 2 (Planner):** Formulating strategy...")
                prompt_agent2 = f"You are Agent 2 (Strategy Agent). Research findings:\n{res1.text}\n\nGoal:\n{user_goal}\nProvide actionable resolution steps."
                res2 = client.models.generate_content(model=selected_model, contents=prompt_agent2)
                st.session_state.agent_logs.append(("Agent 2 (Planner)", res2.text))

                st.write("⚙️ **Agent 3 (Executor):** Generating final deliverable...")
                prompt_agent3 = f"You are Agent 3 (Execution Agent). Strategy plan:\n{res2.text}\n\nGenerate final ready-to-use output."
                res3 = client.models.generate_content(model=selected_model, contents=prompt_agent3)
                st.session_state.agent_logs.append(("Agent 3 (Executor)", res3.text))

                status.update(label="Workflow Execution Complete!", state="complete", expanded=False)

            st.success("Execution completed! View details in the Workflow Logs tab.")

# Tab 3: Workflow Logs
with tab3:
    st.subheader("Execution Output & Agent Logs")
    if not st.session_state.agent_logs:
        st.info("No workflow results yet. Execute a task in the **Multi-Agent Executor** tab.")
    else:
        for agent_name, output in st.session_state.agent_logs:
            st.markdown(f"""
            <div class="agent-card">
                <div class="agent-title">📌 {agent_name}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(output)
            st.divider()
