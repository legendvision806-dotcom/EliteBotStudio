
import os
import streamlit as st
import chromadb
from chromadb.config import Settings
from google import genai

# Page Configuration
st.set_page_config(
    page_title="EliteBotStudios - ChromaDB RAG Agent Platform", 
    page_icon="🌿", 
    layout="wide"
)

# --- Custom CSS for Light Green & Unique Styling ---
st.markdown("""
<style>
    /* Main Background Accent - Soft Mint Neutral */
    .stApp {
        background-color: #f4f8f5;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Title Banner Styling - Emerald / Forest Gradient */
    .header-banner {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 60%, #40916c 100%);
        padding: 28px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 20px rgba(27, 67, 50, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .header-banner h1 {
        color: #ffffff !important;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .header-banner p {
        color: #d8f3dc;
        margin-top: 8px;
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #ebf4ee;
        border-right: 1px solid #d8f3dc;
    }

    /* Tab Customization */
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
        color: #2d6a4f;
        background-color: #e2ece9;
        border: 1px solid #d8f3dc;
        transition: all 0.2s ease-in-out;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2d6a4f !important;
        color: #ffffff !important;
        border-color: #2d6a4f !important;
        box-shadow: 0 4px 10px rgba(45, 106, 79, 0.2);
    }

    /* Primary Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2d6a4f 0%, #40916c 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(45, 106, 79, 0.15);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        box-shadow: 0 6px 16px rgba(27, 67, 50, 0.25);
        transform: translateY(-1px);
    }

    /* Agent Result Cards */
    .agent-card {
        background-color: #ffffff;
        border-left: 5px solid #52b788;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        border-top: 1px solid #eaf4ed;
        border-right: 1px solid #eaf4ed;
        border-bottom: 1px solid #eaf4ed;
    }
    .agent-title {
        color: #1b4332;
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Custom Metric Container */
    [data-testid="stMetricValue"] {
        color: #2d6a4f !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
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
    st.info("💡 **Tip:** Upload your docs in Tab 1, then head to Tab 2 to run your multi-agent workflow.")

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
    st.caption("Upload text or markdown files to index them into ChromaDB for semantic retrieval.")
    
    with st.container():
        uploaded_files = st.file_uploader("Upload Text (.txt) or Markdown (.md) documents:", type=["txt", "md"], accept_multiple_files=True)

        if st.button("📥 Index Documents into ChromaDB", use_container_width=True, type="primary"):
            if not api_key:
                st.error("Please enter your Google Gemini API Key in the sidebar.")
            elif not uploaded_files:
                st.warning("Please upload at least one text file.")
            else:
                with st.spinner("Processing & embedding documents into ChromaDB vector database..."):
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
                        content = file.read().decode("utf-8")
                        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
                        
                        for chunk_idx, chunk in enumerate(chunks):
                            doc_counter += 1
                            documents.append(chunk)
                            metadatas.append({"source": file.name, "chunk": chunk_idx})
                            ids.append(f"doc_{doc_counter}")

                    if documents:
                        collection.add(documents=documents, metadatas=metadatas, ids=ids)
                        st.success(f"Successfully indexed **{len(documents)}** chunks across **{len(uploaded_files)}** document(s)!")

    # Collection Stats Display
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

            # Step 1: Semantic Search Retrieval
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
                retrieved_context = f"Vector retrieval note: {str(e)}"

            # Execution Pipeline
            with st.status("Executing Multi-Agent Workflow...", expanded=True) as status:
                st.write("🔍 **Agent 1 (Researcher):** Retrieving & analyzing context...")
                prompt_agent1 = f"You are Agent 1 (Researcher Agent). Context: {retrieved_context}\n\nGoal: {user_goal}\nProvide a structured plan and key facts."
                res1 = client.models.generate_content(model=selected_model, contents=prompt_agent1)
                agent1_output = res1.text
                st.session_state.agent_logs.append(("Agent 1 (Researcher)", agent1_output))

                st.write("💡 **Agent 2 (Planner):** Formulating strategy...")
                prompt_agent2 = f"You are Agent 2 (Strategy Agent). Research findings: {agent1_output}\n\nGoal: {user_goal}\nProvide actionable steps."
                res2 = client.models.generate_content(model=selected_model, contents=prompt_agent2)
                agent2_output = res2.text
                st.session_state.agent_logs.append(("Agent 2 (Planner)", agent2_output))

                st.write("⚙️ **Agent 3 (Executor):** Generating final deliverable...")
                prompt_agent3 = f"You are Agent 3 (Execution Agent). Strategy plan: {agent2_output}\n\nGenerate the final ready-to-use output."
                res3 = client.models.generate_content(model=selected_model, contents=prompt_agent3)
                agent3_output = res3.text
                st.session_state.agent_logs.append(("Agent 3 (Executor)", agent3_output))

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
