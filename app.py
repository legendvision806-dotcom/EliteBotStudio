import os
import streamlit as st
import chromadb
from chromadb.config import Settings
from google import genai

# Page Configuration
st.set_page_config(page_title="EliteBotStudios - ChromaDB RAG Agent Platform", page_icon="🤖", layout="wide")

st.title("🤖 EliteBotStudios Platform")
st.subheader("Action-Oriented Multi-Agent & ChromaDB Vector RAG Workflow Engine")

# Sidebar - Setup
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Google Gemini API Key:", type="password")
    selected_model = st.selectbox("Select Gemini Model:", ["gemini-2.5-flash", "gemini-2.5-pro"])
    st.markdown("---")
    st.info("EliteBotStudios converts user ideas into multi-agent workflows using ChromaDB Vector RAG and Google Gemini.")

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
tab1, tab2, tab3 = st.tabs(["📚 Vector Knowledge Base (ChromaDB)", "🤖 Multi-Agent Executor", "📊 Workflow Logs"])

# Tab 1: Multi-Document Upload & Vector Store Indexing
with tab1:
    st.header("Upload & Index Documents into ChromaDB")
    uploaded_files = st.file_uploader("Upload Text (.txt) or Markdown (.md) documents:", type=["txt", "md"], accept_multiple_files=True)
    
    if st.button("Index Documents into ChromaDB"):
        if not api_key:
            st.error("Please enter your Google Gemini API Key in the sidebar.")
        elif not uploaded_files:
            st.warning("Please upload at least one text file.")
        else:
            with st.spinner("Processing & embedding documents into ChromaDB vector database..."):
                embed_fn = GeminiEmbeddingFunction(api_key=api_key)
                
                # Delete existing collection if re-indexing for demo simplicity
                try:
                    st.session_state.chroma_client.delete_collection("elitebot_docs")
                except Exception:
                    pass

                collection = st.session_state.chroma_client.create_collection(
                    name="elitebot_docs",
                    embedding_function=embed_fn
                )

                documents = []
                metadatas = []
                ids = []

                doc_counter = 0
                for file in uploaded_files:
                    content = file.read().decode("utf-8")
                    # Simple line/chunk split
                    chunks = [c.strip() for c in content.split("\n\n

") if c.strip()]
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        doc_counter += 1
                        documents.append(chunk)
                        metadatas.append({"source": file.name, "chunk": chunk_idx})
                        ids.append(f"doc_{doc_counter}")

                if documents:
                    collection.add(documents=documents, metadatas=metadatas, ids=ids)
                    st.success(f"Successfully indexed {len(documents)} chunks across {len(uploaded_files)} document(s) into ChromaDB!")

    # View Collection Stats
    try:
        col = st.session_state.chroma_client.get_collection("elitebot_docs")
        st.write(f"**Current ChromaDB Collection Size:** {col.count()} vector chunks stored.")
    except Exception:
        st.write("**Current ChromaDB Collection Size:** 0 vector chunks stored.")

# Tab 2: Multi-Agent Workflow Execution with Semantic Retrieval
with tab2:
    st.header("Execute Multi-Agent Workflow")
    user_goal = st.text_area("What complex problem or task should the agents solve?", 
                             placeholder="Example: Search company policy docs to evaluate customer dispute resolution steps.")

    top_k = st.slider("Number of Vector Chunks to Retrieve (k):", min_value=1, max_value=10, value=3)

    if st.button("🚀 Run Multi-Agent Execution"):
        if not api_key:
            st.error("Please enter your Google Gemini API Key in the sidebar.")
        elif not user_goal.strip():
            st.warning("Please provide a task or goal for the agents.")
        else:
            client = genai.Client(api_key=api_key)
            st.session_state.agent_logs = []

            # Step 1: Semantic Search Retrieval via ChromaDB
            retrieved_context = "No relevant documents found in ChromaDB."
            try:
                embed_fn = GeminiEmbeddingFunction(api_key=api_key)
                collection = st.session_state.chroma_client.get_collection(
                    name="elitebot_docs", 
                    embedding_function=embed_fn
                )
                results = collection.query(query_texts=[user_goal], n_results=top_k)
                
                if results and results.get("documents") and len(results["documents"][0]) > 0:
                    retrieved_context = "
---
".join(results["documents"][0])
            except Exception as e:
                retrieved_context = f"Vector retrieval note: {str(e)}"

            # --- Agent 1: Researcher & ChromaDB RAG Context Analyzer ---
            with st.spinner("Agent 1 (Researcher) retrieving semantic context from ChromaDB..."):
                prompt_agent1 = f"""
                You are Agent 1 (Researcher Agent) at EliteBotStudios.
                Goal: Analyze the following user task using context semantically retrieved from ChromaDB.

                Retrieved Vector Context:
                {retrieved_context}

                User Goal:
                {user_goal}

                Provide a structured plan and extract key facts needed to solve this problem.
                """
                res1 = client.models.generate_content(model=selected_model, contents=prompt_agent1)
                agent1_output = res1.text
                st.session_state.agent_logs.append(("Agent 1 (Researcher)", agent1_output))

            # --- Agent 2: Problem Solver & Action Planner ---
            with st.spinner("Agent 2 (Planner) devising solution strategy..."):
                prompt_agent2 = f"""
                You are Agent 2 (Strategy & Action Agent) at EliteBotStudios.
                Review the research findings from Agent 1 and create a clear, step-by-step resolution plan.

                Agent 1 Research Findings:
                {agent1_output}

                User Goal:
                {user_goal}

                Provide actionable steps and specific decisions.
                """
                res2 = client.models.generate_content(model=selected_model, contents=prompt_agent2)
                agent2_output = res2.text
                st.session_state.agent_logs.append(("Agent 2 (Planner)", agent2_output))

            # --- Agent 3: Action Executor ---
            with st.spinner("Agent 3 (Executor) producing final action deliverables..."):
                prompt_agent3 = f"""
                You are Agent 3 (Execution Agent) at EliteBotStudios.
                Take the strategy from Agent 2 and execute the final output (e.g., formal report, finalized response, or action draft).

                Strategy Plan from Agent 2:
                {agent2_output}

                Generate the final ready-to-use output.
                """
                res3 = client.models.generate_content(model=selected_model, contents=prompt_agent3)
                agent3_output = res3.text
                st.session_state.agent_logs.append(("Agent 3 (Executor)", agent3_output))

            st.success("Multi-agent vector workflow executed successfully!")

# Tab 3: Workflow Logs & Results
with tab3:
    st.header("Agent Execution Results")
    if not st.session_state.agent_logs:
        st.info("No workflow has been executed yet. Go to the Multi-Agent Executor tab to start.")
    else:
        for agent_name, output in st.session_state.agent_logs:
            st.subheader(f"📌 {agent_name}")
            st.markdown(output)
            st.markdown("---")
