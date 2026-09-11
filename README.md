# EliteBotStudios - ChromaDB Multi-Document RAG Platform

This updated EliteBotStudios architecture integrates ChromaDB vector database storage with Google Gemini embeddings (`text-embedding-004`) to handle semantic retrieval across multi-document knowledge bases.

## New Features Included
- **ChromaDB Vector Store**: Automatically embeds and indexes uploaded `.txt` and `.md` document chunks.
- **Semantic Vector Retrieval**: Performs top-k similarity queries using Google Gemini embeddings before passing context to Agent 1.
- **Multi-Agent Orchestration**: Agent 1 (Researcher) -> Agent 2 (Strategy Planner) -> Agent 3 (Executor).

## Setup & Deployment Instructions

1. Extract the downloaded zip file:
   ```bash
   unzip elitebotstudios_chromadb_rag.zip
   cd elitebotstudios_chromadb_rag
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

4. Upload documents in Tab 1, enter your Google Gemini API key in the sidebar, and execute multi-agent semantic tasks in Tab 2!
