# 💬 Chat with your PDFs

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload one or more PDF 
documents and ask questions about them in natural language — with every answer traced 
back to the exact source document, page, and text snippet it came from.

## ✨ Features

- **Multi-PDF upload & chat** — index multiple documents and query them conversationally
- **Streaming responses** — answers stream token-by-token instead of appearing all at once
- **Source attribution** — every answer shows which document(s), page(s), and chunks it was grounded in
- **Incremental indexing** — new uploads are embedded and added to the existing knowledge base without re-processing previously indexed files
- **Session isolation** — each user session gets its own storage folder and conversation thread, so multiple users never see each other's documents or chat history
- **Tool-calling agent architecture** — built on LangGraph's agent framework, with a dedicated retrieval tool the LLM invokes on demand

## 🛠️ Tech Stack

- **Frontend/UI:** Streamlit
- **Orchestration:** LangChain, LangGraph (`create_agent`, checkpointed memory)
- **LLM:** Groq (`openai/gpt-oss-20b`)
- **Embeddings:** OpenAI (`text-embedding-3-small`)
- **Vector Store:** LangChain `InMemoryVectorStore`
- **Document Loading:** PyPDF

## 🚀 How it works

1. Upload PDF(s) → text is extracted, chunked, and embedded into a per-session vector store
2. Ask a question → a LangGraph agent decides whether to call the `retrieve_context` tool
3. Relevant chunks are retrieved via similarity search and passed as context to the LLM
4. The answer streams back in real time, with an expandable **"Sources used"** panel showing exactly which document/page backed the response
