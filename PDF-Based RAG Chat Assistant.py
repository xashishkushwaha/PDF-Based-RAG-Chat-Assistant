"""
PDF Q&A Chat App
----------------
Upload one or more PDFs, build a vector store + RAG agent, then chat with it.
Responses stream token-by-token, sources are shown under each answer, and
each browser session gets its own isolated file storage + vector store.
"""

import os
import shutil
import uuid

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from dataclasses import dataclass, field
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


# ======================================================================
# CONFIG
# ======================================================================
BASE_DOCS_PATH = "./docs_files/"

st.set_page_config(
    page_title="Chat with your PDFs",
    page_icon="📄",
    layout="centered",
)


# ======================================================================
# RAG STATE CONTAINER
# ======================================================================
# LangGraph executes tool calls in a background task/thread that does NOT
# carry Streamlit's ScriptRunContext. That means `st.session_state.xxx`
# attribute access raises AttributeError if used *inside* a @tool function.
#
# Fix: keep everything the tool needs to read/write in a plain Python
# object instead. Plain objects need no Streamlit context — only the
# special st.session_state accessor does. We still store this object as a
# single entry *inside* st.session_state so it survives reruns; the tool
# just closes over the object itself rather than reaching into
# st.session_state each time it runs.
@dataclass
class RAGState:
    vector_store: object = None
    last_sources: list = field(default_factory=list)


# ======================================================================
# SESSION STATE
# ======================================================================
def init_session_state():
    """
    Create all session_state keys the app relies on, if not already present.
    Each browser session gets a unique `session_id`, used both as its own
    upload folder (session isolation) and as the LangGraph checkpoint
    thread_id (so conversations from different users never mix).
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex

    if "rag_state" not in st.session_state:
        st.session_state.rag_state = RAGState()

    defaults = {
        "document_uploaded": False,
        "agent": None,
        "messages": [],        # chat history: [{"role": "user"/"ai", "content": str}]
        "uploaded_names": [],  # names of files currently indexed, for the sidebar
        "processed_names": set(),  # filenames already embedded, to avoid re-embedding
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# Every session gets its own folder on disk -> no cross-session file mixing.
SESSION_DOCS_PATH = os.path.join(BASE_DOCS_PATH, st.session_state.session_id)
THREAD_ID = st.session_state.session_id


# ======================================================================
# DOCUMENT PROCESSING
# ======================================================================
def load_and_split_new_files(files, path: str):
    """
    Load ONLY the given files (not the whole folder) and split into chunks.
    Skips any file whose name is already in session_state.processed_names,
    so re-running process on an existing session never re-embeds old files.
    Returns (chunks, empty_files) where empty_files is a list of filenames
    that produced no extractable text (e.g. scanned/image-only PDFs).
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    all_chunks = []
    empty_files = []

    for file in files:
        if file.name in st.session_state.processed_names:
            continue  # already embedded in an earlier upload batch

        file_path = os.path.join(path, file.name)
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Detect PDFs with no extractable text (scanned/image-only pages)
        text_length = sum(len(d.page_content.strip()) for d in docs)
        if text_length == 0:
            empty_files.append(file.name)
            continue

        chunks = splitter.split_documents(documents=docs)
        all_chunks.extend(chunks)
        st.session_state.processed_names.add(file.name)

    return all_chunks, empty_files


def build_or_extend_vector_store(chunks):
    """
    Embed `chunks` and either create a new vector store (first upload) or
    add to the existing one in-place (subsequent uploads). The retrieval
    tool closes over `rag_state` directly and reads `rag_state.vector_store`
    at call time, so growing this store here is instantly visible to the
    agent with no need to rebuild it or touch st.session_state from a
    background thread.
    """
    rag_state = st.session_state.rag_state
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if rag_state.vector_store is None:
        rag_state.vector_store = InMemoryVectorStore.from_documents(
            documents=chunks, embedding=embeddings
        )
    else:
        rag_state.vector_store.add_documents(chunks)


def create_agent_once():
    """
    Build the retrieval tool + agent a single time per session. The tool
    closes over the `rag_state` object directly (captured below, NOT via
    st.session_state) so it works even when LangGraph runs it outside
    Streamlit's script-run context. `rag_state.vector_store` is read fresh
    on every call, so the tool automatically sees documents added in later
    upload batches without the agent needing to be rebuilt.
    """
    if st.session_state.agent is not None:
        return  # already built for this session

    rag_state = st.session_state.rag_state  # plain object, safe to close over
    
    # llm = ChatGroq(model="openai/gpt-oss-20b")
    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07")
    

    @tool
    def retrieve_context(query: str):
        """Retrieve document chunks relevant to a query from the vector database."""
        print("Tool called with:", query)
        store = rag_state.vector_store
        if store is None:
            return "No documents have been indexed yet."

        results = store.similarity_search(query=query, k=4)

        # Save the retrieved chunks (on the plain object, not st.session_state)
        # so the main script can show them as "Sources" after the tool returns.
        rag_state.last_sources = [
            {
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "page": doc.metadata.get("page", "?"),
                "snippet": doc.page_content[:300],
            }
            for doc in results
        ]

        return "\n\n".join(doc.page_content for doc in results)

    memory = InMemorySaver()

    system_prompt = """You are a helpful assistant that answers questions using retrieved context.
Your knowledge base consists of the details from the uploaded document(s).
ALWAYS use the `retrieve_context` tool for questions requiring information from the documents."""

    st.session_state.agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory,
    )


def process_uploaded_files(files, path: str):
    """
    Full pipeline for a batch of uploaded files: save to disk, load + split
    only the new ones, embed them, and (re)create the agent if needed.
    Returns the list of filenames that had no extractable text, so the
    caller can warn the user.
    """
    os.makedirs(path, exist_ok=True)

    for file in files:
        with open(os.path.join(path, file.name), "wb") as f:
            f.write(file.getvalue())

    chunks, empty_files = load_and_split_new_files(files, path)

    if chunks:
        build_or_extend_vector_store(chunks)

    create_agent_once()

    st.session_state.document_uploaded = st.session_state.rag_state.vector_store is not None
    st.session_state.uploaded_names = sorted(st.session_state.processed_names)

    return empty_files


def reset_session():
    """Delete this session's files on disk and clear all related state."""
    if os.path.exists(SESSION_DOCS_PATH):
        shutil.rmtree(SESSION_DOCS_PATH)

    st.session_state.document_uploaded = False
    st.session_state.agent = None
    st.session_state.rag_state = RAGState()  # fresh vector_store + last_sources
    st.session_state.messages = []
    st.session_state.uploaded_names = []
    st.session_state.processed_names = set()


# ======================================================================
# STREAMING HELPER
# ======================================================================
def stream_agent_response(agent, query: str, thread_id: str):
    """
    Generator that yields text chunks as the agent produces them, so it can
    be passed directly to st.write_stream(). Tool-call tokens and tool node
    output are filtered out - only the final assistant text is streamed.
    """
    for token, metadata in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": thread_id}},
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") == "tools":
            continue
        if getattr(token, "tool_calls", None):
            continue
        if token.content:
            yield token.content


# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.header("📄 Documents")

    if st.session_state.document_uploaded:
        st.success(f"{len(st.session_state.uploaded_names)} file(s) indexed")
        for name in st.session_state.uploaded_names:
            st.caption(f"• {name}")

        st.divider()
        more_files = st.file_uploader(
            "Add more PDFs to this session",
            type=["pdf"],
            accept_multiple_files=True,
            key="more_files_uploader",
        )
        if more_files:
            with st.spinner("Embedding new file(s)..."):
                empty = process_uploaded_files(more_files, SESSION_DOCS_PATH)
            if empty:
                st.warning(
                    "No extractable text found in: " + ", ".join(empty)
                    + " (likely scanned/image-only PDFs)."
                )
            st.rerun()

        st.divider()
        if st.button("🔄 Start over with new files", use_container_width=True):
            reset_session()
            st.rerun()
    else:
        st.info("No documents indexed yet.")


# ======================================================================
# MAIN — UPLOAD UI (first-time, before any documents exist)
# ======================================================================
st.title("💬 Chat with your PDFs")

if not st.session_state.document_uploaded:
    st.markdown("Upload one or more PDF files to get started.")

    uploaded = st.file_uploader(
        label="Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        with st.spinner("Reading, chunking, and embedding your documents..."):
            empty_files = process_uploaded_files(uploaded, SESSION_DOCS_PATH)

        if not st.session_state.document_uploaded:
            # Every uploaded file was empty/unreadable
            st.error(
                "No extractable text found in any uploaded file: "
                + ", ".join(empty_files)
                + ". Try a text-based PDF, or an OCR'd version of a scanned one."
            )
        else:
            if empty_files:
                st.warning(
                    "No extractable text found in: " + ", ".join(empty_files)
                    + " — the rest were indexed successfully."
                )
            st.rerun()


# ======================================================================
# MAIN — CHAT UI
# ======================================================================
if st.session_state.document_uploaded and st.session_state.agent:

    # Replay existing chat history on every rerun
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    query = st.chat_input("Ask anything about your uploaded documents...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        st.session_state.rag_state.last_sources = []  # cleared, tool will refill if called

        with st.chat_message("ai"):
            full_response = st.write_stream(
                stream_agent_response(st.session_state.agent, query, THREAD_ID)
            )

            # Show the chunks that were actually retrieved for this answer
            if st.session_state.rag_state.last_sources:
                with st.expander("📚 Sources used"):
                    for i, src in enumerate(st.session_state.rag_state.last_sources, start=1):
                        st.markdown(
                            f"**{i}. {src['source']}** (page {src['page']})"
                        )
                        st.caption(src["snippet"] + "...")

        st.session_state.messages.append({"role": "ai", "content": full_response})