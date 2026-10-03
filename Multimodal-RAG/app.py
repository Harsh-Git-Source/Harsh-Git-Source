from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src.pipeline.rag_pipeline import MultimodalRAGPipeline

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem;}
.hero {
    padding: 1.5rem 1.8rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, #1f2937);
    color: white;
    margin-bottom: 1.2rem;
}
.hero h1 {margin-bottom: .2rem;}
.card {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: .8rem;
}
.small {opacity: .72; font-size: .9rem;}
</style>
""",
    unsafe_allow_html=True,
)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "stats" not in st.session_state:
    st.session_state.stats = None
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown(
    """
<div class="hero">
  <h1>📚 DocuMind AI</h1>
  <div>Multimodal Document Intelligence & RAG</div>
  <div class="small">Ask questions across PDF text, tables and images.</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Configuration")
    st.caption(f"Embedding: `{settings.embedding_model}`")
    st.caption(f"Groq: `{settings.groq_model}`")
    st.caption(f"Gemini: `{settings.gemini_model}`")

    missing = []
    if not settings.groq_api_key:
        missing.append("GROQ_API_KEY")
    if not settings.gemini_api_key:
        missing.append("GEMINI_API_KEY")

    if missing:
        st.warning("Missing: " + ", ".join(missing))
        st.caption("Copy `.env.example` to `.env` and add your API keys.")

tab_upload, tab_chat, tab_sources = st.tabs(
    ["📤 Upload & Index", "💬 Ask Documents", "🔎 Retrieval Details"]
)

with tab_upload:
    st.subheader("Upload documents")
    uploaded = st.file_uploader(
        "Upload one or more PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        st.info(f"{len(uploaded)} document(s) selected.")

    if st.button("🚀 Process Documents", type="primary", disabled=not uploaded):
        if not settings.groq_api_key or not settings.gemini_api_key:
            st.error("Add both GROQ_API_KEY and GEMINI_API_KEY before processing.")
        else:
            paths = []
            progress = st.progress(0)
            status = st.empty()

            for i, file in enumerate(uploaded):
                path = settings.upload_dir / file.name
                path.write_bytes(file.getbuffer())
                paths.append(str(path))
                progress.progress((i + 1) / len(uploaded))
                status.write(f"Saved `{file.name}`")

            try:
                with st.spinner("Extracting, summarizing and indexing..."):
                    pipeline = MultimodalRAGPipeline()
                    stats = pipeline.ingest(paths)
                st.session_state.pipeline = pipeline
                st.session_state.stats = stats
                st.session_state.history = []
                st.success("Documents indexed successfully.")
            except Exception as exc:
                st.exception(exc)

    if st.session_state.stats:
        st.subheader("Document insights")
        stats = st.session_state.stats
        cols = st.columns(4)
        cols[0].metric("Documents", stats["documents"])
        cols[1].metric("Text", stats["text"])
        cols[2].metric("Tables", stats["table"])
        cols[3].metric("Images", stats["image"])
        st.caption(f"Indexed retrieval items: {stats.get('indexed_items', 0)}")

with tab_chat:
    st.subheader("Ask your documents")
    if st.session_state.pipeline is None:
        st.info("Upload and process a PDF first.")
    else:
        for turn in st.session_state.history:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                st.write(turn["answer"])

        question = st.chat_input(
            "Ask something about the uploaded documents..."
        )
        if question:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Retrieving multimodal evidence..."):
                    try:
                        result = st.session_state.pipeline.ask(question)
                        st.write(result["answer"])
                        with st.expander("Sources"):
                            for source in result["sources"]:
                                st.write(
                                    f"**{source['source']}** | "
                                    f"page {source['page']} | "
                                    f"{source['modality']}"
                                )
                        st.session_state.history.append(
                            {
                                "question": question,
                                "answer": result["answer"],
                                "result": result,
                            }
                        )
                    except Exception as exc:
                        st.exception(exc)

with tab_sources:
    st.subheader("Latest retrieval")
    if not st.session_state.history:
        st.info("Ask a question to inspect retrieved evidence.")
    else:
        result = st.session_state.history[-1]["result"]
        for source in result["sources"]:
            st.markdown(
                f"- **{source['source']}** — page {source['page']} — "
                f"`{source['modality']}`"
            )

        if result["retrieved_text"]:
            st.markdown("### Retrieved text / tables")
            for idx, text in enumerate(result["retrieved_text"], start=1):
                with st.expander(f"Context {idx}"):
                    st.write(text)

        if result["retrieved_images"]:
            st.markdown("### Retrieved images")
            cols = st.columns(min(3, len(result["retrieved_images"])))
            for idx, image_path in enumerate(result["retrieved_images"]):
                with cols[idx % len(cols)]:
                    st.image(image_path, caption=Path(image_path).name)
