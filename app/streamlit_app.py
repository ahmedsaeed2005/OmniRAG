import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import hashlib
import streamlit as st

from src.ingestion.pdf_loader import load_pdf
from src.chunking.chunker import create_chunks
from src.embeddings.embedder import create_embeddings
from src.retrieval.vector_store import create_vector_store
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.graph.workflow import build_graph

# =========================================
# Page Config
# =========================================

st.set_page_config(
    page_title="OmniRAG",
    page_icon="🤖",
    layout="wide"
)


# =========================================
# Title
# =========================================

st.title("🤖 OmniRAG")

st.markdown(
    """
### Multimodal Agentic RAG System

Upload a PDF, ask questions, and let the system:

**Retrieve → Rerank → Evaluate → Answer**
"""
)


# =========================================
# Sidebar
# =========================================

st.sidebar.header("⚙️ Settings")

use_vision = st.sidebar.checkbox(
    "Enable Vision",
    value=False
)

st.sidebar.info(
    "Vision uses Gemini to analyze images inside the PDF."
)


# =========================================
# Build Retriever
# =========================================

@st.cache_resource
def build_retriever(file_bytes, file_hash, vision):

    os.makedirs(
        "data/uploads",
        exist_ok=True
    )

    file_path = os.path.join(
        "data/uploads",
        f"{file_hash}.pdf"
    )

    with open(
        file_path,
        "wb"
    ) as f:
        f.write(file_bytes)

    # Load PDF
    pages = load_pdf(
        file_path,
        use_vision=vision
    )

    # Create chunks
    chunks = create_chunks(
        pages
    )

    # Embeddings
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = create_embeddings(
        texts
    )

    # Vector store
    index = create_vector_store(
        embeddings
    )

    # Retrieval pipeline
    retriever = RetrievalPipeline(
        chunks,
        index
    )

    return retriever, len(pages), len(chunks)


# =========================================
# Upload PDF
# =========================================

uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)


if uploaded_file:

    file_bytes = uploaded_file.getvalue()

    file_hash = hashlib.md5(
        file_bytes
    ).hexdigest()

    with st.spinner(
        "Processing document..."
    ):

        retriever, pages_count, chunks_count = build_retriever(
            file_bytes,
            file_hash,
            use_vision
        )

    st.success(
        f"Document processed successfully! "
        f"Pages: {pages_count} | Chunks: {chunks_count}"
    )


    # =====================================
    # Question
    # =====================================

    question = st.text_input(
        "💬 Ask a question about the document"
    )


    if question:

        graph = build_graph()

        with st.spinner(
            "🤖 Agent is thinking..."
        ):

            result = graph.invoke({

                "question": question,

                "retriever": retriever,

                "results": [],

                "answer": "",

                "retry_count": 0

            })


        # =================================
        # Answer
        # =================================

        st.subheader("🧠 Answer")

        st.write(
            result["answer"]
        )


        # =================================
        # Sources
        # =================================

        st.subheader("📚 Sources")

        for i, source in enumerate(
            result["results"],
            start=1
        ):

            page = source["metadata"]["page"]

            score = source.get(
                "rerank_score",
                0
            )

            with st.expander(
                f"Source {i} — Page {page}"
            ):

                st.write(
                    f"**Rerank Score:** {score:.4f}"
                )

                st.write(
                    source["text"]
                )

else:

    st.info(
        "👆 Upload a PDF to start using OmniRAG."
    )