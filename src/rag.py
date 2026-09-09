from src.ingestion.pdf_loader import load_pdf
from src.chunking.chunker import create_chunks
from src.embeddings.embedder import create_embeddings
from src.retrieval.vector_store import create_vector_store
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.llm.gemini import generate_answer


def build_rag(
    file_path,
    use_vision=False
):

    # =========================
    # 1. Load Document
    # =========================

    pages = load_pdf(
        file_path,
        use_vision=use_vision
    )

    # =========================
    # 2. Create Chunks
    # =========================

    chunks = create_chunks(
        pages
    )

    # =========================
    # 3. Create Embeddings
    # =========================

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = create_embeddings(
        texts
    )

    # =========================
    # 4. Create Vector Store
    # =========================

    index = create_vector_store(
        embeddings
    )

    # =========================
    # 5. Create Retrieval Pipeline
    # =========================

    retriever = RetrievalPipeline(
        chunks,
        index
    )

    return retriever


def ask(
    retriever,
    question
):

    # =========================
    # 1. Retrieve
    # =========================

    results = retriever.retrieve(
        question
    )

    # =========================
    # 2. Generate Answer
    # =========================

    answer = generate_answer(
        question,
        results
    )

    return {
        "answer": answer,
        "sources": results
    }