from src.ingestion.pdf_loader import load_pdf
from src.chunking.chunker import create_chunks
from src.embeddings.embedder import create_embeddings
from src.retrieval.vector_store import create_vector_store
from src.retrieval.retrieval_pipeline import RetrievalPipeline


# =========================
# 1. Load PDF
# =========================

pages = load_pdf(
    "data/sample.pdf",
    use_vision=False
)

print("Pages:", len(pages))


# =========================
# 2. Chunks
# =========================

chunks = create_chunks(pages)

print("Chunks:", len(chunks))


# =========================
# 3. Embeddings
# =========================

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = create_embeddings(
    texts
)

print("Embeddings:", embeddings.shape)


# =========================
# 4. Vector Store
# =========================

index = create_vector_store(
    embeddings
)


# =========================
# 5. Retrieval Pipeline
# =========================

retriever = RetrievalPipeline(
    chunks,
    index
)


# =========================
# 6. Question
# =========================

question = (
    "What are the five logical layers "
    "in the data processing architecture?"
)


# =========================
# 7. Retrieve
# =========================

results = retriever.retrieve(
    question,
    semantic_k=5,
    keyword_k=5,
    final_k=3
)


# =========================
# 8. Print Results
# =========================

print("\n==============================")
print("RETRIEVAL RESULTS")
print("==============================\n")


for i, result in enumerate(
    results,
    start=1
):

    print(f"Result {i}")

    print(
        f"Page: {result['metadata']['page']}"
    )

    print(
        f"Chunk: {result['metadata']['chunk_id']}"
    )

    print(
        f"Score: {result['rerank_score']}"
    )

    print(
        result["text"]
    )

    print(
        "\n------------------------------\n"
    )