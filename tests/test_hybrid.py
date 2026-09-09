from src.ingestion.pdf_loader import load_pdf
from src.chunking.chunker import create_chunks

from src.embeddings.embedder import create_embeddings

from src.retrieval.vector_store import create_vector_store
from src.retrieval.retriever import retrieve_chunks

from src.retrieval.keyword import (
    create_keyword_index,
    search_keyword
)

from src.retrieval.hybrid import hybrid_search


# =========================
# 1. Load PDF
# =========================

pages = load_pdf("data/sample.pdf")

print("Pages:", len(pages))


# =========================
# 2. Create Chunks
# =========================

chunks = create_chunks(pages)

print("Chunks:", len(chunks))


# =========================
# 3. Create Embeddings
# =========================

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = create_embeddings(texts)

print("Embeddings:", embeddings.shape)


# =========================
# 4. Create FAISS
# =========================

index = create_vector_store(embeddings)

print("FAISS vectors:", index.ntotal)


# =========================
# 5. Create BM25
# =========================

bm25 = create_keyword_index(chunks)


# =========================
# 6. Ask Question
# =========================

question = "What is Big Data?"


# =========================
# 7. Semantic Search
# =========================

query_embedding = create_embeddings(
    [question]
)

semantic_results = retrieve_chunks(
    index,
    chunks,
    query_embedding,
    k=5
)


# =========================
# 8. Keyword Search
# =========================

keyword_results = search_keyword(
    bm25,
    chunks,
    question,
    k=5
)


# =========================
# 9. Hybrid Search
# =========================

hybrid_results = hybrid_search(
    semantic_results,
    keyword_results,
    k=5
)


# =========================
# 10. Print Results
# =========================

print("\n==============================")
print("HYBRID SEARCH RESULTS")
print("==============================")


for rank, result in enumerate(
    hybrid_results,
    start=1
):

    print("\n--------------------")

    print("Rank:", rank)

    print(
        "Page:",
        result["metadata"]["page"]
    )

    print(
        "Chunk:",
        result["metadata"]["chunk_id"]
    )

    print(
        "Hybrid Score:",
        result["hybrid_score"]
    )

    print("Text:")

    print(result["text"])