from src.ingestion.pdf_loader import load_pdf
from src.chunking.chunker import create_chunks
from src.retrieval.keyword import (
    create_keyword_index,
    search_keyword
)


# =========================
# 1. Load PDF
# =========================

pages = load_pdf("data/sample.pdf")


# =========================
# 2. Create Chunks
# =========================

chunks = create_chunks(pages)


# =========================
# 3. Create BM25 Index
# =========================

bm25 = create_keyword_index(chunks)


# =========================
# 4. Search
# =========================

query = "What is Big Data?"

results = search_keyword(
    bm25,
    chunks,
    query,
    k=3
)


# =========================
# 5. Print Results
# =========================

for rank, result in enumerate(results, start=1):

    print("\n====================")

    print("Rank:", rank)
    print("Page:", result["metadata"]["page"])
    print("Chunk:", result["metadata"]["chunk_id"])
    print("Score:", result["score"])

    print("Text:")
    print(result["text"])