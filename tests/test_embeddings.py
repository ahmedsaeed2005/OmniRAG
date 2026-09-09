from src.ingestion.pdf_loader import load_pdf
from src.chunking.chunker import create_chunks
from src.embeddings.embedder import create_embeddings
from src.retrieval.vector_store import (
    create_vector_store,
    search_vector_store
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
# 3. Extract Text
# =========================

texts = [
    chunk["text"]
    for chunk in chunks
]


# =========================
# 4. Create Embeddings
# =========================

embeddings = create_embeddings(texts)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)


# =========================
# 5. Create FAISS
# =========================

index = create_vector_store(embeddings)

print("FAISS vectors:", index.ntotal)


# =========================
# 6. Test Query
# =========================

query = "What is Big Data?"

query_embedding = create_embeddings([query])[0]

distances, indices = search_vector_store(
    index,
    query_embedding,
    k=3
)

print("\nTop Results:")

for rank, idx in enumerate(indices[0], start=1):

    print("\n====================")

    print("Rank:", rank)

    print(
        "Page:",
        chunks[idx]["metadata"]["page"]
    )

    print(
        "Chunk:",
        chunks[idx]["metadata"]["chunk_id"]
    )

    print(
        "Distance:",
        distances[0][rank - 1]
    )

    print(
        "Text:",
        chunks[idx]["text"][:300]
    )