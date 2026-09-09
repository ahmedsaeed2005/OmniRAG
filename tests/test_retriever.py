from src.ingestion.pdf_loader import load_pdf
from src.chunking.chunker import create_chunks
from src.embeddings.embedder import create_embeddings
from src.retrieval.vector_store import create_vector_store
from src.retrieval.retriever import retrieve_chunks


# 1. Load PDF
pages = load_pdf("data/sample.pdf")


# 2. Create chunks
chunks = create_chunks(pages)


# 3. Create embeddings
texts = [chunk["text"] for chunk in chunks]

embeddings = create_embeddings(texts)


# 4. Create FAISS
index = create_vector_store(embeddings)


# 5. Query
query = "What is Big Data?"

query_embedding = create_embeddings([query])


# 6. Retrieve
results = retrieve_chunks(
    index,
    chunks,
    query_embedding,
    k=3
)


# 7. Print results
for rank, result in enumerate(results, start=1):

    print("\n====================")

    print("Rank:", rank)
    print("Page:", result["metadata"]["page"])
    print("Chunk:", result["metadata"]["chunk_id"])
    print("Distance:", result["distance"])

    print("Text:")
    print(result["text"])