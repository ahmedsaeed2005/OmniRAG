from src.ingestion.pdf_loader import load_pdf
from src.chunking.chunker import create_chunks
from src.embeddings.embedder import create_embeddings

from src.retrieval.vector_store import create_vector_store
from src.retrieval.retrieval_pipeline import RetrievalPipeline


# =========================
# 1. Load PDF
# =========================

pages = load_pdf(
    "data/sample.pdf"
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
# 4. Create FAISS
# =========================

index = create_vector_store(
    embeddings
)


# =========================
# 5. Create Retrieval Pipeline
# =========================

pipeline = RetrievalPipeline(
    chunks,
    index
)


# =========================
# 6. Ask Question
# =========================

questions = [
    "What is Big Data?",
    "Explain the 3 V's on page 8"
]


for question in questions:

    print("\n\n================================")
    print("QUESTION:", question)
    print("================================")

    results = pipeline.retrieve(
        question,
        final_k=3
    )

    for rank, result in enumerate(
        results,
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
            "Rerank Score:",
            result["rerank_score"]
        )

        print("\nText:")

        print(result["text"])

# =========================
# 7. Retrieve
# =========================

results = pipeline.retrieve(
    question
)


# =========================
# 8. Print Results
# =========================

print("\n==============================")
print("FINAL RETRIEVAL RESULTS")
print("==============================")


for rank, result in enumerate(
    results,
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
        "Rerank Score:",
        result["rerank_score"]
    )

    print("\nText:")

    print(result["text"])