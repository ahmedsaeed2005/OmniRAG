from src.ingestion.pdf_loader import load_pdf
from src.chunking.chunker import create_chunks
from src.embeddings.embedder import create_embeddings
from src.retrieval.vector_store import create_vector_store
from src.retrieval.retrieval_pipeline import RetrievalPipeline

from src.graph.workflow import build_graph


# =========================
# 1. Load PDF
# =========================

pages = load_pdf(
    "data/sample.pdf",
    use_vision=False
)

print("Pages:", len(pages))


# =========================
# 2. Create chunks
# =========================

chunks = create_chunks(pages)

print("Chunks:", len(chunks))


# =========================
# 3. Create embeddings
# =========================

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = create_embeddings(texts)

print("Embeddings:", embeddings.shape)


# =========================
# 4. Create vector store
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


# =========================
# 6. Build Agentic RAG
# =========================

graph = build_graph()


# =========================
# 7. Ask Question
# =========================

question = "What are the five logical layers?"


result = graph.invoke({

    "question": question,

    "retriever": retriever,

    "results": [],

    "answer": "",

    "retry_count": 0

})

# =========================
# 8. Print Result
# =========================

print("\n==============================")
print("AGENTIC RAG RESULT")
print("==============================\n")

print(result["answer"])

print("\n==============================")
print("SOURCES")
print("==============================\n")

for source in result["results"]:

    print(
        f"Page: {source['metadata']['page']}"
    )

    print(
        f"Score: {source['rerank_score']}"
    )

    print(
        source["text"][:300]
    )

    print("------------------------------")