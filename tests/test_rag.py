from src.ingestion.pdf_loader import load_pdf
from src.chunking.chunker import create_chunks
from src.embeddings.embedder import create_embeddings
from src.retrieval.vector_store import create_vector_store
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.llm.gemini import create_llm


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

index = create_vector_store(
    embeddings
)

print("FAISS vectors:", index.ntotal)


# =========================
# 5. Create Retrieval Pipeline
# =========================

retrieval = RetrievalPipeline(
    chunks,
    index
)


# =========================
# 6. Ask Question
# =========================

question = "What are the logical layers in a data processing architecture?"

# =========================
# 7. Retrieve
# =========================

results = retrieval.retrieve(
    question,
    final_k=3
)


# =========================
# 8. Build Context
# =========================

context_parts = []

for result in results:

    page = result["metadata"]["page"]
    text = result["text"]

    context_parts.append(
        f"[Source: Page {page}]\n{text}"
    )

context = "\n\n".join(
    context_parts
)


# =========================
# 9. Create Gemini
# =========================

llm = create_llm()


# =========================
# 10. Prompt
# =========================

prompt = f"""
You are a helpful assistant answering questions
using only the provided document context.

Context:
{context}

Question:
{question}

Instructions:

- Answer only from the provided context.
- Do not make up information.
- Mention the page number when using information.
- If the answer is not available in the context,
  say that the information is not available
  in the document.
- Give a clear and concise answer.
"""


# =========================
# 11. Generate Answer
# =========================

response = llm.invoke(
    prompt
)


# =========================
# 12. Extract Answer
# =========================

if isinstance(response.content, list):

    answer = ""

    for item in response.content:

        if item.get("type") == "text":

            answer += item.get(
                "text",
                ""
            )

else:

    answer = response.content


# =========================
# 13. Print Answer
# =========================

print("\n==============================")
print("FINAL ANSWER")
print("==============================")

print(answer)


# =========================
# 14. Print Sources
# =========================

print("\n==============================")
print("SOURCES")
print("==============================")

for result in results:

    page = result["metadata"]["page"]
    chunk_id = result["metadata"]["chunk_id"]

    print(
        f"- Page {page}, Chunk {chunk_id}"
    )