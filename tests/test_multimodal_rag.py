import fitz

from src.chunking.chunker import create_chunks
from src.embeddings.embedder import create_embeddings
from src.retrieval.vector_store import create_vector_store
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.ingestion.document_parser import extract_page_content
from src.llm.gemini import create_llm


# =========================
# 1. Open PDF
# =========================

pdf = fitz.open("data/sample.pdf")

# Page 9
page = pdf[9]


# =========================
# 2. Extract Page Content
# =========================

content = extract_page_content(
    page,
    use_vision=True
)

pdf.close()


# =========================
# 3. Create Page Object
# =========================

pages = [
    {
        "page": 9,
        "text": content["text"],
        "images": content["images"]
    }
]


# =========================
# 4. Create Chunks
# =========================

chunks = create_chunks(pages)

print("\nChunks:", len(chunks))


# =========================
# 5. Create Embeddings
# =========================

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = create_embeddings(texts)

print(
    "Embeddings:",
    embeddings.shape
)


# =========================
# 6. Create FAISS
# =========================

index = create_vector_store(
    embeddings
)


# =========================
# 7. Create Retrieval Pipeline
# =========================

pipeline = RetrievalPipeline(
    chunks,
    index
)


# =========================
# 8. User Question
# =========================

question = (
    "What are the five logical layers "
    "in the data processing architecture?"
)


# =========================
# 9. Retrieve Relevant Chunks
# =========================

results = pipeline.retrieve(
    question,
    semantic_k=5,
    keyword_k=5,
    final_k=3
)


print("\n==============================")
print("RETRIEVED CONTEXT")
print("==============================\n")


for i, result in enumerate(
    results,
    start=1
):

    print(f"Result {i}")
    print(
        "Page:",
        result["metadata"]["page"]
    )
    print(
        "Chunk:",
        result["metadata"]["chunk_id"]
    )

    print(
        result["text"]
    )

    print(
        "\n------------------------------\n"
    )


# =========================
# 10. Build Context
# =========================

context_parts = []

for result in results:

    page = result["metadata"]["page"]
    chunk_id = result["metadata"]["chunk_id"]

    text = result["text"]

    context_parts.append(
        f"[Page {page}, Chunk {chunk_id}]\n"
        f"{text}"
    )


context = "\n\n".join(
    context_parts
)


# =========================
# 11. Create Gemini
# =========================

llm = create_llm()


# =========================
# 12. Build RAG Prompt
# =========================

prompt = f"""
You are an AI assistant answering questions
using retrieved information from a document.

Answer the user's question using ONLY the
provided context.

If the answer is not available in the context,
say that the information is not available.

Do not invent information.

User Question:
{question}

Retrieved Context:
{context}

Instructions:
- Give a clear and concise answer.
- Explain the five logical layers.
- Preserve the correct order from Layer 1 to Layer 5.
- Mention the page number where the information came from.
"""


# =========================
# 13. Generate Final Answer
# =========================

response = llm.invoke(prompt)


# =========================
# 14. Print Final Answer
# =========================

print("\n==============================")
print("FINAL ANSWER")
print("==============================\n")

content = response.content

if isinstance(content, list):

    text_parts = []

    for item in content:

        if (
            isinstance(item, dict)
            and item.get("type") == "text"
        ):

            text_parts.append(
                item.get("text", "")
            )

    final_answer = "\n".join(
        text_parts
    ).strip()

else:

    final_answer = str(content).strip()


print(final_answer)