from src.ingestion.pdf_loader import load_pdf
from src.chunking.chunker import create_chunks
from src.embeddings.embedder import create_embeddings
from src.retrieval.vector_store import create_vector_store
from src.retrieval.retriever import retrieve_chunks
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

texts = [chunk["text"] for chunk in chunks]

embeddings = create_embeddings(texts)

print("Embeddings:", embeddings.shape)


# =========================
# 4. Create FAISS
# =========================

index = create_vector_store(embeddings)

print("FAISS vectors:", index.ntotal)



# =========================
# 5. Ask Question
# =========================

question = "What is Big Data?"

query_embedding = create_embeddings([question])


# =========================
# 6. Retrieve Relevant Chunks
# =========================

results = retrieve_chunks(
    index,
    chunks,
    query_embedding,
    k=3
)


print("\nRetrieved Context:")

for result in results:
    print("\n--------------------")
    print("Page:", result["metadata"]["page"])
    print("Text:")
    print(result["text"])


# =========================
# 7. Build Context
# =========================

context_parts = []

for result in results:

    page = result["metadata"]["page"]
    text = result["text"]

    context_parts.append(
        f"[Source: Page {page}]\n{text}"
    )


context = "\n\n".join(context_parts)

# =========================
# 8. Create LLM
# =========================

llm = create_llm()


# =========================
# 9. Create Prompt
# =========================

prompt = f"""
You are a helpful assistant answering questions based only on the provided context.

Context:
{context}

Question:
{question}

Instructions:
- Answer using only the provided context.
- Do not make up information.
- When using information from a source, mention its page number.
- If the answer is not available in the context, say that the information is not available in the document.
- Give a clear and concise answer.
"""


# =========================
# 10. Generate Answer
# =========================
response = llm.invoke(prompt)


# =========================
# 11. Extract Answer
# =========================

if isinstance(response.content, list):

    answer = ""

    for item in response.content:

        if item.get("type") == "text":
            answer += item.get("text", "")

else:

    answer = response.content


# =========================
# 12. Print Final Answer
# =========================

print("\n====================")
print("Final Answer:")
print(answer)


# =========================
# 13. Print Sources
# =========================

print("\nSources:")

for result in results:

    page = result["metadata"]["page"]

    print(f"- Page {page}")