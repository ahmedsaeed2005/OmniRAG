from langchain_google_genai import ChatGoogleGenerativeAI


def create_llm():

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0
    )

    return llm


def generate_answer(question, results):

    llm = create_llm()

    context_parts = []

    for result in results:

        page = result["metadata"]["page"]
        text = result["text"]

        context_parts.append(
            f"[Page {page}]\n{text}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided context.

If the answer is not available in the context,
say that the information is not available in the document.

Always mention the page number(s) used.

Question:
{question}

Context:
{context}

Answer:
"""

    response = llm.invoke(prompt)

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

        return "\n".join(
            text_parts
        ).strip()

    return str(content).strip()