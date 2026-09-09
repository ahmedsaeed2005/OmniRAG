from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk_id, chunk in enumerate(page_chunks):
            chunks.append({
                "text": chunk,
                "metadata": {
                    "page": page["page"],
                    "chunk_id": chunk_id
                }
            })

    return chunks