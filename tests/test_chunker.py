from src.ingestion.pdf_loader import load_pdf
from src.chunking.chunker import create_chunks


pages = load_pdf("data/sample.pdf")

chunks = create_chunks(pages)

print("Number of pages:", len(pages))
print("Number of chunks:", len(chunks))

for chunk in chunks[:5]:
    print("\n====================")
    print("Page:", chunk["metadata"]["page"])
    print("Chunk ID:", chunk["metadata"]["chunk_id"])
    print("Text:")
    print(chunk["text"])