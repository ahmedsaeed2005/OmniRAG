from src.ingestion.pdf_loader import load_pdf


pages = load_pdf(
    "data/sample.pdf"
)


print(
    "Number of pages:",
    len(pages)
)


for page in pages:

    print("\n====================")
    print("Page:", page["page"])
    print(
        "Text length:",
        len(page["text"])
    )
    print(
        "Images:",
        len(page["images"])
    )