import fitz

from src.ingestion.document_parser import extract_page_content


document = fitz.open("data/sample.pdf")


test_pages = [9, 12, 18, 23]


for page_number in test_pages:

    page = document[page_number - 1]

    result = extract_page_content(page)

    print("\n====================")
    print("PAGE:", page_number)
    print("====================")

    print("Combined text length:")
    print(len(result["text"]))

    print("\nOCR + TEXT:")
    print(result["text"][:1500])


document.close()