import fitz

from src.ingestion.document_parser import extract_page_content


pdf = fitz.open("data/sample.pdf")

# Page 9
page = pdf[8]

content = extract_page_content(
    page,
    use_vision=True
)

print("\n==============================")
print("PAGE TEXT")
print("==============================\n")

print(content["text"])

pdf.close()