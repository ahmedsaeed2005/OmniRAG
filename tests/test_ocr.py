from PIL import Image

from src.ingestion.ocr import extract_text_from_image


image = Image.open("data/test_page.png")

text = extract_text_from_image(image)

print("\n====================")
print("OCR RESULT")
print("====================")
print(text)