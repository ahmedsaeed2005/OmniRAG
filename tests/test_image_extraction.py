import fitz

from src.vision.image_analyzer import extract_images_from_page


pdf_path = "data/sample.pdf"

document = fitz.open(pdf_path)

page = document[8]   # Page 9

images = extract_images_from_page(page)

print("Extracted images:", len(images))

for image in images:

    print(
        f"Page: {image['page']} | "
        f"Image: {image['image_number']} | "
        f"Size: {image['width']}x{image['height']} | "
        f"Path: {image['path']}"
    )

document.close()