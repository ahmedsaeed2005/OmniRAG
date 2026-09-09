import fitz

from src.ingestion.document_parser import extract_page_content


def load_pdf(file_path, use_vision=False):

    document = fitz.open(file_path)

    pages = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        content = extract_page_content(
            page,
            use_vision=use_vision
        )

        pages.append({
            "page": page_number,
            "text": content["text"],
            "images": content["images"]
        })

    document.close()

    return pages