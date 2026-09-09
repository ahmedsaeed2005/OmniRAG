import io
import os
import fitz

from PIL import Image

from src.ingestion.ocr import extract_text_from_image
from src.vision.image_analyzer import analyze_image


def extract_page_content(page, use_vision=False):

    # =========================
    # 1. Extract normal PDF text
    # =========================

    text = page.get_text().strip()

    combined_text = text

    # =========================
    # 2. Extract images
    # =========================

    images = page.get_images(full=True)

    for image_number, image in enumerate(
        images,
        start=1
    ):

        xref = image[0]

        image_data = page.parent.extract_image(
            xref
        )

        image_bytes = image_data["image"]
        extension = image_data["ext"]

        # =========================
        # 3. Convert image → PIL
        # =========================

        pil_image = Image.open(
            io.BytesIO(image_bytes)
        )

        # =========================
        # 4. OCR
        # =========================

        ocr_text = extract_text_from_image(
            pil_image
        ).strip()

        if ocr_text:

            combined_text += (
                "\n\n[OCR]\n"
                + ocr_text
            )

        # =========================
        # 5. Save image
        # =========================

        output_dir = "data/extracted_images"

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        output_path = os.path.join(
            output_dir,
            f"page_{page.number + 1}_image_{image_number}.{extension}"
        )

        with open(
            output_path,
            "wb"
        ) as f:

            f.write(image_bytes)

        # =========================
        # 6. Gemini Vision
        # =========================

        if use_vision:

            vision_text = analyze_image(
                output_path
            )

            if vision_text:

                combined_text += (
                    "\n\n[VISION]\n"
                    + vision_text
                )

    # =========================
    # 7. Return
    # =========================

    return {
        "text": combined_text,
        "images": images
    }