import base64

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from src.cache.cache_manager import (
    get_cached_vision,
    save_vision_cache
)


def analyze_image(image_path):

    # =========================
    # 1. Check Cache
    # =========================

    cached_result = get_cached_vision(
        image_path
    )

    if cached_result is not None:

        print(
            f"[VISION CACHE] {image_path}"
        )

        return cached_result

    print(
        f"[VISION API] {image_path}"
    )

    # =========================
    # 2. Create Vision Model
    # =========================

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0
    )

    # =========================
    # 3. Read Image
    # =========================

    with open(
        image_path,
        "rb"
    ) as image_file:

        image_bytes = image_file.read()

    image_base64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    # =========================
    # 4. Prompt
    # =========================

    prompt = """
Analyze this image carefully.

Describe:

1. What the image contains.
2. Any diagrams, architecture, charts, or tables.
3. Important labels and text visible in the image.
4. Relationships between the elements.
5. The main idea or meaning of the image.

Return a clear description that can be used
as knowledge in a Retrieval-Augmented Generation system.

Do not invent information that is not visible in the image.

If the image is only decorative and contains no useful
information, clearly say that it is decorative.
"""

    # =========================
    # 5. Send to Gemini
    # =========================

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/jpeg;base64,"
                        + image_base64
                    )
                }
            }
        ]
    )

    response = llm.invoke(
        [message]
    )

    # =========================
    # 6. Extract Text
    # =========================

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

        vision_text = "\n".join(
            text_parts
        ).strip()

    else:

        vision_text = str(
            content
        ).strip()

    # =========================
    # 7. Save Cache
    # =========================

    save_vision_cache(
        image_path,
        vision_text
    )

    return vision_text