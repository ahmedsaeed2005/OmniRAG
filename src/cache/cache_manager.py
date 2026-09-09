import os
import hashlib


CACHE_DIR = "data/vision_cache"


def get_image_hash(image_path):

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    return hashlib.md5(
        image_bytes
    ).hexdigest()


def get_cached_vision(image_path):

    image_hash = get_image_hash(
        image_path
    )

    cache_path = os.path.join(
        CACHE_DIR,
        f"{image_hash}.txt"
    )

    if not os.path.exists(cache_path):
        return None

    with open(
        cache_path,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


def save_vision_cache(
    image_path,
    vision_text
):

    os.makedirs(
        CACHE_DIR,
        exist_ok=True
    )

    image_hash = get_image_hash(
        image_path
    )

    cache_path = os.path.join(
        CACHE_DIR,
        f"{image_hash}.txt"
    )

    with open(
        cache_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(vision_text)