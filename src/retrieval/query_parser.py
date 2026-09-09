import re


def extract_page_number(query):

    patterns = [
        r"page\s+(\d+)",
        r"page\s*#\s*(\d+)",
        r"صفحة\s+(\d+)",
        r"الصفحة\s+(\d+)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            query.lower()
        )

        if match:
            return int(match.group(1))

    return None