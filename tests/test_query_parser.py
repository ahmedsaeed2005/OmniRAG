from src.retrieval.query_parser import extract_page_number


queries = [
    "Explain the 3 V's on page 8",
    "What is on page 12?",
    "اشرح صفحة 5",
    "ما الموجود في الصفحة 9؟",
    "What is Big Data?"
]


for query in queries:

    page = extract_page_number(query)

    print("\nQuestion:", query)
    print("Page:", page)