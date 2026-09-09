from rank_bm25 import BM25Okapi


def create_keyword_index(chunks):

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    tokenized_documents = [
        document.lower().split()
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    return bm25


def search_keyword(
    bm25,
    chunks,
    query,
    k=3
):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = scores.argsort()[::-1][:k]

    results = []

    for index in ranked_indices:

        results.append({
            "text": chunks[index]["text"],
            "metadata": chunks[index]["metadata"],
            "score": float(scores[index])
        })

    return results