from sentence_transformers import CrossEncoder


model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_results(
    query,
    results,
    k=3
):

    pairs = []

    for result in results:

        pairs.append(
            (
                query,
                result["text"]
            )
        )

    scores = model.predict(pairs)

    reranked = []

    for result, score in zip(
        results,
        scores
    ):

        reranked.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "rerank_score": float(score)
        })

    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked[:k]