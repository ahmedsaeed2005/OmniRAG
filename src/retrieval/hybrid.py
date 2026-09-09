def hybrid_search(
    semantic_results,
    keyword_results,
    k=3
):

    scores = {}

    # Semantic Search
    for rank, result in enumerate(semantic_results, start=1):

        chunk_id = (
            result["metadata"]["page"],
            result["metadata"]["chunk_id"]
        )

        rrf_score = 1 / (60 + rank)

        scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score

    # Keyword Search
    for rank, result in enumerate(keyword_results, start=1):

        chunk_id = (
            result["metadata"]["page"],
            result["metadata"]["chunk_id"]
        )

        rrf_score = 1 / (60 + rank)

        scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score

    # Sort by combined score
    ranked_chunks = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    # Get original result information
    all_results = semantic_results + keyword_results

    for (page, chunk_id), score in ranked_chunks[:k]:

        for result in all_results:

            if (
                result["metadata"]["page"] == page
                and result["metadata"]["chunk_id"] == chunk_id
            ):

                results.append({
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "hybrid_score": score
                })

                break

    return results