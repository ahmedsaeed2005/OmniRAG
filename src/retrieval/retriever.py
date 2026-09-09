def retrieve_chunks(
    index,
    chunks,
    query_embedding,
    k=3
):
    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for distance, index_id in zip(distances[0], indices[0]):
        results.append({
            "text": chunks[index_id]["text"],
            "metadata": chunks[index_id]["metadata"],
            "distance": float(distance)
        })

    return results