from src.embeddings.embedder import create_embeddings

from src.retrieval.retriever import retrieve_chunks

from src.retrieval.keyword import (
    create_keyword_index,
    search_keyword
)

from src.retrieval.hybrid import hybrid_search

from src.retrieval.reranker import rerank_results

from src.retrieval.query_parser import extract_page_number


class RetrievalPipeline:

    def __init__(
        self,
        chunks,
        index
    ):

        self.chunks = chunks
        self.index = index

        self.bm25 = create_keyword_index(
            chunks
        )

    def retrieve(
        self,
        question,
        semantic_k=5,
        keyword_k=5,
        final_k=3
    ):

        # =========================
        # 1. Detect Page
        # =========================

        page_number = extract_page_number(
            question
        )

        # =========================
        # 2. Page-Aware Retrieval
        # =========================

        if page_number is not None:

            page_chunks = [
                chunk
                for chunk in self.chunks
                if chunk["metadata"]["page"] == page_number
            ]

            if not page_chunks:
                return []

            page_texts = [
                chunk["text"]
                for chunk in page_chunks
            ]

            page_embeddings = create_embeddings(
                page_texts
            )

            from src.retrieval.vector_store import (
                create_vector_store
            )

            page_index = create_vector_store(
                page_embeddings
            )

            page_bm25 = create_keyword_index(
                page_chunks
            )

            query_embedding = create_embeddings(
                [question]
            )

            semantic_results = retrieve_chunks(
                page_index,
                page_chunks,
                query_embedding,
                k=min(
                    semantic_k,
                    len(page_chunks)
                )
            )

            keyword_results = search_keyword(
                page_bm25,
                page_chunks,
                question,
                k=min(
                    keyword_k,
                    len(page_chunks)
                )
            )

        # =========================
        # 3. Normal Retrieval
        # =========================

        else:

            query_embedding = create_embeddings(
                [question]
            )

            semantic_results = retrieve_chunks(
                self.index,
                self.chunks,
                query_embedding,
                k=semantic_k
            )

            keyword_results = search_keyword(
                self.bm25,
                self.chunks,
                question,
                k=keyword_k
            )

        # =========================
        # 4. Hybrid
        # =========================

        hybrid_results = hybrid_search(
            semantic_results,
            keyword_results,
            k=min(
                5,
                len(
                    semantic_results
                    + keyword_results
                )
            )
        )

        # =========================
        # 5. Reranker
        # =========================

        final_results = rerank_results(
            question,
            hybrid_results,
            k=min(
                final_k,
                len(hybrid_results)
            )
        )

        return final_results