from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.llm.gemini import generate_answer


class RAGState(TypedDict):

    question: str
    retriever: RetrievalPipeline
    results: list
    answer: str
    retry_count: int


def retrieve_node(state: RAGState):

    print("\n[AGENT] Retrieving documents...")

    results = state["retriever"].retrieve(
        state["question"]
    )

    return {
        "results": results
    }


def evaluate_results_node(state: RAGState):

    print("[AGENT] Evaluating retrieval quality...")

    results = state["results"]

    if not results:
        return {
            "retry_count": state["retry_count"] + 1
        }

    best_score = results[0]["rerank_score"]

    print(f"[AGENT] Best rerank score: {best_score}")

    return {
        "retry_count": state["retry_count"]
    }


def route_after_evaluation(state: RAGState):

    results = state["results"]
    retry_count = state["retry_count"]

    if not results:
        print("[AGENT] No results found.")

        if retry_count < 2:
            print("[AGENT] Retrying retrieval...")
            return "retrieve"

        return "answer"

    best_score = results[0]["rerank_score"]

    # Threshold مبدئي للتجربة
    if best_score < 0 and retry_count < 2:

        print("[AGENT] Weak results detected.")
        print("[AGENT] Retrying retrieval...")

        return "retrieve"

    print("[AGENT] Results are good enough.")

    return "answer"


def answer_node(state: RAGState):

    print("[AGENT] Generating answer...")

    answer = generate_answer(
        state["question"],
        state["results"]
    )

    return {
        "answer": answer
    }


def build_graph():

    graph = StateGraph(RAGState)

    graph.add_node(
        "retrieve",
        retrieve_node
    )

    graph.add_node(
        "evaluate",
        evaluate_results_node
    )

    graph.add_node(
        "answer",
        answer_node
    )

    graph.add_edge(
        START,
        "retrieve"
    )

    graph.add_edge(
        "retrieve",
        "evaluate"
    )

    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "retrieve": "retrieve",
            "answer": "answer"
        }
    )

    graph.add_edge(
        "answer",
        END
    )

    return graph.compile()