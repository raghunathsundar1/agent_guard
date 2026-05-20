"""GuardedRAGPipeline — wrap any RAG callable with the agent_guard safety subgraph.

Usage (simple callable):

    from agent_guard import GuardedRAGPipeline

    def my_rag(query: str) -> str:
        # your vector search + LLM here
        return "The answer is ..."

    pipeline = GuardedRAGPipeline(my_rag)
    result = pipeline.ask("What is the refund policy?")
    print(result["rag_response"])   # answer, or None if blocked

Usage (RAG callable that also returns sources):

    def my_rag(query: str) -> dict:
        return {"answer": "...", "sources": ["doc1", "doc2"]}

    pipeline = GuardedRAGPipeline(my_rag, returns_sources=True)

Usage (LangGraph Runnable):

    from langgraph.graph import StateGraph
    # build your own RAG graph whose input key is "query" ...
    rag_graph = my_rag_graph.compile()

    pipeline = GuardedRAGPipeline(rag_graph, is_langgraph=True)
"""
import logging
from typing import Any, Callable

from langgraph.graph import StateGraph, END

from .language import detect_language_node
from .pii import pii_node
from .prompt_injection import prompt_injection_test
from .question_enhancer import enhance_question
from .state import GraphState

logger = logging.getLogger(__name__)


def _make_rag_node(
    rag_fn: Callable,
    returns_sources: bool,
    is_langgraph: bool,
) -> Callable[[GraphState], dict]:
    def node(state: GraphState) -> dict:
        query = state.get("context_enhanced_message") or state.get("pii_masked_message", "")

        try:
            if is_langgraph:
                raw = rag_fn.invoke({"query": query})
                rag_response = raw.get("answer") or raw.get("response") or str(raw)
                rag_sources = raw.get("sources") or raw.get("documents") or []
            elif returns_sources:
                raw = rag_fn(query)
                rag_response = raw.get("answer") or raw.get("response") or str(raw)
                rag_sources = raw.get("sources") or raw.get("documents") or []
            else:
                rag_response = str(rag_fn(query))
                rag_sources = []
        except Exception as e:
            logger.error("RAG function raised an error: %s", e, exc_info=True)
            return {"rag_response": None, "rag_sources": [], "status": "error"}

        logger.info("RAG node completed (query_len=%d, response_len=%d)", len(query), len(str(rag_response)))
        return {"rag_response": rag_response, "rag_sources": rag_sources, "status": "passed"}

    return node


class GuardedRAGPipeline:
    """
    Composes agent_guard's safety pipeline with any RAG callable as a single
    LangGraph graph.

    Graph topology:
        language → pii → injection ──(safe)──▶ enhance → rag → END
                                   └─(unsafe)─▶ blocked → END

    Args:
        rag_fn: Any callable ``(query: str) -> str | dict``, or a compiled
                LangGraph runnable that accepts ``{"query": str}``.
        returns_sources: Set True when ``rag_fn`` returns a dict with
                         ``{"answer": ..., "sources": [...]}`` instead of a
                         plain string.
        is_langgraph: Set True when ``rag_fn`` is a compiled LangGraph graph.
    """

    def __init__(
        self,
        rag_fn: Callable,
        *,
        returns_sources: bool = False,
        is_langgraph: bool = False,
    ) -> None:
        self._compiled = self._build(rag_fn, returns_sources, is_langgraph)

    @staticmethod
    def _route_after_injection(state: GraphState) -> str:
        return "enhance" if state.get("is_safe") else "blocked"

    @staticmethod
    def _blocked_node(state: GraphState) -> dict:
        return {"status": "blocked"}

    def _build(self, rag_fn, returns_sources, is_langgraph):
        rag_node = _make_rag_node(rag_fn, returns_sources, is_langgraph)

        g = StateGraph(GraphState)
        g.add_node("language", detect_language_node)
        g.add_node("pii", pii_node)
        g.add_node("injection", prompt_injection_test)
        g.add_node("enhance", enhance_question)
        g.add_node("rag", rag_node)
        g.add_node("blocked", self._blocked_node)

        g.set_entry_point("language")
        g.add_edge("language", "pii")
        g.add_edge("pii", "injection")
        g.add_conditional_edges(
            "injection",
            self._route_after_injection,
            {"enhance": "enhance", "blocked": "blocked"},
        )
        g.add_edge("enhance", "rag")
        g.add_edge("rag", END)
        g.add_edge("blocked", END)
        return g.compile()

    def ask(self, message: str) -> GraphState:
        """Run the full guarded pipeline and return the final state."""
        return self._compiled.invoke({"message": message})

    def __call__(self, message: str) -> GraphState:
        return self.ask(message)

    @property
    def graph(self):
        return self._compiled
