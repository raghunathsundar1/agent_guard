from langgraph.graph import StateGraph, END

from .state import GraphState
from .language import detect_language_node
from .pii import pii_node
from .prompt_injection import prompt_injection_test
from .question_enhancer import enhance_question


def _route_after_injection(state: GraphState) -> str:
    return "enhance" if state.get("is_safe") else "blocked"


def _blocked_node(state: GraphState) -> dict:
    return {"status": "blocked"}


def _passed_node(state: GraphState) -> dict:
    return {"status": "passed"}


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("language", detect_language_node)
    g.add_node("pii", pii_node)
    g.add_node("injection", prompt_injection_test)
    g.add_node("enhance", enhance_question)
    g.add_node("passed", _passed_node)
    g.add_node("blocked", _blocked_node)

    g.set_entry_point("language")
    g.add_edge("language", "pii")
    g.add_edge("pii", "injection")
    g.add_conditional_edges(
        "injection",
        _route_after_injection,
        {"enhance": "enhance", "blocked": "blocked"},
    )
    g.add_edge("enhance", "passed")
    g.add_edge("passed", END)
    g.add_edge("blocked", END)

    return g.compile()


_compiled = None


def run_pipeline(message: str) -> GraphState:
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled.invoke({"message": message})
