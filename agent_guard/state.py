from typing import Any, TypedDict, Literal


class GraphState(TypedDict, total=False):
    message: str
    language: str
    pii_masked_message: str
    is_safe: bool
    explanation: str
    context_enhanced_message: str
    status: Literal["passed", "blocked", "error"]
    rag_response: str
    rag_sources: list[Any]
