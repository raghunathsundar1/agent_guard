import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .state import GraphState

logger = logging.getLogger(__name__)


ENHANCEMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert technical assistant optimizing user queries for a Vector Search Engine. "
        "Your task is to take a user's messy question and convert it into a highly specific, "
        "natural language search query. Strip away conversational filler and pleas for help, "
        "but keep the query grammatically structured as a clear technical question or statement. "
        "Output the optimized query in the SAME language as the input — do not translate. "
        "Output ONLY the optimized query, no explanation."
    )),
    ("user", "Detected language: {language}\nOriginal Question: {question}"),
])

_llm: Optional[ChatOpenAI] = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    return _llm


def enhance_question(state: GraphState) -> dict:
    """Rephrase the PII-masked message into a clean vector-search query."""
    original = (state.get("pii_masked_message") or "").strip()
    language = state.get("language", "en")

    if not original:
        logger.warning("No 'pii_masked_message' in state; skipping enhancement.")
        return {"context_enhanced_message": ""}

    try:
        chain = ENHANCEMENT_PROMPT | _get_llm() | StrOutputParser()
        enhanced = chain.invoke({"question": original, "language": language}).strip()

        is_useless = (
            not enhanced
            or enhanced.upper() in ("N/A", "NONE", "NULL", "-", ".")
            or enhanced.lower().startswith("i cannot")
            or enhanced.lower().startswith("i'm unable")
            or len(enhanced) < 4
        )
        if is_useless:
            logger.warning("Enhancer returned a non-useful response %r; using original.", enhanced)
            return {"context_enhanced_message": original}

        logger.info("Question enhanced (lang=%s).", language)
        return {"context_enhanced_message": enhanced}
    except Exception as e:
        logger.error("Question enhancement failed: %s", e, exc_info=True)
        return {"context_enhanced_message": original}
