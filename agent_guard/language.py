import logging

from lingua import LanguageDetectorBuilder, IsoCode639_1

from .state import GraphState

logger = logging.getLogger(__name__)

# Build once at import time. `with_preloaded_language_models()` keeps all models
# in memory after first use — slower cold start but fast on every subsequent call.
# `with_minimum_relative_distance(0.1)` requires the winning language to lead the
# runner-up by at least 10 percentage points, which eliminates the
# English ↔ Danish/Norwegian/Swedish/Dutch confusion that langdetect suffers from.
_detector = (
    LanguageDetectorBuilder
    .from_all_languages()
    .with_minimum_relative_distance(0.1)
    .build()
)


def detect_language_node(state: GraphState) -> dict:
    text = (state.get("message") or "").strip()

    if len(text) < 5:
        return {"language": "en"}

    result = _detector.detect_language_of(text)

    if result is None:
        logger.info("Language detection inconclusive; defaulting to 'en'")
        return {"language": "en"}

    lang = result.iso_code_639_1.name.lower()
    logger.info("Detected language: %s", lang)
    return {"language": lang}
