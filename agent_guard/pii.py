import logging
from typing import List, Optional

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

from .state import GraphState

logger = logging.getLogger(__name__)

SUPPORTED_LANGS = {"en", "es", "fr", "de", "it"}

# Entities actually worth redacting in a RAG query pipeline.
# DATE_TIME and NRP are intentionally excluded — they have extremely high
# false-positive rates (any number looks like a date; common nouns look like
# nationalities) and dates in queries are almost never sensitive PII.
DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "URL",
    "US_SSN",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "UK_NHS",
    "MEDICAL_LICENSE",
    "CRYPTO",
    "LOCATION",
]

_SPACY_MODEL_MAP = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "it": "it_core_news_sm",
}


_REGEX_LANG = "en"


def _build_regex_fallback_analyzer(nlp_engine=None) -> AnalyzerEngine:
    """Universal regex-based analyzer for languages without a spaCy NER model.

    Regex patterns don't depend on tokenization, so we register them under
    language `en` and call `.analyze(..., language="en")` regardless of the
    actual input language. An NLP engine is shared with the main analyzer
    when available to avoid loading spaCy twice.
    """
    registry = RecognizerRegistry(supported_languages=[_REGEX_LANG])

    patterns = [
        ("EMAIL_ADDRESS", [Pattern("email", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", 0.9)]),
        ("PHONE_NUMBER", [Pattern("phone", r"\+?\d[\d\s().-]{7,}\d", 0.5)]),
        ("CREDIT_CARD", [Pattern("cc", r"\b(?:\d[ -]*?){13,19}\b", 0.4)]),
        ("IP_ADDRESS", [Pattern("ipv4", r"\b(?:\d{1,3}\.){3}\d{1,3}\b", 0.7)]),
        ("URL", [Pattern("url", r"https?://[^\s<>\"']+", 0.6)]),
        ("IBAN_CODE", [Pattern("iban", r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", 0.7)]),
    ]
    for entity, pats in patterns:
        registry.add_recognizer(
            PatternRecognizer(supported_entity=entity, patterns=pats, supported_language=_REGEX_LANG)
        )

    kwargs = {"registry": registry, "supported_languages": [_REGEX_LANG]}
    if nlp_engine is not None:
        kwargs["nlp_engine"] = nlp_engine
    return AnalyzerEngine(**kwargs)


class PIIAnonymizerService:
    """
    Multi-language PII anonymization using Microsoft Presidio.

    Attempts to load spaCy NER models for English, Spanish, French, German, and Italian.
    Missing models are logged and skipped at startup. For any language outside the
    loaded set, a universal regex recognizer set handles emails, phone numbers,
    credit cards, IPs, URLs, and IBANs.
    """

    def __init__(self) -> None:
        loaded_langs: List[str] = []
        models_to_load = []
        for lang, model in _SPACY_MODEL_MAP.items():
            try:
                import spacy
                spacy.load(model)
                models_to_load.append({"lang_code": lang, "model_name": model})
                loaded_langs.append(lang)
            except (OSError, ImportError) as e:
                logger.warning(
                    "spaCy model '%s' for language '%s' not installed; "
                    "PII detection for '%s' will use regex fallback. (%s)",
                    model, lang, lang, e,
                )

        self.supported_langs = set(loaded_langs)

        nlp_engine = None
        if models_to_load:
            nlp_engine = NlpEngineProvider(
                nlp_configuration={"nlp_engine_name": "spacy", "models": models_to_load}
            ).create_engine()
            self.analyzer: Optional[AnalyzerEngine] = AnalyzerEngine(
                nlp_engine=nlp_engine,
                supported_languages=loaded_langs,
            )
            logger.info("Presidio analyzer initialized for languages: %s", loaded_langs)
        else:
            self.analyzer = None
            logger.warning("No spaCy models available; all PII will use regex fallback.")

        self.anonymizer = AnonymizerEngine()
        fallback_engine = nlp_engine if (nlp_engine and _REGEX_LANG in loaded_langs) else None
        self.regex_fallback = _build_regex_fallback_analyzer(nlp_engine=fallback_engine)

    def anonymize_pii(
        self,
        text: str,
        language: str = "en",
        block_list: Optional[List[str]] = None,
        allow_list: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
    ) -> str:
        if not text or not isinstance(text, str):
            return str(text) if text else text

        block_list = block_list or []
        allow_list = allow_list or []
        # Use the curated default list unless the caller explicitly overrides.
        # Passing None to Presidio means "all entities", which includes DATE_TIME
        # and NRP — both have very high false-positive rates on business text.
        active_entities = entities if entities is not None else DEFAULT_ENTITIES

        try:
            use_native = self.analyzer is not None and language in self.supported_langs

            if use_native:
                ad_hoc = []
                if block_list:
                    ad_hoc.append(
                        PatternRecognizer(
                            supported_entity="CUSTOM_BLOCK_LIST",
                            deny_list=block_list,
                            supported_language=language,
                        )
                    )
                results = self.analyzer.analyze(
                    text=text,
                    language=language,
                    entities=active_entities,
                    allow_list=allow_list,
                    ad_hoc_recognizers=ad_hoc or None,
                )
            else:
                ad_hoc = []
                if block_list:
                    ad_hoc.append(
                        PatternRecognizer(
                            supported_entity="CUSTOM_BLOCK_LIST",
                            deny_list=block_list,
                            supported_language=_REGEX_LANG,
                        )
                    )
                results = self.regex_fallback.analyze(
                    text=text,
                    language=_REGEX_LANG,
                    entities=active_entities,
                    allow_list=allow_list,
                    ad_hoc_recognizers=ad_hoc or None,
                )

            return self.anonymizer.anonymize(text=text, analyzer_results=results).text

        except Exception as e:
            logger.error("PII anonymization failed: %s", e, exc_info=True)
            raise RuntimeError("Anonymization pipeline failed. Halting to prevent data leak.") from e


_service: Optional[PIIAnonymizerService] = None


def get_pii_service() -> PIIAnonymizerService:
    global _service
    if _service is None:
        _service = PIIAnonymizerService()
    return _service


def pii_node(state: GraphState) -> dict:
    text = state.get("message", "") or ""
    language = state.get("language", "en")
    masked = get_pii_service().anonymize_pii(text, language=language)
    return {"pii_masked_message": masked}
