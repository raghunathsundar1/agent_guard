import agent_guard.question_enhancer as qe


class _FakeChain:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.captured_input: dict | None = None

    def __or__(self, other):
        return self

    def invoke(self, kwargs):
        self.captured_input = kwargs
        return self.response_text


def _patch_chain(monkeypatch, response: str) -> _FakeChain:
    fake = _FakeChain(response)
    monkeypatch.setattr(qe, "ENHANCEMENT_PROMPT", fake)
    monkeypatch.setattr(qe, "_get_llm", lambda: fake)
    qe._llm = fake
    return fake


def test_returns_enhanced_query(monkeypatch):
    _patch_chain(monkeypatch, "What causes HTTP 500 errors when submitting a Stripe payment form?")
    result = qe.enhance_question({
        "pii_masked_message": "hey my payment form gives 500 idk help",
        "language": "en",
    })
    assert result["context_enhanced_message"].startswith("What causes HTTP 500")


def test_empty_message_returns_empty():
    result = qe.enhance_question({"pii_masked_message": "", "language": "en"})
    assert result == {"context_enhanced_message": ""}


def test_language_is_forwarded(monkeypatch):
    fake = _patch_chain(monkeypatch, "consulta optimizada")
    qe.enhance_question({"pii_masked_message": "ayúdame con mi pago", "language": "es"})
    assert fake.captured_input["language"] == "es"


def test_falls_back_to_original_on_error(monkeypatch):
    class _Boom:
        def __or__(self, other):
            return self

        def invoke(self, _):
            raise RuntimeError("LLM down")

    boom = _Boom()
    monkeypatch.setattr(qe, "ENHANCEMENT_PROMPT", boom)
    monkeypatch.setattr(qe, "_get_llm", lambda: boom)
    qe._llm = boom

    result = qe.enhance_question({"pii_masked_message": "original text", "language": "en"})
    assert result["context_enhanced_message"] == "original text"
