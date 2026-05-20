import pytest

from agent_guard.pii import PIIAnonymizerService, pii_node


@pytest.fixture(scope="module")
def service() -> PIIAnonymizerService:
    return PIIAnonymizerService()


def test_english_email_redacted(service: PIIAnonymizerService):
    out = service.anonymize_pii("my email is foo@bar.com please contact me", language="en")
    assert "foo@bar.com" not in out


def test_unsupported_language_uses_regex_fallback(service: PIIAnonymizerService):
    out = service.anonymize_pii("私のメールはfoo@bar.comです", language="ja")
    assert "foo@bar.com" not in out


def test_undetected_language_uses_regex_fallback(service: PIIAnonymizerService):
    out = service.anonymize_pii("Contact: foo@bar.com", language="und")
    assert "foo@bar.com" not in out


def test_spanish_email_redacted(service: PIIAnonymizerService):
    if "es" not in service.supported_langs:
        pytest.skip("es_core_news_sm not installed")
    out = service.anonymize_pii("mi correo es foo@bar.com gracias", language="es")
    assert "foo@bar.com" not in out


def test_empty_input_returns_empty(service: PIIAnonymizerService):
    assert service.anonymize_pii("", language="en") == ""


def test_credit_card_regex_fallback(service: PIIAnonymizerService):
    out = service.anonymize_pii("カードは4111 1111 1111 1111です", language="ja")
    assert "4111 1111 1111 1111" not in out


def test_pii_node_uses_language_from_state(service: PIIAnonymizerService):
    state = {"message": "Mail: alice@example.com", "language": "und"}
    result = pii_node(state)
    assert "alice@example.com" not in result["pii_masked_message"]
