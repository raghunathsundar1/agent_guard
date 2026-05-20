# agent_guard

A **multi-language safety pipeline** for AI agents. Takes a raw user message
and runs four checks in sequence:

1. **Language detection** — identifies the input's language (ISO 639-1).
2. **PII anonymization** — redacts emails, phone numbers, names, credit
   cards, etc. using Microsoft Presidio (multi-language) with a universal
   regex fallback for unsupported languages.
3. **Prompt-injection detection** — `gpt-4o-mini` classifies the message as
   SAFE or MALICIOUS, with explicit handling of multilingual jailbreaks.
4. **Query enhancement** — rewrites the message into a vector-search-friendly
   query, **preserving the input language**.

If a message is flagged malicious, the pipeline short-circuits and the
enhancement step is skipped.

## Demo

**Safe input** — PII is redacted and the query is enhanced for vector search:

![Safe input demo](screenshots/demo_safe.png)

**Malicious input** — injection attack detected and blocked before reaching the RAG system:

![Blocked input demo](screenshots/demo_blocked.png)

## Architecture

```
language ─▶ pii ─▶ injection ─┬─▶ enhance ─▶ passed ─▶ END
                              └─▶ blocked ─▶ END
```

Built on [LangGraph](https://github.com/langchain-ai/langgraph).

## Quickstart

```bash
# 1. Install
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# 2. spaCy NER models (English required; others recommended)
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download de_core_news_sm
python -m spacy download it_core_news_sm

# 3. Config
cp .env.example .env
# edit .env and set OPENAI_API_KEY

# 4a. Run via CLI
python main.py "My email is alice@example.com, please summarize this article"

# 4b. Run as a server
uvicorn server:app --reload
# then in another shell:
curl -X POST http://localhost:8000/guard \
     -H "Content-Type: application/json" \
     -d '{"message":"hola, mi correo es foo@bar.com"}'

# 5. Tests
pytest -v
```

## Sample output

```json
{
  "message": "My email is alice@example.com, please summarize this article",
  "language": "en",
  "pii_masked_message": "My email is <EMAIL_ADDRESS>, please summarize this article",
  "is_safe": true,
  "explanation": "STATUS: SAFE\nCONFIDENCE: High\nREASON: Benign request to summarize.",
  "context_enhanced_message": "Summary of the provided article",
  "status": "passed"
}
```

For a malicious input the pipeline ends at `"status": "blocked"` and
`context_enhanced_message` is absent.

## API

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/health` | GET | – | `{"status":"ok"}` |
| `/guard` | POST | `{"message": "..."}` | Full `GraphState` |
| `/ask` | POST | `{"message": "...", "upstream_url": "..."}` | Full `GraphState` + `rag_response` |

## Language support

| Component | Coverage |
|---|---|
| Language detection | 55+ languages via `lingua-language-detector` |
| PII (full NER) | `en, es, fr, de, it` — additional spaCy models can be added |
| PII (regex fallback) | Every other language (emails, phones, cards, IPs, URLs, IBANs) |
| Prompt injection | Any language — `gpt-4o-mini` natively multilingual |
| Query enhancement | Any language — output preserves the input's language |

## License

MIT
