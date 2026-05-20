# CLAUDE.md

Guidance for Claude Code (or any future LLM agent) working in this repository.

## What this project is

`agent_guard` is a **multi-language safety pipeline** for AI agents, built on
[LangGraph](https://github.com/langchain-ai/langgraph). It takes a raw user
message and runs it through four stages:

```
language ─▶ pii ─▶ injection ─┬─▶ enhance ─▶ passed ─▶ END
                              └─▶ blocked ─▶ END
```

1. **language** — `langdetect` identifies the input's ISO 639-1 code.
2. **pii** — Microsoft Presidio anonymises PII using a per-language spaCy NER
   model when available (`en, es, fr, de, it`), or a universal regex
   recognizer set (email / phone / credit card / IP / URL / IBAN) for any
   other language.
3. **injection** — `gpt-4o-mini` classifies the PII-masked message as SAFE or
   MALICIOUS. The system prompt explicitly handles multilingual jailbreaks
   and always outputs English STATUS/CONFIDENCE/REASON.
4. **enhance** (only if SAFE) — `gpt-4o-mini` rewrites the message into a
   vector-search-friendly query **in the same language as the input**.

If `injection` flags the message MALICIOUS the graph short-circuits to a
`blocked` terminal node and `context_enhanced_message` is never set.

## Layout

| Path | Purpose |
|---|---|
| [agent_guard/state.py](agent_guard/state.py) | `GraphState` TypedDict — keys: `message`, `language`, `pii_masked_message`, `is_safe`, `explanation`, `context_enhanced_message`, `status` |
| [agent_guard/language.py](agent_guard/language.py) | `detect_language_node` (first graph node) |
| [agent_guard/pii.py](agent_guard/pii.py) | `PIIAnonymizerService`, `pii_node`, regex fallback |
| [agent_guard/prompt_injection.py](agent_guard/prompt_injection.py) | `SECURITY_SYSTEM_PROMPT`, `prompt_injection_test` |
| [agent_guard/question_enhancer.py](agent_guard/question_enhancer.py) | `ENHANCEMENT_PROMPT`, `enhance_question` |
| [agent_guard/graph.py](agent_guard/graph.py) | `build_graph()`, `run_pipeline(message)` |
| [agent_guard/logging_config.py](agent_guard/logging_config.py) | `configure_logging()` — call from entry points only |
| [main.py](main.py) | CLI entry point |
| [server.py](server.py) | FastAPI app (`/guard`, `/health`) |
| [tests/](tests/) | pytest suite; LLMs mocked, Presidio is real |

## Running

```bash
# one-time install
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm   # optional but recommended
python -m spacy download fr_core_news_sm   # optional
python -m spacy download de_core_news_sm   # optional
python -m spacy download it_core_news_sm   # optional

# CLI
python main.py "My email is alice@example.com, please summarize this article"

# Server
uvicorn server:app --reload
curl -X POST localhost:8000/guard -H "Content-Type: application/json" \
    -d '{"message":"hello world"}'

# Tests
pytest -v
```

## Conventions (important — read before editing)

- **Library modules NEVER call `load_dotenv()`.** Only `main.py` and
  `server.py` do. Library code must not perform side effects on import.
- **Nodes return dicts, never mutate state in place.** LangGraph merges the
  returned dict into the state. Mutating `state[...] = ...` and returning
  nothing silently drops the update on some graph configurations.
- **LLMs are lazy module-level singletons** via `_get_llm()`. Do not
  re-instantiate `ChatOpenAI` per request.
- **`PIIAnonymizerService` is a process-wide singleton** via
  `get_pii_service()`. First instantiation loads spaCy models (slow — 5–15s).
- **No `print()` in library code** — use `logger`.

## Gotchas

- **First request is slow** because `PIIAnonymizerService.__init__` lazy-loads
  every available spaCy model on first call.
- **Missing spaCy models degrade gracefully.** If e.g. `de_core_news_sm` is
  not installed, the service logs a warning at startup and routes German
  text to the regex fallback. The five models in [requirements.txt](requirements.txt)
  comments are recommended but only `en_core_web_sm` is strictly necessary.
- **Regex fallback misses person/location/org names.** For non-Western
  languages where you need NER (e.g. Japanese), install
  `ja_core_news_sm` and add an entry to `_SPACY_MODEL_MAP` in
  [agent_guard/pii.py](agent_guard/pii.py).
- **`.env` is gitignored.** Copy [.env.example](.env.example) and set
  `OPENAI_API_KEY`.
- **Tests mock the OpenAI LLMs** (no network), but **do** instantiate a real
  Presidio service. `test_pii.py` skips Spanish tests if `es_core_news_sm`
  isn't installed.

## Architectural decisions worth knowing

- **Conditional edge after injection check** (not after enhancement) — this
  means a malicious prompt never reaches the enhancer, saving an OpenAI call
  and avoiding any downstream use of attacker-controlled text.
- **Language detection runs first** so every node downstream sees
  `state["language"]`. Detection is deterministic (`DetectorFactory.seed = 0`)
  for reproducible tests.
- **`GraphState` uses `total=False`** so each node only has to return the
  keys it produces — LangGraph merges partial updates.
