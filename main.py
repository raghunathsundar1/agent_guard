"""CLI entry point for the agent_guard safety pipeline.

Usage:
    python main.py "your message here"
    python main.py            # prompts for input interactively
"""
import io
import json
import sys

from dotenv import load_dotenv

from agent_guard import run_pipeline
from agent_guard.logging_config import configure_logging


def main() -> int:
    load_dotenv()
    configure_logging()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
    else:
        try:
            message = input("Enter a message: ").strip()
        except EOFError:
            print("No input provided.", file=sys.stderr)
            return 2

    if not message:
        print("Empty message; nothing to do.", file=sys.stderr)
        return 2

    final_state = run_pipeline(message)
    print(json.dumps(final_state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
