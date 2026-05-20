import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging with a consistent timestamped format.

    Call this once from an entry point (main.py or server.py), never from
    library modules. Idempotent — safe to call multiple times.
    """
    root = logging.getLogger()
    if getattr(root, "_agent_guard_configured", False):
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    root._agent_guard_configured = True  # type: ignore[attr-defined]
