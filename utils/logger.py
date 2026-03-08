"""
utils/logger.py
Structured logging for InsureAgent using structlog.

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)

    log.info("claim_processed",
        user_id="P-1001",
        verdict="APPROVED",
        payout=700,
        latency_ms=1240,
        model="teacher",
        tool_calls=["lookup_policy", "check_rules", "calculate_payout"]
    )

Output (JSON in production, coloured in development):
    {
        "event": "claim_processed",
        "user_id": "P-1001",
        "verdict": "APPROVED",
        "payout": 700,
        "latency_ms": 1240,
        "model": "teacher",
        "tool_calls": ["lookup_policy", "check_rules", "calculate_payout"],
        "timestamp": "2026-03-08T14:00:00.000000Z",
        "level": "info"
    }
"""

import logging
import os
import structlog


def configure_logging():
    """
    Configure structlog.

    Development (LOG_FORMAT=text): coloured, human-readable output
    Production  (LOG_FORMAT=json): JSON output, one line per event
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "text").lower()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(message)s",
    )

    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if log_format == "json":
        # Production: JSON output for log aggregation (Datadog, CloudWatch)
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: coloured, readable output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__):
    """Return a structlog logger bound to the given module name."""
    return structlog.get_logger(name)


# Configure on import
configure_logging()


# ── STANDARD LOG EVENTS ───────────────────────────────────────────────────────
# Centralise event names to avoid typos across modules

class Events:
    CLAIM_RECEIVED   = "claim_received"
    CLAIM_PROCESSED  = "claim_processed"
    TOOL_CALLED      = "tool_called"
    TOOL_ERROR       = "tool_error"
    AGENT_STARTED    = "agent_started"
    AGENT_FINISHED   = "agent_finished"
    HALLUCINATION    = "hallucination_detected"
    MODEL_LOADED     = "model_loaded"
    VALIDATION_RUN   = "validation_run"
    VALIDATION_ERROR = "validation_error"