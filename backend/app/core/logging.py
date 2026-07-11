import logging
import sys
from collections.abc import MutableMapping
from typing import Any, cast

import structlog

_PII_KEYS = {"email", "password", "full_name", "phone", "address"}


def _redact_pii(_logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]) -> Any:
    for key in list(event_dict):
        if key.lower() in _PII_KEYS:
            event_dict[key] = "***redacted***"
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=log_level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            _redact_pii,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(log_level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
