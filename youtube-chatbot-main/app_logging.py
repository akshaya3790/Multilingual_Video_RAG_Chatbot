"""Logging for the app: levels, timestamps and a short id per request.

Every log line carries the id of the question or video load it belongs to, so
the lines of one request can be found together. Set LOG_LEVEL=DEBUG for more.
"""
import contextvars
import logging
import os
import uuid


_request_id = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = _request_id.get()
        return True


def setup_logging():
    logger = logging.getLogger("ytchat")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO")
    logger.propagate = False
    return logger


def new_request_id():
    """Start a new request: later log lines in this thread carry its id."""
    request_id = uuid.uuid4().hex[:8]
    _request_id.set(request_id)
    return request_id
