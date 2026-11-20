"""JSON logs with a request id on every line.

One JSON object per line is what log platforms (Loki, CloudWatch, Splunk,
Datadog) parse without custom regexes. The request id ties together every
log line of one request and is returned to the client in X-Request-ID, so a
user report ("my answer was wrong") can be traced to the exact log lines.
"""
import contextvars
import json
import logging
import time

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id.get(),
            "msg": record.getMessage(),
        }
        line.update(getattr(record, "fields", {}))  # log.info("...", extra={"fields": {...}})
        if record.exc_info:
            line["exc"] = self.formatException(record.exc_info)
        return json.dumps(line)


def setup_logging(level: str = "INFO"):
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
