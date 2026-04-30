"""Logging setup for docling-proxy. Single source of truth for the docling_proxy logger."""
import logging, sys, json
from datetime import datetime as dt
import os

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_FORMAT = os.getenv("LOG_FORMAT", "text").lower()

class _JsonFormatter(logging.Formatter):
    """Минимальный JSON-форматтер без внешних зависимостей."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": dt.utcfromtimestamp(record.created).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _make_log_handler(include_logger_name: bool) -> logging.Handler:
    """Один и тот же handler-фабричный конструктор для всех логгеров.

    include_logger_name=True — добавляет `name:` (для uvicorn.access /
    uvicorn.error, чтобы видеть откуда пришла строка). Для нашего
    docling_proxy — без, чтобы не плодить шум в каждой строке.
    """
    handler = logging.StreamHandler(sys.stdout)
    if _LOG_FORMAT == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        fmt = (
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            if include_logger_name
            else "%(asctime)s [%(levelname)s] %(message)s"
        )
        handler.setFormatter(logging.Formatter(
            fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S",
        ))
    return handler


def _init_logging() -> logging.Logger:
    lg = logging.getLogger("docling_proxy")
    lg.handlers.clear()
    lg.addHandler(_make_log_handler(include_logger_name=False))
    lg.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    lg.propagate = False  # не дублируем в root/uvicorn
    return lg


def _retrofit_uvicorn_loggers() -> None:
    """Привести uvicorn-логи к общему формату прокси (timestamp + level + name).

    uvicorn ставит свои handlers ПОСЛЕ импорта main.py (в Config.configure_logging),
    поэтому делать это в module-level бесполезно — переопределение
    вызывается из lifespan startup, когда uvicorn-конфиг уже применён.
    """
    handler = _make_log_handler(include_logger_name=True)
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.addHandler(handler)
        lg.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
        lg.propagate = False

logger = _init_logging()
