"""Configuration: ENV variables, defaults, parsing helpers."""
import os, logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("docling_proxy")

def _env_bool(name: str, default: bool) -> bool:
    """Парсер булевых ENV: true/1/yes/on → True, false/0/no/off/"" → False.

    Регистронезависимо. Неизвестное значение → default + warning. Цель —
    явные дефолты вместо «как парсит os.getenv».
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    s = raw.strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off", ""):
        return False
    logger.warning(f"[config] {name}={raw!r} not a boolean, using default={default}")
    return default


def _resolve_int_threshold(payload_value, env_value: int, env_source: str, name: str, rid8: str):
    """Резолв payload > env > default для целочисленных параметров.

    Возвращает (value, source). Невалидные значения (не-int, отрицательные,
    пустая строка) → fallback на env с WARNING.
    """
    if payload_value is None:
        return env_value, env_source
    s = str(payload_value).strip()
    if s == "":
        return env_value, env_source
    try:
        v = int(s)
    except (TypeError, ValueError):
        logger.warning(
            f"[rid={rid8}] {name}={s!r} not a valid int, fallback to env"
        )
        return env_value, env_source
    if v < 0:
        logger.warning(
            f"[rid={rid8}] {name}={v} negative, fallback to env"
        )
        return env_value, env_source
    return v, "payload"


def _resolve_bool_flag(payload_value, env_value: bool, env_source: str, name: str, rid8: str):
    """Резолв payload > env > default для булевых параметров (true/1/yes/on)."""
    if payload_value is None:
        return env_value, env_source
    s = str(payload_value).strip().lower()
    if s == "":
        return env_value, env_source
    if s in ("true", "1", "yes", "on"):
        return True, "payload"
    if s in ("false", "0", "no", "off"):
        return False, "payload"
    logger.warning(
        f"[rid={rid8}] {name}={s!r} not a boolean, fallback to env"
    )
    return env_value, env_source
LOG_DIR = "./logs"

DOCLING_URL = os.getenv("DOCLING_URL")
GOTENBERG_URL = os.getenv("GOTENBERG_URL", "http://10.121.3.201:3004")

DEFAULT_VLM_URL = os.getenv("DEFAULT_VLM_URL")
DEFAULT_VLM_API_KEY = os.getenv("DEFAULT_VLM_API_KEY")
DEFAULT_VLM_MODEL = os.getenv("DEFAULT_VLM_MODEL")
DEFAULT_VLM_TIMEOUT = os.getenv("DEFAULT_VLM_TIMEOUT")
DEFAULT_VLM_CONCURRENCY = os.getenv("DEFAULT_VLM_CONCURRENCY")
DEFAULT_VLM_MAX_CONCURRENT_DOCS = os.getenv("DEFAULT_VLM_MAX_CONCURRENT_DOCS", "2")
DEFAULT_VLM_MAX_COMPLETION_TOKENS = int(os.getenv("DEFAULT_VLM_MAX_COMPLETION_TOKENS", "2048"))

# ── OCR SDK интеграция (v4.0) ──
OCR_SDK_URL = os.getenv("OCR_SDK_URL", "http://10.121.3.201:9996")
OCR_SDK_INBOX_CONTAINER = os.getenv("OCR_SDK_INBOX_CONTAINER", "/inbox")
OCR_SDK_ENABLED = os.getenv("OCR_SDK_ENABLED", "false").lower() == "true"
OCR_SDK_TIMEOUT = int(os.getenv("OCR_SDK_TIMEOUT", "600"))
ENRICH_PICTURES_WITH_122B = _env_bool("ENRICH_PICTURES_WITH_122B", default=True)

# ── Маршрутизация и качество рендеринга (настраивается из .env, override per-request через form-data) ──
# Порог страниц для TEXT PDF: <=порога → VLM full-page, >порога → standard+picture_description.
# 0 → TEXT PDF никогда не идёт в VLM (всегда standard).
_TEXT_PDF_VLM_THRESHOLD_DEFAULT = 20
TEXT_PDF_VLM_THRESHOLD = int(
    os.environ.get("TEXT_PDF_VLM_THRESHOLD")
    or _TEXT_PDF_VLM_THRESHOLD_DEFAULT
)
TEXT_PDF_VLM_THRESHOLD_SOURCE = "env" if os.environ.get("TEXT_PDF_VLM_THRESHOLD") else "default"

# Отправлять ли SCAN PDF в VLM full-page. False → SCAN идёт в standard
# pipeline (docling сам OCR'ит и режет на полигоны через picture
# description). По умолчанию True — VLM качественнее на сканах.
_SCAN_PDF_FULL_PAGE_DEFAULT = True
SCAN_PDF_FULL_PAGE = _env_bool("SCAN_PDF_FULL_PAGE", default=_SCAN_PDF_FULL_PAGE_DEFAULT)
SCAN_PDF_FULL_PAGE_SOURCE = "env" if os.environ.get("SCAN_PDF_FULL_PAGE") else "default"
# scale для VLM-пайплайнов (build_vlm_pipeline_model_api и build_custom_model).
DEFAULT_VLM_SCALE = float(os.environ.get("DEFAULT_VLM_SCALE", "1.5"))
# images_scale для standard-пайплайна: реально влияет на разрешение картинок,
# отправляемых в picture_description_api. Linear: 1.0 ≈ 36 DPI, 2.0 ≈ 96 DPI (A4).
DEFAULT_IMAGES_SCALE = float(os.environ.get("DEFAULT_IMAGES_SCALE", "2.0"))

# ── PostgreSQL-статистика (fire-and-forget, не влияет на latency) ──
# При STATS_ENABLED=false весь блок выключен: очередь/пул/worker не создаются,
# накладных расходов ноль (см. README-STATS.md).
STATS_ENABLED = os.getenv("STATS_ENABLED", "false").lower() == "true"
STATS_DB_DSN = os.getenv("STATS_DB_DSN", "")
STATS_QUEUE_SIZE = int(os.getenv("STATS_QUEUE_SIZE", "10000"))
STATS_BATCH_SIZE = int(os.getenv("STATS_BATCH_SIZE", "50"))
STATS_FLUSH_INTERVAL_SEC = float(os.getenv("STATS_FLUSH_INTERVAL_SEC", "5"))

# ── Retry на upstream-сбои docling (только httpx-исключения и 502/503/504) ──
# Не ретраим 500, 4xx и partial_success: первое часто воспроизводимо, последнее
# держит семафор впустую. Retry проходит ВНУТРИ семафора, чтобы общий cap
# vlm_max_concurrent_docs не пробивался.
DOCLING_RETRY_MAX_ATTEMPTS = int(os.getenv("DOCLING_RETRY_MAX_ATTEMPTS", "2"))
DOCLING_RETRY_BACKOFF_SEC = float(os.getenv("DOCLING_RETRY_BACKOFF_SEC", "1.0"))

# ── VLM proxy gateway (sampling injection + truncate analytics) ──
# Прокси выступает gateway'ем между docling-serve и LiteLLM/SGLang. Когда
# VLM_PROXY_ENABLED=true, build_*() инжектируют в конфиги docling-serve
# url=VLM_PROXY_URL?profile=... , и docling-serve ходит сначала сюда.
# Прокси добавляет sampling-параметры из соответствующего профиля,
# системный промпт, считает truncate-кейсы и форвардит в upstream
# (LiteLLM). Подробнее — README.md, раздел «VLM endpoint».
VLM_PROXY_ENABLED = os.getenv("VLM_PROXY_ENABLED", "false").lower() == "true"
VLM_PROXY_URL = os.getenv("VLM_PROXY_URL", "").strip()
# Upstream для самого прокси. Backwards-compat: если VLM_UPSTREAM_URL не
# задан — используем DEFAULT_VLM_URL/DEFAULT_VLM_API_KEY (LiteLLM).
VLM_UPSTREAM_URL = os.getenv("VLM_UPSTREAM_URL", "").strip() or DEFAULT_VLM_URL
VLM_UPSTREAM_API_KEY = (
    os.getenv("VLM_UPSTREAM_API_KEY", "").strip() or DEFAULT_VLM_API_KEY
)

VLM_REQUEST_LOG_FILE = os.getenv("VLM_REQUEST_LOG_FILE", "./logs/vlm_requests.jsonl")
VLM_TRUNCATE_LOG_DIR = os.getenv("VLM_TRUNCATE_LOG_DIR", "./logs/truncated")
VLM_TRUNCATE_SAVE_PAYLOAD = os.getenv("VLM_TRUNCATE_SAVE_PAYLOAD", "true").lower() == "true"
VLM_TRUNCATE_RETENTION_DAYS = int(os.getenv("VLM_TRUNCATE_RETENTION_DAYS", "30"))

# ── Retention для лог-файлов и диагностических дампов ──
# Все пути крутятся в одном LOG_DIR (или каталоге VLM_REQUEST_LOG_FILE).
# Retention в днях; vlm_requests_*.jsonl ещё ограничен размером
# суммарно по каталогу — страховка от вечного роста счётчика.
VLM_REQUEST_LOG_RETENTION_DAYS = int(os.getenv("VLM_REQUEST_LOG_RETENTION_DAYS", "90"))
VLM_REQUEST_LOG_MAX_SIZE_MB = int(os.getenv("VLM_REQUEST_LOG_MAX_SIZE_MB", "5120"))
NULL_RESPONSE_LOG_RETENTION_DAYS = int(os.getenv("NULL_RESPONSE_LOG_RETENTION_DAYS", "30"))
ERROR_RESPONSE_LOG_RETENTION_DAYS = int(os.getenv("ERROR_RESPONSE_LOG_RETENTION_DAYS", "30"))
def _load_sampling_profile(prefix: str) -> dict:
    """Собрать словарь sampling-параметров из ENV.

    Все параметры опциональные: пустой ENV → ключ не попадает в результат
    → upstream получает значение по умолчанию (SGLang/LiteLLM). Это даёт
    возможность отключать конкретный параметр без правки кода.
    """
    spec = {
        "TEMPERATURE":        ("temperature",        float),
        "TOP_P":              ("top_p",              float),
        "TOP_K":              ("top_k",              int),
        "MIN_P":              ("min_p",              float),
        "PRESENCE_PENALTY":   ("presence_penalty",   float),
        "REPETITION_PENALTY": ("repetition_penalty", float),
        "MAX_TOKENS":         ("max_tokens",         int),
    }
    out: dict = {}
    for env_key, (api_key, caster) in spec.items():
        raw = os.getenv(f"{prefix}_{env_key}", "").strip()
        if raw == "":
            continue
        try:
            out[api_key] = caster(raw)
        except (TypeError, ValueError):
            logger.warning(
                f"VLM sampling: bad value for {prefix}_{env_key}={raw!r}, skipped"
            )
    return out


VLM_FULL_PAGE_SAMPLING = _load_sampling_profile("VLM_FULL_PAGE")
VLM_PICTURE_DESC_SAMPLING = _load_sampling_profile("VLM_PICTURE_DESC")

ENRICH_LABELS = {"image", "chart", "engineering_drawing", "cad_drawing", "electrical_diagram", "seal", "stamp"}
