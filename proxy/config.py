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


def _env_int(name: str, default: int) -> int:
    """Безопасный int ENV. Отсутствие / пустая строка / invalid → default + WARNING.

    Используем для всех int-ENV — особенно когда переменная пробрасывается через
    docker-compose `${VAR}` без значения в `.env` (там Docker подставляет пустую
    строку, и os.getenv(..., default) возвращает '' вместо default → int('') → ValueError).
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        logger.warning(f"[config] {name}={raw!r} not a valid int, using default={default}")
        return default


def _env_float(name: str, default: float) -> float:
    """Безопасный float ENV. Отсутствие / пустая строка / invalid → default + WARNING."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        logger.warning(f"[config] {name}={raw!r} not a valid float, using default={default}")
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


def _env_int_or_zero(name: str, default: str) -> int:
    """Parse non-negative int env. Empty/'0'/invalid -> 0 (= disabled).

    Особая семантика: 0 означает «функционал отключён» (используется для
    target_pixels). Для обычных int-ENV (timeout/retention/queue/batch и т.п.)
    предпочитайте _env_int(name, default) — он возвращает явный default при
    пустой строке, а не 0.
    """
    raw = os.getenv(name, default).strip()
    if raw == "":
        return 0
    try:
        v = int(raw)
        return v if v >= 0 else 0
    except (TypeError, ValueError):
        logger.warning(f"[config] {name}={raw!r} not a valid int, using 0 (disabled)")
        return 0


LOG_DIR = "./logs"

DOCLING_URL = os.getenv("DOCLING_URL")
GOTENBERG_URL = os.getenv("GOTENBERG_URL", "http://10.121.3.201:3004")

DEFAULT_VLM_URL = os.getenv("DEFAULT_VLM_URL")
DEFAULT_VLM_API_KEY = os.getenv("DEFAULT_VLM_API_KEY")
DEFAULT_VLM_MODEL = os.getenv("DEFAULT_VLM_MODEL")
DEFAULT_VLM_TIMEOUT = os.getenv("DEFAULT_VLM_TIMEOUT")
DEFAULT_VLM_CONCURRENCY = os.getenv("DEFAULT_VLM_CONCURRENCY")
DEFAULT_VLM_MAX_CONCURRENT_DOCS = os.getenv("DEFAULT_VLM_MAX_CONCURRENT_DOCS", "2")
DEFAULT_VLM_MAX_COMPLETION_TOKENS = _env_int("DEFAULT_VLM_MAX_COMPLETION_TOKENS", 2048)

# ── OCR SDK интеграция (v4.0) ──
OCR_SDK_URL = os.getenv("OCR_SDK_URL", "http://10.121.3.201:9996")
OCR_SDK_INBOX_CONTAINER = os.getenv("OCR_SDK_INBOX_CONTAINER", "/inbox")
OCR_SDK_ENABLED = _env_bool("OCR_SDK_ENABLED", default=False)
OCR_SDK_TIMEOUT = _env_int("OCR_SDK_TIMEOUT", 600)
ENRICH_PICTURES_WITH_122B = _env_bool("ENRICH_PICTURES_WITH_122B", default=True)

# ── Маршрутизация и качество рендеринга (настраивается из .env, override per-request через form-data) ──
# Порог страниц для TEXT PDF: <=порога → VLM full-page, >порога → standard+picture_description.
# 0 → TEXT PDF никогда не идёт в VLM (всегда standard).
_TEXT_PDF_VLM_THRESHOLD_DEFAULT = 20
TEXT_PDF_VLM_THRESHOLD = _env_int("TEXT_PDF_VLM_THRESHOLD", _TEXT_PDF_VLM_THRESHOLD_DEFAULT)
TEXT_PDF_VLM_THRESHOLD_SOURCE = "env" if (os.environ.get("TEXT_PDF_VLM_THRESHOLD") or "").strip() else "default"

# Отправлять ли SCAN PDF в VLM full-page. False → SCAN идёт в standard
# pipeline (docling сам OCR'ит и режет на полигоны через picture
# description). По умолчанию True — VLM качественнее на сканах.
_SCAN_PDF_FULL_PAGE_DEFAULT = True
SCAN_PDF_FULL_PAGE = _env_bool("SCAN_PDF_FULL_PAGE", default=_SCAN_PDF_FULL_PAGE_DEFAULT)
SCAN_PDF_FULL_PAGE_SOURCE = "env" if (os.environ.get("SCAN_PDF_FULL_PAGE") or "").strip() else "default"

# ── Детектор скана (is_scan_pdf) ──
# Метрика "scan vs text": считаем БУКВЫ (alphabetic, unicode — кириллица/
# латиница), а не сырую длину. Причина: PDF из "Print To PDF" с телом в
# CID-шрифтах без ToUnicode отдаёт через extractable-слой только цифры/
# реквизиты (из шрифта с рабочей кодировкой) — по сырой длине это проходит
# порог и маскирует неизвлекаемое тело → документ ошибочно уходит в standard
# вместо распознавания. Буквы на таких страницах ≈ 0 → корректно ловится скан.
# Сэмплируем равномерно по всему документу, а не только первые страницы.
#
# SCAN_TEXT_METRIC: letters (по умолчанию) | chars (legacy, мгновенный откат
# к старому поведению по сырой длине через .env, без отката кода).
SCAN_TEXT_METRIC = (os.getenv("SCAN_TEXT_METRIC", "letters").strip().lower() or "letters")
SCAN_MIN_LETTERS_PER_PAGE = _env_int("SCAN_MIN_LETTERS_PER_PAGE", 50)
SCAN_MIN_CHARS_PER_PAGE = _env_int("SCAN_MIN_CHARS_PER_PAGE", 100)  # для legacy-режима chars
SCAN_DETECT_PAGES = _env_int("SCAN_DETECT_PAGES", 10)  # сколько страниц сэмплировать (равномерно)

# scale для VLM-пайплайнов (build_vlm_pipeline_model_api и build_custom_model).
DEFAULT_VLM_SCALE = _env_float("DEFAULT_VLM_SCALE", 1.5)
# images_scale для standard-пайплайна: реально влияет на разрешение картинок,
# отправляемых в picture_description_api. Linear: 1.0 ≈ 36 DPI, 2.0 ≈ 96 DPI (A4).
DEFAULT_IMAGES_SCALE = _env_float("DEFAULT_IMAGES_SCALE", 2.0)

# ── PostgreSQL-статистика (fire-and-forget, не влияет на latency) ──
# При STATS_ENABLED=false весь блок выключен: очередь/пул/worker не создаются,
# накладных расходов ноль (см. README-STATS.md).
STATS_ENABLED = _env_bool("STATS_ENABLED", default=False)
STATS_DB_DSN = os.getenv("STATS_DB_DSN", "")
STATS_QUEUE_SIZE = _env_int("STATS_QUEUE_SIZE", 10000)
STATS_BATCH_SIZE = _env_int("STATS_BATCH_SIZE", 50)
STATS_FLUSH_INTERVAL_SEC = _env_float("STATS_FLUSH_INTERVAL_SEC", 5.0)

# ── Retry на upstream-сбои docling (только httpx-исключения и 502/503/504) ──
# Не ретраим 500, 4xx и partial_success: первое часто воспроизводимо, последнее
# держит семафор впустую. Retry проходит ВНУТРИ семафора, чтобы общий cap
# vlm_max_concurrent_docs не пробивался.
DOCLING_RETRY_MAX_ATTEMPTS = _env_int("DOCLING_RETRY_MAX_ATTEMPTS", 2)
DOCLING_RETRY_BACKOFF_SEC = _env_float("DOCLING_RETRY_BACKOFF_SEC", 1.0)

# ── VLM proxy gateway (sampling injection + truncate analytics) ──
# Прокси выступает gateway'ем между docling-serve и LiteLLM/SGLang. Когда
# VLM_PROXY_ENABLED=true, build_*() инжектируют в конфиги docling-serve
# url=VLM_PROXY_URL?profile=... , и docling-serve ходит сначала сюда.
# Прокси добавляет sampling-параметры из соответствующего профиля,
# системный промпт, считает truncate-кейсы и форвардит в upstream
# (LiteLLM). Подробнее — README.md, раздел «VLM endpoint».
VLM_PROXY_ENABLED = _env_bool("VLM_PROXY_ENABLED", default=False)
VLM_PROXY_URL = os.getenv("VLM_PROXY_URL", "").strip()
# Upstream для самого прокси. Backwards-compat: если VLM_UPSTREAM_URL не
# задан — используем DEFAULT_VLM_URL/DEFAULT_VLM_API_KEY (LiteLLM).
VLM_UPSTREAM_URL = os.getenv("VLM_UPSTREAM_URL", "").strip() or DEFAULT_VLM_URL
VLM_UPSTREAM_API_KEY = (
    os.getenv("VLM_UPSTREAM_API_KEY", "").strip() or DEFAULT_VLM_API_KEY
)

VLM_REQUEST_LOG_FILE = os.getenv("VLM_REQUEST_LOG_FILE", "./logs/vlm_requests.jsonl")
VLM_TRUNCATE_LOG_DIR = os.getenv("VLM_TRUNCATE_LOG_DIR", "./logs/truncated")
VLM_TRUNCATE_SAVE_PAYLOAD = _env_bool("VLM_TRUNCATE_SAVE_PAYLOAD", default=True)
VLM_TRUNCATE_RETENTION_DAYS = _env_int("VLM_TRUNCATE_RETENTION_DAYS", 30)

# ── Retention для лог-файлов и диагностических дампов ──
VLM_REQUEST_LOG_RETENTION_DAYS = _env_int("VLM_REQUEST_LOG_RETENTION_DAYS", 90)
VLM_REQUEST_LOG_MAX_SIZE_MB = _env_int("VLM_REQUEST_LOG_MAX_SIZE_MB", 5120)
NULL_RESPONSE_LOG_RETENTION_DAYS = _env_int("NULL_RESPONSE_LOG_RETENTION_DAYS", 30)
ERROR_RESPONSE_LOG_RETENTION_DAYS = _env_int("ERROR_RESPONSE_LOG_RETENTION_DAYS", 30)


# ── Adaptive image resize before upstream forward ──
# Каждое изображение в payload /v1/chat/completions перед форвардом в LiteLLM/SGLang
# приводится к target_pixels (по площади, с сохранением aspect ratio через LANCZOS).
# Маленькие изображения (< VLM_MIN_PIXELS) проходят без изменений — модель сама
# апскейлит до своих минимумов. 0 или пусто = ресайз отключён для профиля.
# Эмпирический оптимум для Qwen3.5-VL-122B — 950000 px (см. отчёт «Устранение
# лупов» 25.04.2026, раздел 5).
#
# Используем _env_int_or_zero — у target_pixels особая семантика: 0=disabled
# (а не «как пишет код»). Для VLM_MIN_PIXELS — обычный _env_int с явным
# default'ом, иначе при пустой строке min_pixels превратится в 0 и ресайзить
# начнёт даже крошечные иконки.
VLM_FULL_PAGE_TARGET_PIXELS = _env_int_or_zero("VLM_FULL_PAGE_TARGET_PIXELS", "950000")
VLM_PICTURE_DESC_TARGET_PIXELS = _env_int_or_zero("VLM_PICTURE_DESC_TARGET_PIXELS", "950000")
VLM_MIN_PIXELS = _env_int("VLM_MIN_PIXELS", 200704)

# Опционально: при truncate-дампе сохранять и оригинал картинки (до ресайза)
# рядом с фактически отправленной — для разбора.
VLM_TRUNCATE_SAVE_ORIGINAL_IMAGES = _env_bool("VLM_TRUNCATE_SAVE_ORIGINAL_IMAGES", default=False)


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

# ── Адаптивный контроль исходящих VLM-запросов (capacity gate) ──
# Gateway придерживает форвард в LiteLLM/vLLM ровно настолько, насколько занята
# модель — сигнал берём из Prometheus /metrics инстанса vLLM. Цель: грузить
# модель до реальной ёмкости (есть слот — занимаем), но не растить хвост в
# очереди самого vLLM. Подробно — README.md «Adaptive VLM gate».
#
# Флаг отключения: VLM_GATE_ENABLED=false → форвард без ожидания (поведение 1:1
# как до фичи), безопасный rollback.
VLM_GATE_ENABLED = _env_bool("VLM_GATE_ENABLED", default=True)
# Источник сигнала ёмкости. URL напрямую на vLLM (НЕ LiteLLM :4000 — у него
# свой формат). Auth не нужен — /metrics открыт.
VLM_METRICS_URL = os.getenv("VLM_METRICS_URL", "http://10.121.3.190:9989/metrics")
# Имена метрик (vLLM). Внимание: в этой версии vLLM именно kv_cache_usage_perc,
# а не gpu_cache_usage_perc. Вынесены в ENV на случай смены движка/версии.
VLM_METRIC_RUNNING = os.getenv("VLM_METRIC_RUNNING", "vllm:num_requests_running")
VLM_METRIC_WAITING = os.getenv("VLM_METRIC_WAITING", "vllm:num_requests_waiting")
VLM_METRIC_KV = os.getenv("VLM_METRIC_KV", "vllm:kv_cache_usage_perc")
# Период опроса /metrics (мс) и таймаут одного запроса (мс).
VLM_METRICS_POLL_MS = _env_int("VLM_METRICS_POLL_MS", 300)
VLM_METRICS_TIMEOUT_MS = _env_int("VLM_METRICS_TIMEOUT_MS", 1000)
# Возраст снапшота (мс), после которого данные считаем устаревшими и
# переходим в fallback-режим (локальный кап).
VLM_METRICS_STALE_MS = _env_int("VLM_METRICS_STALE_MS", 2000)
# Порог KV-cache (доля 0..1): пока eff_kv < порога — есть запас.
VLM_GATE_KV_THRESHOLD = _env_float("VLM_GATE_KV_THRESHOLD", 0.85)
# Допускаем, пока waiting <= W. W=0 — строго без backlog; 2–4 держит GPU
# «накормленным» на максимум throughput. Тюнится по /metrics.
VLM_GATE_WAITING_MAX = _env_int("VLM_GATE_WAITING_MAX", 0)
# Оценка KV/запрос, когда running==0 (нечем самокалиброваться). Анти-овершут
# для холодного старта пачки.
VLM_GATE_DEFAULT_PER_REQ_KV = _env_float("VLM_GATE_DEFAULT_PER_REQ_KV", 0.02)
# Fallback-кап одновременных in-flight, когда метрики недоступны/устарели.
# НЕ постоянный потолок: работает только в режиме «метрики недоступны».
VLM_GATE_FALLBACK_MAX_INFLIGHT = _env_int("VLM_GATE_FALLBACK_MAX_INFLIGHT", 24)
# Бюджет ожидания в гейте (сек) и доля, после которой — last-resort форвард
# (страницу не дропаем; пусть уйдёт в очередь vLLM + WARNING-сигнал перегрузки).
# Бюджет согласован с таймаутом docling→gateway (DEFAULT_VLM_TIMEOUT): держим
# меньше, иначе таймаут переедет сюда.
VLM_GATE_WAIT_BUDGET_SEC = _env_float("VLM_GATE_WAIT_BUDGET_SEC", 600.0)
VLM_GATE_LAST_RESORT_FRACTION = _env_float("VLM_GATE_LAST_RESORT_FRACTION", 0.8)
# Как часто ждущий перепроверяет ёмкость, если его не разбудили событием (мс).
VLM_GATE_RECHECK_MS = _env_int("VLM_GATE_RECHECK_MS", 100)

ENRICH_LABELS = {"image", "chart", "engineering_drawing", "cad_drawing", "electrical_diagram", "seal", "stamp"}

# ── Распаковка архивов (zip / tar / 7z / rar, рекурсивно) ──
# Workaround поверх docling: docling-serve не умеет принимать архив и
# обрабатывать его содержимое. Прокси разворачивает архив (включая вложенные),
# прогоняет каждый поддерживаемый документ через обычный пайплайн обработки
# и склеивает результат в один markdown. Подробно — README.md, раздел «Архивы».
#
# zip и tar-семейство (tar/tar.gz/tgz/tar.bz2/tar.xz) — через stdlib, без
# доп. зависимостей. 7z требует py7zr, rar — rarfile + системный бинарник
# unrar/bsdtar. Если библиотека/бинарник недоступны — соответствующий архив
# помечается как необработанный (не падаем).
ARCHIVE_PROCESSING_ENABLED = _env_bool("ARCHIVE_PROCESSING_ENABLED", default=True)
# Максимальная глубина вложенности (архив в архиве). Защита от рекурсивных бомб.
ARCHIVE_MAX_DEPTH = _env_int("ARCHIVE_MAX_DEPTH", 5)
# Максимальное число извлекаемых файлов суммарно по всему дереву.
ARCHIVE_MAX_FILES = _env_int("ARCHIVE_MAX_FILES", 200)
# Максимальный суммарный распакованный объём (защита от zip-бомб), МБ.
ARCHIVE_MAX_TOTAL_MB = _env_int("ARCHIVE_MAX_TOTAL_MB", 500)
ARCHIVE_MAX_TOTAL_BYTES = ARCHIVE_MAX_TOTAL_MB * 1024 * 1024
