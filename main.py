import time
import fitz  # PyMuPDF
from fastapi import FastAPI, Request, Response
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from datetime import datetime as dt
import os, json, httpx, asyncio, zipfile, uuid, base64, re, glob, logging, sys, traceback
import xml.etree.ElementTree as ET

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# Логирование
# ═══════════════════════════════════════════════════════════════
# LOG_LEVEL — DEBUG/INFO/WARNING/ERROR, default INFO.
# LOG_FORMAT — text (default) или json для агрегаторов (Loki, ELK).
# uvicorn/FastAPI-логгеры не трогаем — у них свой формат, их не глушим.

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

# ═══════════════════════════════════════════════════════════════
# Глобальные переменные из .env
# ═══════════════════════════════════════════════════════════════
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
TEXT_PDF_VLM_THRESHOLD = int(os.environ.get("TEXT_PDF_VLM_THRESHOLD", "20"))
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

DEFAULT_VLM_PROMPT = (
    "Проанализируй это изображение из документа. Выполни ОБА шага:\n\n"
    "1. ТЕКСТ: Если на изображении есть текст (подписи, заголовки, метки, числа, "
    "водяные знаки) — извлеки его ПОЛНОСТЬЮ и ТОЧНО. Для каждого блока текста укажи, "
    "ГДЕ он расположен и К ЧЕМУ относится (например: «надпись на левой табличке», "
    "«заголовок графика», «подпись под осью X»). Сохрани структуру. Нечитаемые символы "
    "замени на «?».\n\n"
    "2. ОПИСАНИЕ: Кратко опиши визуальное содержимое — тип изображения (график, схема, "
    "фото, таблица, диаграмма, скриншот), расположение ключевых элементов и их взаимосвязь.\n\n"
    "Отвечай на русском. Будь точным — не пропускай информацию и не выдумывай."
)

# Промпт для описания отдельных регионов (OCR SDK enrichment).
# Отличается от DEFAULT_VLM_PROMPT тем, что допускает лаконичные ответы на
# типовые случаи (подпись/штамп/пусто) и прямо запрещает достраивать
# обрезанный текст — защита от галлюцинаций на неточных кропах.
ENRICH_VLM_PROMPT = (
    "Опиши кратко, что изображено на этом фрагменте документа. Фрагмент мог быть "
    "вырезан неточно и содержать только часть элемента, пустое поле или фоновый шум.\n\n"
    "ПРАВИЛА:\n"
    "• Рукописная подпись или росчерк — ответь: «Рукописная подпись».\n"
    "• Штамп или печать — прочитай видимый текст штампа; если читаемого текста нет — "
    "«Штамп без читаемого текста».\n"
    "• Логотип — назови организацию, если узнаёшь её по начертанию; иначе «Логотип».\n"
    "• График, диаграмма, схема, чертёж — опиши максимально подробно: "
    "тип, оси и единицы измерения, легенду, заголовок, ключевые числовые значения, "
    "штамп чертежа (организация, номер, дата, подписи), спецификацию, "
    "позиции деталей, размеры, допуски, соединения, сечения, виды.\n"
    "• Фотография или иллюстрация — опиши сюжет и ключевые объекты.\n"
    "• Пустое поле, фон, декоративная линия, рамка, штрихкод без читаемых цифр, "
    "одиночный росчерк без контекста — ответь одним словом: «Пусто».\n\n"
    "ЗАПРЕТЫ:\n"
    "• НЕ достраивай обрезанные слова. Если видишь фрагмент текста — "
    "приведи ровно то что видно, в кавычках, без предположений о полном слове.\n"
    "• НЕ описывай декоративные элементы (линии, маркеры, отступы, пустое место).\n"
    "• НЕ добавляй вступлений вида «На изображении видно...».\n\n"
    "Отвечай на русском, строго по факту."
)

DEFAULT_VLM_PIPELINE_PROMPT = (
    "Твоя задача: извлечь содержимое страницы документа в формате Markdown "
    "и завершить ответ маркером `[END_OF_PAGE]`.\n\n"
    "Правила извлечения:\n"
    "- Сохраняй язык оригинала и структуру (заголовки, списки, таблицы).\n"
    "- В таблицах выводи только строки, где есть хотя бы одна заполненная "
    "ячейка. Полностью пустые строки пропускай — это нужно, чтобы результат "
    "оставался компактным.\n"
    "- Декоративные элементы (рамки листа, линии разметки, водяные знаки) "
    "не описывай — они не несут информации.\n"
    "- Для инженерных чертежей извлекай: заполненные поля штампа "
    "(организация, номер чертежа, наименование, дата, подписи); позиции "
    "спецификации (формат, зона, позиция, обозначение, наименование, "
    "количество); ключевые размеры и обозначения.\n"
    "- Для графиков, схем и диаграмм — 2–4 предложения в квадратных "
    "скобках: что изображено, оси и единицы, ключевые значения.\n\n"
    "Финальное правило (обязательное): после того как весь осмысленный "
    "контент извлечён, выведи на отдельной новой строке ровно такой маркер:\n\n"
    "[END_OF_PAGE]\n\n"
    "Формат маркера: открывающая квадратная скобка `[`, затем `END_OF_PAGE` "
    "заглавными латинскими буквами с подчёркиваниями, затем закрывающая "
    "квадратная скобка `]`. Никаких пробелов внутри, никаких угловых "
    "скобок. После маркера ничего не пиши."
)

# Системный промпт для VLM endpoint, профиль full_page (полностраничный
# OCR через /v1/chat/completions?profile=full_page). Зафиксирован после
# калибровки sampling Qwen3.5-122B-GPTQ-Int4 в апреле 2026 — содержит
# reframing'и таблиц, защиту от описательного loop'а на пустых ячейках,
# инструкции по инженерным чертежам и /no_think в конце для надёжного
# отключения thinking mode на сторону модели.
DEFAULT_VLM_FULL_PAGE_PROMPT = (
    "Извлеки содержимое страницы в формате Markdown, сохраняя язык оригинала.\n\n"
    "Текст:\n"
    "- Копируй дословно, без перевода и перефразирования.\n"
    "- Сохраняй структуру: заголовки, абзацы, списки, цитаты, подписи под изображениями, колонтитулы.\n"
    "- Многоколоночный текст читай по колонкам, сверху вниз слева направо.\n\n"
    "Формулы:\n"
    "- Математические формулы выводи в LaTeX: $формула$ для inline и $$формула$$ для блочных.\n\n"
    "Визуальный контент — описывай в квадратных скобках, только если он несёт информацию:\n"
    "- Фотографии, иллюстрации, графики, диаграммы, схемы, чертежи, карты — 2–4 предложения: "
    "что изображено, ключевые объекты, оси и значения для графиков, компоненты и связи для схем.\n"
    "- Инженерные чертежи — дополнительно извлекай заполненные поля штампа "
    "(организация, номер, наименование, дата, подписи) и позиции спецификации.\n"
    "- Логотипы, печати, штампы — одна строка с описанием.\n"
    "- Декоративные элементы (виньетки, орнаменты, разделители глав, буквицы, узорные рамки, "
    "фоновые водяные знаки, разлиновка по ГОСТ) — пропускай, не описывай.\n\n"
    "Если страница пустая или содержит только декор — верни пустой ответ.\n\n"
    "Таблицы — это данные, а не визуальная сетка:\n"
    "- Читай таблицу строго построчно, сверху вниз, слева направо. Не перескакивай по столбцам.\n"
    "- После последней строки с данными остановись. Пустые строки внизу таблицы "
    "(незаполненная сетка) — это место для будущих записей, их не выводи и не достраивай.\n"
    "- Столбцы, пустые во всех строках, исключай из вывода целиком.\n"
    "- Если столбец имеет подзаголовки (например, `Кол.` делится на `-` и `-01`), выводи их "
    "как отдельные столбцы с именами `Кол.(-)` и `Кол.(-01)`.\n"
    "- Если строка содержит подчёркнутый или курсивный заголовок-разделитель раздела "
    "(например, «Документация», «Сборочные единицы», «Стандартные изделия», «Материалы», "
    "«Прочие изделия», «Комплекты») — выводи его как отдельную строку `### Заголовок` перед "
    "продолжением таблицы, не как ячейку таблицы.\n"
    "- Если в столбце `Обозначение` или `Наименование` запись занимает две строки сетки "
    "(например, название на первой строке и ГОСТ/ТУ на второй) — объединяй обе в одну ячейку "
    "через перенос строки `<br>` или пробел.\n"
    "- Строки, где не заполнена ни одна ячейка, не выводи.\n"
    "- Если в строке заполнена только одна ячейка — выводи её не как строку таблицы, "
    "а пунктом списка: `- значение` или `- поле: значение`.\n\n"
    "/no_think"
)

# Системный промпт для VLM endpoint, профиль picture_desc (описание
# вырезанного региона/картинки в standard pipeline). Совпадает с
# DEFAULT_VLM_PROMPT — он адекватен для описания регионов и не требует
# изменений по итогам калибровки апреля 2026.
DEFAULT_VLM_PICTURE_DESC_PROMPT = DEFAULT_VLM_PROMPT


def _load_prompt_from_file(env_var: str, default: str) -> str:
    """Прочитать промпт из файла по пути из ENV. Пустой/несуществующий → default."""
    path = os.getenv(env_var, "").strip()
    if not path:
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            logger.warning(f"VLM prompt: file {path} empty, falling back to default")
            return default
        logger.info(f"VLM prompt: loaded {env_var} from {path} ({len(content)} chars)")
        return content
    except Exception as e:
        logger.warning(f"VLM prompt: cannot read {env_var}={path}: {e}; using default")
        return default


VLM_FULL_PAGE_PROMPT = _load_prompt_from_file(
    "VLM_PROMPT_FULL_PAGE_FILE", DEFAULT_VLM_FULL_PAGE_PROMPT
)
VLM_PICTURE_DESC_PROMPT = _load_prompt_from_file(
    "VLM_PROMPT_PICTURE_DESC_FILE", DEFAULT_VLM_PICTURE_DESC_PROMPT
)


# ═══════════════════════════════════════════════════════════════
# httpx AsyncClient + inbox cleanup
# ═══════════════════════════════════════════════════════════════

def cleanup_old_inbox_files(max_age_seconds: int = 3600):
    """Delete files older than max_age_seconds from shared inbox. Safety net for leaked files."""
    inbox = OCR_SDK_INBOX_CONTAINER
    if not os.path.isdir(inbox):
        return
    now = time.time()
    count = 0
    for f in os.listdir(inbox):
        fp = os.path.join(inbox, f)
        try:
            if os.path.isfile(fp) and (now - os.path.getmtime(fp)) > max_age_seconds:
                os.remove(fp)
                count += 1
        except Exception:
            pass
    if count > 0:
        logger.info(f"[inbox cleanup] Removed {count} stale file(s) from {inbox}")


async def _periodic_inbox_cleanup():
    """Background task: clean inbox every 30 minutes."""
    while True:
        await asyncio.sleep(1800)
        try:
            cleanup_old_inbox_files()
        except Exception as e:
            logger.error(f"[inbox cleanup] Error: {e}")


# ═══════════════════════════════════════════════════════════════
# Статистика запросов в PostgreSQL (опциональная, fire-and-forget)
# ═══════════════════════════════════════════════════════════════
# Вся подсистема активна только при STATS_ENABLED=true и доступном
# PostgreSQL. Любая запись в БД — из фоновой корутины батчами. В
# горячем пути запроса блокирующих вызовов нет, при переполнении
# очереди события отбрасываются с warning. Ошибка записи никогда
# не вызывает HTTP-ошибку у клиента. Подробнее — README-STATS.md.

_STATS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS docling_requests (
    id                      BIGSERIAL PRIMARY KEY,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    request_id              UUID,
    filename                TEXT,
    file_size_bytes         BIGINT,
    file_pages              INTEGER,
    doc_type                TEXT,
    pipeline                TEXT,
    http_status             INTEGER,
    duration_total_ms       INTEGER,
    duration_docling_ms     INTEGER,
    duration_queue_wait_ms  INTEGER,
    timings_json            JSONB,
    error_message           TEXT,
    client_ip               TEXT,
    user_agent              TEXT
);
CREATE INDEX IF NOT EXISTS idx_docling_requests_created_at  ON docling_requests (created_at);
CREATE INDEX IF NOT EXISTS idx_docling_requests_doc_type    ON docling_requests (doc_type);
CREATE INDEX IF NOT EXISTS idx_docling_requests_pipeline    ON docling_requests (pipeline);
CREATE INDEX IF NOT EXISTS idx_docling_requests_http_status ON docling_requests (http_status);
"""


async def _stats_init_conn(conn):
    """Codec для JSONB — чтобы в executemany передавать dict напрямую."""
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )


async def _stats_ensure_schema(pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(_STATS_SCHEMA_SQL)


async def _stats_insert_batch(pool, rows: list) -> None:
    query = """
        INSERT INTO docling_requests (
            created_at, request_id, filename, file_size_bytes, file_pages,
            doc_type, pipeline, http_status, duration_total_ms,
            duration_docling_ms, duration_queue_wait_ms, timings_json,
            error_message, client_ip, user_agent
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
    """
    args = [
        (
            r.get("created_at"),
            r.get("request_id"),
            r.get("filename"),
            r.get("file_size_bytes"),
            r.get("file_pages"),
            r.get("doc_type"),
            r.get("pipeline"),
            r.get("http_status"),
            r.get("duration_total_ms"),
            r.get("duration_docling_ms"),
            r.get("duration_queue_wait_ms"),
            r.get("timings_json"),
            r.get("error_message"),
            r.get("client_ip"),
            r.get("user_agent"),
        )
        for r in rows
    ]
    async with pool.acquire() as conn:
        await conn.executemany(query, args)


async def _stats_worker(app: FastAPI) -> None:
    """Корутина-воркер: вытаскивает записи из очереди и пишет пачками.

    Трижды корректная: на полный батч, по таймауту flush и при ошибке БД
    — всегда чистит batch и сдвигает last_flush, чтобы не крутиться в
    бесконечном re-try над одними и теми же записями.
    """
    pool = app.state.stats_pool
    queue: asyncio.Queue = app.state.stats_queue
    metrics = app.state.stats_metrics
    batch: list = []
    last_flush = time.time()
    while True:
        remaining = STATS_FLUSH_INTERVAL_SEC - (time.time() - last_flush)
        try:
            if remaining <= 0:
                raise asyncio.TimeoutError
            item = await asyncio.wait_for(queue.get(), timeout=remaining)
            batch.append(item)
            queue.task_done()
        except asyncio.TimeoutError:
            pass
        should_flush = batch and (
            len(batch) >= STATS_BATCH_SIZE
            or (time.time() - last_flush) >= STATS_FLUSH_INTERVAL_SEC
        )
        if should_flush:
            try:
                await _stats_insert_batch(pool, batch)
                metrics["written"] += len(batch)
            except Exception as e:
                metrics["db_errors"] += 1
                logger.warning(
                    f"Stats: batch insert failed ({e}); dropping {len(batch)} records"
                )
            batch = []
            last_flush = time.time()


async def _stats_metrics_loop(app: FastAPI) -> None:
    while True:
        await asyncio.sleep(60)
        m = app.state.stats_metrics
        q: asyncio.Queue = app.state.stats_queue
        logger.info(
            f"Stats: enqueued={m['enqueued']}, written={m['written']}, "
            f"dropped={m['dropped']}, db_errors={m['db_errors']}, "
            f"queue_size={q.qsize()}"
        )


def _stats_enqueue(app: FastAPI, record: dict) -> None:
    """Non-blocking enqueue. Полная очередь → drop + warning (не часто)."""
    if not STATS_ENABLED or getattr(app.state, "stats_queue", None) is None:
        return
    m = app.state.stats_metrics
    try:
        app.state.stats_queue.put_nowait(record)
        m["enqueued"] += 1
    except asyncio.QueueFull:
        m["dropped"] += 1
        if m["dropped"] == 1 or m["dropped"] % 100 == 0:
            logger.warning(
                f"Stats: queue full, dropping record (total dropped={m['dropped']})"
            )


def _stats_set(request: Request, **fields) -> None:
    """Положить метки в stats-контекст текущего запроса. Безопасно при OFF."""
    state = getattr(request, "state", None)
    bag = getattr(state, "stats", None) if state is not None else None
    if bag is None:
        return
    for k, v in fields.items():
        if v is not None:
            bag[k] = v


@asynccontextmanager
async def lifespan(app: FastAPI):
    # uvicorn ставит свои handlers через Config.configure_logging() ПОСЛЕ
    # импорта main.py — переопределяем здесь, к моменту первого запроса
    # access/error логи уже идут через наш formatter с timestamp + level.
    _retrofit_uvicorn_loggers()
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(1200.0, connect=10.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        follow_redirects=False,
    )
    cleanup_old_inbox_files()
    cleanup_task = asyncio.create_task(_periodic_inbox_cleanup())
    # Ротация vlm_requests_*.jsonl + null/error response дампов: при
    # старте и раз в сутки. Retention настраивается через ENV (см.
    # *_RETENTION_DAYS), есть размерный cap для vlm_requests.
    cleanup_old_vlm_request_logs()
    cleanup_old_null_response_logs()
    logs_cleanup_task = asyncio.create_task(_periodic_logs_cleanup())
    # Ротация truncate-дампов VLM endpoint'а: при старте и раз в сутки.
    cleanup_old_truncate_dumps()
    truncate_cleanup_task = asyncio.create_task(_periodic_truncate_dumps_cleanup())

    # ── Активные значения параметров конфигурации (для быстрой диагностики) ──
    logger.info(
        f"[CONFIG] TEXT_PDF_VLM_THRESHOLD={TEXT_PDF_VLM_THRESHOLD}, "
        f"DEFAULT_VLM_SCALE={DEFAULT_VLM_SCALE}, "
        f"DEFAULT_IMAGES_SCALE={DEFAULT_IMAGES_SCALE}"
    )
    logger.info(f"[config] ENRICH_PICTURES_WITH_122B = {ENRICH_PICTURES_WITH_122B}")
    logger.info(
        f"[logs-cleanup] vlm_requests retention={VLM_REQUEST_LOG_RETENTION_DAYS}d"
        f" size_cap={VLM_REQUEST_LOG_MAX_SIZE_MB}MB"
    )
    logger.info(
        f"[logs-cleanup] null_responses retention={NULL_RESPONSE_LOG_RETENTION_DAYS}d"
    )
    logger.info(
        f"[logs-cleanup] error_responses retention={ERROR_RESPONSE_LOG_RETENTION_DAYS}d"
    )
    logger.info(
        f"[logs-cleanup] truncate_dumps retention={VLM_TRUNCATE_RETENTION_DAYS}d"
    )
    # ── VLM endpoint конфиг (показываем без секретов) ──
    logger.info(
        f"[VLM-PROXY] enabled={VLM_PROXY_ENABLED} "
        f"proxy_url={VLM_PROXY_URL or '<unset>'} "
        f"upstream={VLM_UPSTREAM_URL} "
        f"truncate_dir={VLM_TRUNCATE_LOG_DIR} "
        f"save_payload={VLM_TRUNCATE_SAVE_PAYLOAD} "
        f"retention_days={VLM_TRUNCATE_RETENTION_DAYS}"
    )
    logger.info(
        f"[VLM-PROXY] full_page sampling={VLM_FULL_PAGE_SAMPLING} "
        f"prompt_chars={len(VLM_FULL_PAGE_PROMPT)}"
    )
    logger.info(
        f"[VLM-PROXY] picture_desc sampling={VLM_PICTURE_DESC_SAMPLING} "
        f"prompt_chars={len(VLM_PICTURE_DESC_PROMPT)}"
    )

    # ── Статистика ──
    app.state.stats_pool = None
    app.state.stats_queue = None
    app.state.stats_worker_task = None
    app.state.stats_metrics_task = None
    app.state.stats_metrics = {"enqueued": 0, "written": 0, "dropped": 0, "db_errors": 0}

    if STATS_ENABLED:
        try:
            import asyncpg  # ленивый импорт: без STATS_ENABLED не требуется
            if not STATS_DB_DSN:
                raise RuntimeError("STATS_DB_DSN is empty")
            app.state.stats_pool = await asyncpg.create_pool(
                dsn=STATS_DB_DSN, min_size=1, max_size=5, init=_stats_init_conn
            )
            await _stats_ensure_schema(app.state.stats_pool)
            app.state.stats_queue = asyncio.Queue(maxsize=STATS_QUEUE_SIZE)
            app.state.stats_worker_task = asyncio.create_task(_stats_worker(app))
            app.state.stats_metrics_task = asyncio.create_task(_stats_metrics_loop(app))
            logger.info(
                f"Stats collection enabled (queue={STATS_QUEUE_SIZE}, "
                f"batch={STATS_BATCH_SIZE}, flush={STATS_FLUSH_INTERVAL_SEC}s)"
            )
        except Exception as e:
            logger.error(f"Stats: init failed ({e}); continuing with stats disabled")
            app.state.stats_pool = None
            app.state.stats_queue = None
    else:
        logger.info("Stats collection disabled (STATS_ENABLED=false)")

    yield

    cleanup_task.cancel()
    logs_cleanup_task.cancel()
    truncate_cleanup_task.cancel()

    # Graceful shutdown stats: дать воркеру добить очередь в пределах 10с.
    if app.state.stats_queue is not None:
        try:
            await asyncio.wait_for(app.state.stats_queue.join(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning(
                f"Stats: queue drain timed out; "
                f"{app.state.stats_queue.qsize()} records lost"
            )
    for t in (app.state.stats_worker_task, app.state.stats_metrics_task):
        if t is not None:
            t.cancel()
    if app.state.stats_pool is not None:
        try:
            await app.state.stats_pool.close()
        except Exception as e:
            logger.warning(f"Stats: pool close error: {e}")

    await app.state.client.aclose()

app = FastAPI(lifespan=lifespan)


# Middleware: ровно одна точка enqueue на запрос. Handler по пути
# дополняет request.state.stats через _stats_set(...).
@app.middleware("http")
async def _stats_middleware(request: Request, call_next):
    if not STATS_ENABLED:
        return await call_next(request)
    # Инструментируем только основной multipart-эндпоинт конвертации.
    is_convert_file = (
        request.method == "POST"
        and request.url.path.rstrip("/").endswith("/convert/file")
    )
    if not is_convert_file:
        return await call_next(request)
    from datetime import timezone
    bag = {
        "created_at": dt.now(timezone.utc),
        "request_id": uuid.uuid4(),
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
    }
    request.state.stats = bag
    _t = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        bag["http_status"] = 500
        bag["error_message"] = f"{type(e).__name__}: {e}"
        bag["duration_total_ms"] = int((time.time() - _t) * 1000)
        _stats_enqueue(request.app, bag)
        raise
    bag.setdefault("http_status", response.status_code)
    bag.setdefault("duration_total_ms", int((time.time() - _t) * 1000))
    _stats_enqueue(request.app, bag)
    return response


# ═══════════════════════════════════════════════════════════════
# Семафор
# ═══════════════════════════════════════════════════════════════
_semaphore = None
_semaphore_value = 0

def get_semaphore(max_docs: int) -> asyncio.Semaphore:
    global _semaphore, _semaphore_value
    if _semaphore is None or _semaphore_value != max_docs:
        _semaphore = asyncio.Semaphore(max_docs)
        _semaphore_value = max_docs
        # Лог для диагностики: при workers>1 каждый воркер создаст свой
        # семафор и мы увидим в логах разные pid'ы — это и есть симптом
        # рассинхрона. При workers=1 pid должен быть один и тот же.
        logger.info(f"[semaphore] (re)created: max_docs={max_docs}  pid={os.getpid()}")
    return _semaphore


def save(data: list, files: list):
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    file_names = [f[1][0] for f in files] if files else []
    params = {"data": {k: v for k, v in data}, "files": file_names}
    filename = f"params_{dt.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    with open(os.path.join(LOG_DIR, filename), 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
# Автодетекция сканов
# ═══════════════════════════════════════════════════════════════

def is_scan_pdf(pdf_bytes: bytes, min_chars_per_page: int = 100, pages_to_check: int = 3) -> bool:
    """Check if PDF is a scan (no/little extractable text).

    Returns True if PDF appears to be a scanned document.
    Returns False for non-PDF files or PDFs with good text layer.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_checked = min(len(doc), pages_to_check)
        if pages_checked == 0:
            doc.close()
            return False
        total_chars = 0
        for i in range(pages_checked):
            text = doc[i].get_text().strip()
            total_chars += len(text)
        doc.close()
        avg_chars = total_chars / pages_checked
        return avg_chars < min_chars_per_page
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════
# Построение конфигов VLM
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# Детекция OLE-объектов в DOCX
# ═══════════════════════════════════════════════════════════════

def has_ole_objects(file_bytes: bytes, filename: str) -> bool:
    """Check if DOCX/PPTX contains OLE objects (MathType, Equation Editor, etc.)."""
    if not filename.lower().endswith((".docx", ".pptx")):
        return False
    try:
        import io
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            ole_files = [f for f in z.namelist() if "oleObject" in f or "embeddings/oleObject" in f]
            return len(ole_files) > 0
    except Exception:
        return False


async def convert_via_gotenberg(client: httpx.AsyncClient, file_bytes: bytes, filename: str) -> bytes:
    """Convert DOCX/PPTX to PDF via Gotenberg API."""
    gotenberg_url = f"{GOTENBERG_URL}/forms/libreoffice/convert"
    files = [("files", (filename, file_bytes, "application/octet-stream"))]
    resp = await client.post(gotenberg_url, files=files, timeout=120.0)
    if resp.status_code == 200:
        return resp.content
    else:
        raise Exception(f"Gotenberg conversion failed: HTTP {resp.status_code}")


# ═══════════════════════════════════════════════════════════════
# Детекция и обработка .DOC файлов
# ═══════════════════════════════════════════════════════════════

def is_confluence_doc(file_bytes: bytes) -> bool:
    """Check if .doc file is actually a Confluence MIME HTML export.
    
    Confluence exports .doc files that are really MIME-encoded HTML
    (Content-Type: text/html, Content-Transfer-Encoding: quoted-printable).
    """
    try:
        header = file_bytes[:2000].decode('utf-8', errors='ignore')
        if 'MIME-Version' in header and ('Content-Type' in header or 'boundary=' in header):
            return True
        if 'Exported From Confluence' in header:
            return True
        return False
    except Exception:
        return False


def decode_confluence_doc(file_bytes: bytes, filename: str) -> tuple:
    """Decode Confluence MIME HTML .doc to plain HTML.
    
    Returns (html_bytes, html_filename) or (None, None) on failure.
    """
    import email
    import quopri
    
    try:
        msg = email.message_from_bytes(file_bytes)
        
        # Multipart: find HTML part
        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                if ct == 'text/html':
                    payload = part.get_payload(decode=True)
                    if payload:
                        html_name = filename.rsplit('.', 1)[0] + '.html'
                        logger.info(f"Confluence decode: found HTML part ({len(payload)} bytes)")
                        return payload, html_name
        
        # Single part
        payload = msg.get_payload(decode=True)
        if payload:
            html_name = filename.rsplit('.', 1)[0] + '.html'
            return payload, html_name
        
        # Fallback: find <html in raw content and decode quopri
        raw = file_bytes.decode('utf-8', errors='ignore')
        for marker in ('<html', '<HTML', '<!DOCTYPE'):
            idx = raw.find(marker)
            if idx >= 0:
                html_part = raw[idx:]
                decoded = quopri.decodestring(html_part.encode('utf-8', errors='ignore'))
                html_name = filename.rsplit('.', 1)[0] + '.html'
                logger.info(f"Confluence decode: fallback quopri ({len(decoded)} bytes)")
                return decoded, html_name
        
        logger.info(f"Confluence decode: could not extract HTML from {filename}")
        return None, None
    except Exception as e:
        logger.error(f"Confluence decode ERROR: {e}")
        return None, None


def count_pdf_images(pdf_bytes: bytes) -> int:
    """Count total images across all pages of a PDF."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total = 0
        for page in doc:
            total += len(page.get_images())
        doc.close()
        return total
    except Exception:
        return 0


def get_processing_warning(filename: str, page_count: int, image_count: int, is_scan: bool, vlm_concurrency: int = 14) -> str:
    """Generate a user-friendly warning about document processing time with ETA."""
    import math
    
    parts = []
    if page_count > 20:
        parts.append(f"{page_count} страниц")
    if image_count > 10:
        parts.append(f"{image_count} изображений")
    if is_scan:
        parts.append("отсканированный документ")
    
    if not parts:
        return ""
    
    # Оценка времени обработки
    est_seconds = 0
    if is_scan:
        if OCR_SDK_ENABLED:
            # SDK: ~0.5 сек/стр + enrichment
            est_seconds = page_count * 0.5 + 10
        else:
            # VLM: каждая страница → VLM запрос, параллельно по vlm_concurrency
            batches = math.ceil(page_count / vlm_concurrency)
            est_seconds = batches * 20
    else:
        # TEXT PDF > 20 стр. → standard pipeline
        est_seconds = page_count * 0.2  # нативное извлечение текста
        if image_count > 0:
            # + VLM для картинок параллельно по vlm_concurrency
            img_batches = math.ceil(image_count / vlm_concurrency)
            est_seconds += img_batches * 20
    
    detail = ", ".join(parts)
    
    if est_seconds >= 60:
        est_min = math.ceil(est_seconds / 60)
        time_str = f"~{est_min} мин"
    elif est_seconds >= 10:
        time_str = f"~{int(est_seconds)} сек"
    else:
        return ""  # слишком быстро, предупреждение не нужно
    
    return (
        f"Документ «{filename}» содержит {detail}. "
        f"Ориентировочное время обработки: {time_str}."
    )



def _vlm_proxy_url(profile: str) -> str:
    """URL для инжекции в конфиги docling-serve.

    При VLM_PROXY_ENABLED=true и заданном VLM_PROXY_URL — указывает на
    сам прокси с нужным профилем (sampling и system-prompt инжектируются
    уже самим прокси). При false — fallback на DEFAULT_VLM_URL (поведение
    до v4.0, прямой ход docling-serve в LiteLLM/SGLang).

    Это даёт безопасный путь миграции: деплой → прогон smoke на
    /v1/chat/completions → переключение флага в .env → рестарт.
    """
    if VLM_PROXY_ENABLED and VLM_PROXY_URL:
        sep = "&" if "?" in VLM_PROXY_URL else "?"
        return f"{VLM_PROXY_URL}{sep}profile={profile}"
    return DEFAULT_VLM_URL


def build_picture_description_api(vlm_overrides: dict) -> str:
    params = {"model": vlm_overrides.get("vlm_model", DEFAULT_VLM_MODEL), "chat_template_kwargs": {"enable_thinking": False}}
    if "vlm_temperature" in vlm_overrides:
        params["temperature"] = float(vlm_overrides["vlm_temperature"])
    if "vlm_max_tokens" in vlm_overrides:
        params["max_tokens"] = int(vlm_overrides["vlm_max_tokens"])
    api_config = {
        "url": vlm_overrides.get("vlm_url", _vlm_proxy_url("picture_desc")),
        "headers": {"Authorization": f"Bearer {vlm_overrides.get('vlm_api_key', DEFAULT_VLM_API_KEY)}"},
        "params": params,
        "timeout": int(vlm_overrides.get("vlm_timeout", DEFAULT_VLM_TIMEOUT)),
        "concurrency": int(vlm_overrides.get("vlm_concurrency", DEFAULT_VLM_CONCURRENCY)),
        "prompt": vlm_overrides.get("vlm_prompt", DEFAULT_VLM_PROMPT) + "\n/no_think"
    }
    return json.dumps(api_config)


def build_custom_model(vlm_overrides: dict = {}, classification: str = "false") -> str:
    api_config = {
        "engine_options": {
            "engine_type": "api_openai",
            "url": DEFAULT_VLM_URL,
            "headers": {"Authorization": f"Bearer {DEFAULT_VLM_API_KEY}"},
            "timeout": 300
        },
        "model_spec": {
            "name": "Qwen3-VL",
            "default_repo_id": "Qwen/Qwen3-VL-32B-Instruct",
            "prompt": DEFAULT_VLM_PROMPT + "\n/no_think",
            "response_format": "markdown",
            "api_overrides": {
                "api_openai": {
                    "params": {
                        "model": vlm_overrides.get("vlm_model", DEFAULT_VLM_MODEL),
                        "max_completion_tokens": int(vlm_overrides.get("vlm_max_completion_tokens", DEFAULT_VLM_MAX_COMPLETION_TOKENS)),
                        "chat_template_kwargs": {"enable_thinking": False}
                    }
                }
            }
        },
        "prompt": DEFAULT_VLM_PROMPT + "\n/no_think",
        "batch_size": 1,
        "concurrency": int(vlm_overrides.get("vlm_concurrency", DEFAULT_VLM_CONCURRENCY)),
        "scale": float(vlm_overrides.get("vlm_scale", DEFAULT_VLM_SCALE)),
        "picture_area_threshold": 0.01,
        "generation_config": {"max_new_tokens": 2048, "do_sample": False}
    }

    # Если включена классификация, добавляем соответствующие параметры в конфиг
    if classification == "true":
        api_config["classification_min_confidence"] = 0.8
        api_config["classification_deny"] = ['icon', 'logo', 'signature', 'stamp', 'qr_code', 'bar_code']
        # api_config["classification_allow"] = ['other', 'picture_group', 'pie_chart', 'bar_chart', 'stacked_bar_chart', 'line_chart', 'flow_chart', 'scatter_chart', 'heatmap', 'remote_sensing', 'natural_image', 'chemistry_molecular_structure', 'chemistry_markush_structure', 'screenshot', 'map','stratigraphic_chart', 'engineering_drawing','cad_drawing', 'electrical_diagram']


    return json.dumps(api_config)


def build_vlm_pipeline_model_api(vlm_overrides: dict = {}) -> str:
    """VlmModelApi flat format for vlm_pipeline_model_api.

    Sampling-параметры (temperature, top_p, top_k, min_p, presence_penalty,
    repetition_penalty, max_tokens) теперь инжектируются на уровне VLM
    proxy (см. _vlm_proxy_url + endpoint /v1/chat/completions). Здесь их
    больше нет — иначе значение из proxy перетёрлось бы значением из
    конфига docling-serve. Убран и зашитый ранее `temperature: 0.0`.
    """
    config = {
        "url": vlm_overrides.get("vlm_url", _vlm_proxy_url("full_page")),
        "headers": {"Authorization": f"Bearer {vlm_overrides.get('vlm_api_key', DEFAULT_VLM_API_KEY)}"},
        "params": {
            "model": vlm_overrides.get("vlm_model", DEFAULT_VLM_MODEL),
            "max_completion_tokens": int(vlm_overrides.get("vlm_max_completion_tokens", DEFAULT_VLM_MAX_COMPLETION_TOKENS)),
            "chat_template_kwargs": {"enable_thinking": False}
        },
        "prompt": vlm_overrides.get("vlm_pipeline_prompt", DEFAULT_VLM_PIPELINE_PROMPT) + "\n/no_think",
        "response_format": "markdown",
        "timeout": int(vlm_overrides.get("vlm_timeout", DEFAULT_VLM_TIMEOUT)),
        "concurrency": int(vlm_overrides.get("vlm_concurrency", DEFAULT_VLM_CONCURRENCY)),
        "scale": float(vlm_overrides.get("vlm_scale", DEFAULT_VLM_SCALE)),
    }
    return json.dumps(config)



# ═══════════════════════════════════════════════════════════════
# Пост-обработка LaTeX для KaTeX-совместимости
# ═══════════════════════════════════════════════════════════════

# Маркер конца VLM-ответа (см. DEFAULT_VLM_PIPELINE_PROMPT).
# Квадратные скобки и подчёркивания стабильнее токенизируются на Qwen BBPE
# в INT4-квантовании, чем тройные угловые скобки — поэтому маркер именно в
# таком формате. В md_content подчёркивания часто приходят экранированными
# (`[END\_OF\_PAGE]`) — это делает markdown-рендерер docling. Регексы ниже
# принимают обе формы. Пост-обработка удаляет ВСЕ вхождения маркера —
# docling склеивает многостраничный вывод в один md_content, каждая
# страница оканчивается своим маркером.
_END_MARKER_STRICT_RE = re.compile(r"\[END(?:\\_|_)OF(?:\\_|_)PAGE\]")
_END_MARKER_FUZZY_RE = re.compile(
    r"\[?\s*END\\?[_\s]?OF\\?[_\s]?PAGE\s*\]?", re.IGNORECASE
)
# Пустая строка markdown-таблицы: только "|" и пробелы.
_EMPTY_TABLE_ROW_RE = re.compile(r"^\s*\|(?:\s*\|)+\s*$")


def _log_fix_vlm(info: dict) -> None:
    """Unconditional структурированный лог результата fix_vlm_truncation."""
    logger.info(
        f"[fix_vlm_truncation] md_len={info.get('md_len')} "
        f"had_end_marker={info.get('had_end_marker', False)} "
        f"had_fuzzy_marker={info.get('had_fuzzy_marker', False)} "
        f"markers_found_count={info.get('markers_found_count', 0)} "
        f"markers_removed_count={info.get('markers_removed_count', 0)} "
        f"trimmed={info.get('trimmed', False)} "
        f"action={info.get('action')}"
    )


def fix_vlm_truncation(response_bytes: bytes):
    """Пост-обработка md_content докла.

    Возвращает tuple (bytes, info_dict). Поля info:
        action: md_none | md_empty | non_json | no_document | end_marker_strict
                | end_marker_fuzzy | tail_trimmed | no_op
        md_len: длина исходного md_content
        had_end_marker / had_fuzzy_marker: нашли соответствующий вариант
        markers_found_count / markers_removed_count
        trimmed: была ли модификация содержимого

    Логика:
      1) Найдены точные маркеры [END_OF_PAGE] / [END\\_OF\\_PAGE] (markdown-
         экранированные подчёркивания) → отрезаем хвост после ПОСЛЕДНЕГО
         маркера (защита от loop-мусора после него), затем удаляем ВСЕ
         вхождения маркера из получившегося текста. docling многостраничный
         md_content содержит один маркер на страницу, все их удаляем.
      2) Иначе — fuzzy-вариант (неточная токенизация на INT4) → то же.
         Логируется отдельным WARNING с количеством найденных маркеров.
      3) Иначе — если stop_reason=length И после последней содержательной
         строки 5+ пустых табличных строк подряд → режем хвост и преобразуем
         partial_success → success (чтобы OWUI не падал).
      4) Иначе — no_op.
    """
    info = {
        "action": "no_op",
        "md_len": None,
        "had_end_marker": False,
        "had_fuzzy_marker": False,
        "markers_found_count": 0,
        "markers_removed_count": 0,
        "trimmed": False,
    }
    try:
        data = json.loads(response_bytes)
    except Exception:
        info["action"] = "non_json"
        _log_fix_vlm(info)
        return response_bytes, info

    doc = data.get("document")
    if not isinstance(doc, dict):
        info["action"] = "no_document"
        _log_fix_vlm(info)
        return response_bytes, info

    md = doc.get("md_content")
    if md is None:
        info["action"] = "md_none"
        _log_fix_vlm(info)
        return response_bytes, info
    if not isinstance(md, str):
        info["action"] = "md_none"
        _log_fix_vlm(info)
        return response_bytes, info
    if md == "":
        info["md_len"] = 0
        info["action"] = "md_empty"
        _log_fix_vlm(info)
        return response_bytes, info

    info["md_len"] = len(md)
    original_len = len(md)
    was_truncated = any(
        isinstance(e, dict)
        and "stop_reason=length" in str(e.get("error_message", ""))
        for e in (data.get("errors") or [])
    )

    changed = False
    new_md = md

    # 1) Точные маркеры (в т.ч. экранированные markdown'ом подчёркивания)
    strict_matches = list(_END_MARKER_STRICT_RE.finditer(md))
    if strict_matches:
        info["had_end_marker"] = True
        info["markers_found_count"] = len(strict_matches)
        # Обрезка мусорного хвоста ПОСЛЕ последнего маркера (защита от loop)
        tail_cut = md[: strict_matches[-1].end()]
        # Удаляем все оставшиеся маркеры из текста
        new_md = _END_MARKER_STRICT_RE.sub("", tail_cut).rstrip()
        info["markers_removed_count"] = len(strict_matches)
        info["action"] = "end_marker_strict"
        info["trimmed"] = True
        changed = True
    else:
        # 2) Fuzzy-маркеры (неточная токенизация под INT4)
        fuzzy_matches = list(_END_MARKER_FUZZY_RE.finditer(md))
        if fuzzy_matches:
            info["had_fuzzy_marker"] = True
            info["markers_found_count"] = len(fuzzy_matches)
            sample = md[fuzzy_matches[0].start(): fuzzy_matches[0].end()]
            logger.warning(
                f"[fix_vlm_truncation] fuzzy end-marker(s) matched: "
                f"count={len(fuzzy_matches)}, first=«{sample}»"
            )
            tail_cut = md[: fuzzy_matches[-1].end()]
            new_md = _END_MARKER_FUZZY_RE.sub("", tail_cut).rstrip()
            info["markers_removed_count"] = len(fuzzy_matches)
            info["action"] = "end_marker_fuzzy"
            info["trimmed"] = True
            changed = True
        # 3) Эвристическая обрезка при truncation без маркера
        elif was_truncated:
            lines = md.split("\n")
            last_meaningful = -1
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                if _EMPTY_TABLE_ROW_RE.match(line):
                    continue
                last_meaningful = i
            if 0 <= last_meaningful < len(lines) - 5:
                new_md = "\n".join(lines[: last_meaningful + 1]).rstrip()
                data["errors"] = []
                data["status"] = "success"
                info["action"] = "tail_trimmed"
                info["trimmed"] = True
                changed = True

    if not changed:
        _log_fix_vlm(info)
        return response_bytes, info

    doc["md_content"] = new_md
    try:
        out = json.dumps(data, ensure_ascii=False).encode("utf-8")
    except Exception:
        _log_fix_vlm(info)
        return response_bytes, info

    logger.info(
        f"VLM post-process: {info['action']}: {original_len} → {len(new_md)} chars "
        f"(markers_removed={info['markers_removed_count']})"
    )
    _log_fix_vlm(info)
    return out, info


# ═══════════════════════════════════════════════════════════════
# Диагностика null-ответов (md_content=null или partial_success)
# ═══════════════════════════════════════════════════════════════

def make_null_response_markdown(request_id: str = "") -> str:
    """Дружелюбная заглушка для OWUI с указанием времени и request_id.

    Вызывается после исчерпания ретраев либо при null/partial_success от
    docling-serve. Даёт пользователю понять, что документ не обработан,
    что делать, и ID для обращения к админу.
    """
    rid_short = request_id[:8] if request_id else "—"
    return (
        "# ⚠️ Документ не распознан\n\n"
        "Не удалось обработать этот документ. Пожалуйста, попробуйте "
        "загрузить его ещё раз.\n\n"
        "Если проблема повторится:\n"
        "- попробуйте загрузить меньший фрагмент (до 20 страниц);\n"
        "- проверьте, что документ не защищён паролем;\n"
        "- обратитесь к администратору.\n\n"
        "---\n"
        f"_Время: {dt.now().isoformat(timespec='seconds')}  ID: {rid_short}_"
    )

_NULL_RESPONSE_LOG_PATTERN = "null_response_*.json"
_ERROR_RESPONSE_LOG_PATTERN = "error_response_*.json"


def _save_null_response_log(
    *,
    resp_content: bytes,
    http_status: int,
    status_field,
    md_content,
    errors,
    fix_info: dict,
    params_dict: dict,
    file_names: list,
    client_ip,
    total_ms: int,
    request_id: str = "",
    attempts_made: int = 1,
    retry_reasons: list = None,
) -> str:
    """Сохраняет полный дамп null-ответа в LOG_DIR/null_response_*.json.
    Возвращает путь к файлу (или пустую строку при ошибке записи)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    now = dt.now()
    ts = now.strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(LOG_DIR, f"null_response_{ts}.json")

    try:
        full_response = json.loads(resp_content)
    except Exception:
        full_response = {
            "__parse_error__": True,
            "__raw_body_preview__": resp_content[:8000].decode("utf-8", "replace"),
        }

    md_preview = None
    md_len = None
    if isinstance(md_content, str):
        md_len = len(md_content)
        if md_len > 1000:
            md_preview = md_content[:500] + "\n...<cut>...\n" + md_content[-500:]
        else:
            md_preview = md_content

    payload = {
        "timestamp": now.isoformat(),
        "request_id": request_id,
        "attempts_made": attempts_made,
        "retry_reasons": retry_reasons or [],
        "client_ip": client_ip,
        "request_params": params_dict,
        "request_files": file_names,
        "docling_status_code": http_status,
        "docling_response_full": full_response,
        "docling_md_content_preview": md_preview,
        "docling_md_content_length": md_len,
        "docling_errors": errors,
        "docling_status_field": status_field,
        "timing_total_ms": total_ms,
        "fix_vlm_truncation_called": True,
        "fix_vlm_truncation_result": fix_info.get("action"),
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path
    except Exception as e:
        logger.error(f"[NULL-RESPONSE] failed to save {path}: {e}")
        return ""


def _classify_error_type(
    exc: "BaseException | None",
    docling_status_code: "int | None",
) -> str:
    """Определить тип ошибки для поля error_type в дампе."""
    if docling_status_code is not None:
        if docling_status_code in (408, 504):
            return "docling_timeout"
        if 500 <= docling_status_code < 600:
            return "docling_5xx"
        return "other"
    if exc is None:
        return "other"
    if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)):
        return "httpx_read_timeout"
    if isinstance(exc, httpx.ConnectTimeout):
        return "httpx_connect"
    if isinstance(exc, httpx.ConnectError):
        return "httpx_connect"
    if isinstance(exc, httpx.TimeoutException):
        return "httpx_read_timeout"
    if isinstance(exc, httpx.RequestError):
        return "httpx_connect"
    return "proxy_exception"


def _save_error_response_log(
    *,
    error_type: str,
    http_status_returned: int,
    docling_status_code,
    docling_response_body,
    exception,
    params_dict: dict,
    file_names: list,
    client_ip,
    duration_ms: int,
    request_id: str = "",
    attempts_made: int = 1,
    retry_reasons: list = None,
) -> str:
    """Сохраняет дамп ошибочного ответа в LOG_DIR/error_response_*.json.
    Возвращает путь к файлу (или пустую строку при ошибке записи)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    now = dt.now()
    ts = now.strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(LOG_DIR, f"error_response_{ts}.json")

    body_preview = None
    if docling_response_body is not None:
        try:
            if isinstance(docling_response_body, bytes):
                body_preview = docling_response_body[:2000].decode("utf-8", "replace")
            else:
                body_preview = str(docling_response_body)[:2000]
        except Exception:
            body_preview = None

    exc_type = exc_msg = tb_str = None
    if exception is not None:
        exc_type = type(exception).__name__
        try:
            exc_msg = str(exception)
        except Exception:
            exc_msg = None
        try:
            tb_str = "".join(
                traceback.format_exception(type(exception), exception, exception.__traceback__)
            )
        except Exception:
            tb_str = None

    payload = {
        "timestamp": now.isoformat(),
        "request_id": request_id,
        "attempts_made": attempts_made,
        "retry_reasons": retry_reasons or [],
        "client_ip": client_ip,
        "request_params": params_dict,
        "request_files": file_names,
        "error_type": error_type,
        "http_status_returned_to_client": http_status_returned,
        "docling_status_code": docling_status_code,
        "docling_response_body": body_preview,
        "exception_type": exc_type,
        "exception_message": exc_msg,
        "traceback": tb_str,
        "duration_ms": duration_ms,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path
    except Exception as e:
        logger.error(f"[ERROR-RESPONSE] failed to save {path}: {e}")
        return ""


def _cleanup_files_by_mtime(directory: str, pattern: str, retention_days: int, label: str) -> int:
    """Удалить файлы по glob-паттерну старше retention_days. Возвращает счётчик."""
    if not os.path.isdir(directory):
        return 0
    if retention_days <= 0:
        return 0
    cutoff = time.time() - retention_days * 86400
    removed = 0
    for p in glob.glob(os.path.join(directory, pattern)):
        try:
            if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                os.remove(p)
                removed += 1
        except Exception as e:
            logger.warning(f"[logs-cleanup] {label}: failed to remove {p}: {e}")
    if removed > 0:
        logger.info(
            f"[logs-cleanup] {label}: removed {removed} file(s) older than {retention_days}d"
        )
    return removed


def cleanup_old_null_response_logs() -> None:
    """Чистит null_response_*.json и error_response_*.json по своему retention.

    Имя сохранено для обратной совместимости с вызовом из lifespan на
    старте контейнера — внутри теперь оба паттерна с независимыми
    retention из ENV (NULL_RESPONSE_LOG_RETENTION_DAYS,
    ERROR_RESPONSE_LOG_RETENTION_DAYS).
    """
    _cleanup_files_by_mtime(
        LOG_DIR, _NULL_RESPONSE_LOG_PATTERN,
        NULL_RESPONSE_LOG_RETENTION_DAYS, "null_responses",
    )
    _cleanup_files_by_mtime(
        LOG_DIR, _ERROR_RESPONSE_LOG_PATTERN,
        ERROR_RESPONSE_LOG_RETENTION_DAYS, "error_responses",
    )


_VLM_REQUEST_LOG_GLOB = "vlm_requests_*.jsonl"


def cleanup_old_vlm_request_logs() -> None:
    """Чистит vlm_requests_*.jsonl: по retention И по суммарному размеру.

    Размерный лимит — страховка от вечного роста при коротком retention.
    Если VLM_REQUEST_LOG_MAX_SIZE_MB > 0 и сумма больше — удаляются
    самые старые до возврата под лимит, независимо от retention.
    Каждое такое удаление логируется WARNING.
    """
    base_dir = os.path.dirname(VLM_REQUEST_LOG_FILE) or "."
    if not os.path.isdir(base_dir):
        return

    # Шаг 1: retention.
    _cleanup_files_by_mtime(
        base_dir, _VLM_REQUEST_LOG_GLOB,
        VLM_REQUEST_LOG_RETENTION_DAYS, "vlm_requests",
    )

    # Шаг 2: размерный cap.
    if VLM_REQUEST_LOG_MAX_SIZE_MB <= 0:
        return
    cap_bytes = VLM_REQUEST_LOG_MAX_SIZE_MB * 1024 * 1024
    files: list = []
    for p in glob.glob(os.path.join(base_dir, _VLM_REQUEST_LOG_GLOB)):
        try:
            if os.path.isfile(p):
                files.append((os.path.getmtime(p), os.path.getsize(p), p))
        except Exception:
            pass
    total = sum(sz for _, sz, _ in files)
    if total <= cap_bytes:
        return
    files.sort()  # по mtime, старые первые
    removed_bytes = 0
    removed_count = 0
    for mtime, size, path in files:
        if total - removed_bytes <= cap_bytes:
            break
        try:
            os.remove(path)
            removed_bytes += size
            removed_count += 1
            logger.warning(
                f"[logs-cleanup] vlm_requests size-cap: removed {os.path.basename(path)} "
                f"({size} bytes, mtime={dt.utcfromtimestamp(mtime).isoformat()}Z)"
            )
        except Exception as e:
            logger.warning(f"[logs-cleanup] vlm_requests size-cap: remove failed {path}: {e}")
    if removed_count > 0:
        logger.warning(
            f"[logs-cleanup] vlm_requests size-cap: removed {removed_count} file(s), "
            f"freed {removed_bytes // (1024*1024)} MB "
            f"(was {total // (1024*1024)} MB, cap {VLM_REQUEST_LOG_MAX_SIZE_MB} MB)"
        )


async def _periodic_logs_cleanup() -> None:
    """Раз в сутки чистит все плоские лог-источники по их retention.

    Каталог truncate-дампов чистится отдельной задачей
    `_periodic_truncate_dumps_cleanup` — у него структура «каталог на
    дату», логика другая (rmtree по mtime директории).
    """
    while True:
        await asyncio.sleep(86400)
        try:
            cleanup_old_vlm_request_logs()
        except Exception as e:
            logger.error(f"[logs-cleanup] vlm_requests error: {e}")
        try:
            cleanup_old_null_response_logs()
        except Exception as e:
            logger.error(f"[logs-cleanup] null/error responses error: {e}")


def fix_katex_compatibility(response_bytes: bytes) -> bytes:
    """Fix LaTeX in docling response for KaTeX rendering in OpenWebUI."""
    try:
        data = json.loads(response_bytes)
        doc = data.get("document", {})
        if not isinstance(doc, dict):
            return response_bytes
        
        md = doc.get("md_content", "")
        if not md:
            return response_bytes
        
        original_len = len(md)
        
        # Проверяем парность $$ (блочные формулы)
        parts = md.split("$$")
        if len(parts) % 2 == 0:  # нечётное количество $$ = незакрытый блок
            md = md + "\n$$"

        # Fix HTML entities from VLM output (таблицы с &amp; вместо &)
        md = md.replace("&amp;", "&")
        md = md.replace("&lt;", "<")
        md = md.replace("&gt;", ">")

        if len(md) != original_len:
            logger.info(f"KaTeX fix: {original_len} -> {len(md)} chars")
        
        doc["md_content"] = md
        data["document"] = doc
        return json.dumps(data, ensure_ascii=False).encode()
    except Exception as e:
        logger.error(f"KaTeX fix error: {e}")
        return response_bytes



# ═══════════════════════════════════════════════════════════════
# Поддерживаемые форматы и обработка неподдерживаемых
# ═══════════════════════════════════════════════════════════════

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm",
    ".md", ".csv", ".png", ".jpg", ".jpeg", ".tiff", ".tif",
    ".bmp", ".gif", ".webp", ".asciidoc", ".adoc",
    ".xls",  # через xlrd конвертацию
    ".doc",  # через Gotenberg
}

SUPPORT_PORTAL_URL = "https://sd.kscgroup.ru/servicedesk/customer/portal/3/create/583"

def get_unsupported_response(filename: str) -> bytes:
    """Return a friendly error for unsupported file formats."""
    ext = os.path.splitext(filename)[1].lower() if filename else "unknown"
    error_msg = (
        f"К сожалению, файлы в формате «{ext}» пока не поддерживаются. "
        f"Попробуйте экспортировать документ в один из поддерживаемых форматов "
        f"(PDF, DOCX, XLSX, XLS, PPTX, CSV или изображение) и загрузить повторно. "
        f"Если возникнут вопросы — оставьте заявку на портале техподдержки: {SUPPORT_PORTAL_URL}"
    )
    return json.dumps({"detail": error_msg}, ensure_ascii=False).encode()


def convert_xls_to_markdown(file_bytes: bytes, filename: str) -> bytes:
    """Convert .xls file to markdown using xlrd."""
    try:
        import xlrd
    except ImportError:
        return None
    
    try:
        book = xlrd.open_workbook(file_contents=file_bytes)
        all_md = []
        
        for sheet in book.sheets():
            if sheet.nrows == 0:
                continue
            if book.nsheets > 1:
                all_md.append(f"## {sheet.name}")
                all_md.append("")
            
            for rx in range(sheet.nrows):
                row = []
                for cx in range(sheet.ncols):
                    val = sheet.cell_value(rx, cx)
                    if isinstance(val, float) and val == int(val):
                        val = int(val)
                    row.append(str(val).replace("|", "\\|"))
                all_md.append("| " + " | ".join(row) + " |")
                if rx == 0:
                    all_md.append("|" + "|".join(["---"] * sheet.ncols) + "|")
            all_md.append("")
        
        md_content = "\n".join(all_md)
        
        response = {
            "document": {
                "filename": filename,
                "md_content": md_content,
            },
            "status": "success",
            "errors": [],
            "processing_time": 0.1,
        }
        return json.dumps(response, ensure_ascii=False).encode()
    except Exception as e:
        logger.error(f"XLS conversion error: {e}")
        return None


async def convert_doc_to_markdown(client: httpx.AsyncClient, file_bytes: bytes, filename: str) -> bytes:
    """Convert binary .doc to markdown via Gotenberg (doc→PDF) + PyMuPDF (PDF→text).
    
    Analogous to convert_xls_to_markdown — returns docling-compatible JSON response.
    """
    try:
        # Шаг 1: .doc → PDF через Gotenberg
        _t = time.time()
        gotenberg_url = f"{GOTENBERG_URL}/forms/libreoffice/convert"
        files = [("files", (filename, file_bytes, "application/msword"))]
        resp = await client.post(gotenberg_url, files=files, timeout=120.0)
        if resp.status_code != 200:
            logger.error(f"DOC→PDF Gotenberg failed: HTTP {resp.status_code}")
            return None
        pdf_bytes = resp.content
        _gotenberg_ms = (time.time() - _t) * 1000
        logger.info(f"TIMING doc→pdf (Gotenberg): {_gotenberg_ms:.0f}ms ({len(pdf_bytes)} bytes)")
        
        # Шаг 2: PDF → markdown через PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        all_md = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                all_md.append(text)
            # Разделитель страниц
            if page_num < len(doc) - 1 and text:
                all_md.append("")
                all_md.append("---")
                all_md.append("")
        doc.close()
        
        md_content = "\n".join(all_md)
        _total_ms = (time.time() - _t) * 1000
        logger.info(f"TIMING doc→markdown total: {_total_ms:.0f}ms ({len(md_content)} chars)")
        
        response = {
            "document": {
                "filename": filename,
                "md_content": md_content,
            },
            "status": "success",
            "errors": [],
            "processing_time": (time.time() - _t),
        }
        return json.dumps(response, ensure_ascii=False).encode()
    except Exception as e:
        logger.error(f"DOC conversion error: {e}")
        return None



# ═══════════════════════════════════════════════════════════════
# OCR SDK — обработка SCAN PDF (v4.0)
# ═══════════════════════════════════════════════════════════════

def calculate_enrich_max_tokens(label: str, bbox: list) -> int:
    """Динамический max_tokens для обогащения регионов OCR SDK.

    База по label × множитель площади × множитель формы.
    bbox — нормализованные координаты OCR SDK (0..1000 по каждой оси),
    norm_area соответственно лежит в диапазоне 0..1_000_000.

    Логика:
    - Штампы/печати фактически короткие (пара предложений максимум).
    - Обычные картинки (подписи, логотипы, мелкие фото) обычно короткие.
    - Графикам нужен запас под оси, легенду, ключевые значения.
    - Инженерным чертежам законно нужно 3K+ под штамп, размеры,
      спецификации, позиции, сечения.
    - Крошечные или сильно вытянутые регионы обычно декоративные
      или ошибочно классифицированные — снижаем cap, чтобы VLM не
      ушла в описательный loop.

    Пороги — начальные прикидки, после выката калибруются на реальных
    документах (книги, CAD-чертежи, документы со штампами).
    """
    x1, y1, x2, y2 = bbox
    w, h = (x2 - x1), (y2 - y1)
    norm_area = w * h

    base_by_label = {
        "seal":                 256,
        "stamp":                256,
        "image":                768,
        "chart":                1536,
        "engineering_drawing":  3072,
        "cad_drawing":          3072,
        "electrical_diagram":   3072,
    }
    base = base_by_label.get(label, 768)

    # Множитель площади: кропы <1% страницы обычно декоративные/мисклассы
    if norm_area < 10_000:
        area_mult = 0.4
    elif norm_area < 100_000:
        area_mult = 0.7
    else:
        area_mult = 1.0

    # Множитель формы: сильно вытянутые регионы обычно рамки,
    # штрихкоды, декоративные линии, а не контент
    aspect = max(w, h) / max(min(w, h), 1)
    shape_mult = 0.5 if aspect > 5 else 1.0

    result = int(base * area_mult * shape_mult)
    # Пол: модель должна хотя бы успеть сказать «Пусто»
    return max(result, 128)


async def enrich_image_regions(
    client: httpx.AsyncClient, markdown: str, json_result: list,
    pdf_bytes: bytes, vlm_overrides: dict
) -> str:
    """Find image-type regions in SDK output and describe them via 122B VLM."""
    regions_to_enrich = []
    for page_idx, page_regions in enumerate(json_result):
        if not isinstance(page_regions, list):
            continue
        for region in page_regions:
            label = region.get("label", "")
            if label in ENRICH_LABELS and region.get("content") is None:
                bbox = region.get("bbox_2d")
                if bbox and len(bbox) == 4:
                    regions_to_enrich.append({"page": page_idx, "bbox": bbox, "label": label})

    if not regions_to_enrich:
        logger.info("OCR SDK enrichment: no image regions found")
        return markdown

    logger.info(f"OCR SDK enrichment: {len(regions_to_enrich)} region(s) to describe via 122B")

    vlm_url = vlm_overrides.get("vlm_url", DEFAULT_VLM_URL)
    vlm_api_key = vlm_overrides.get("vlm_api_key", DEFAULT_VLM_API_KEY)
    vlm_model = vlm_overrides.get("vlm_model", DEFAULT_VLM_MODEL)
    vlm_timeout = int(vlm_overrides.get("vlm_timeout", DEFAULT_VLM_TIMEOUT))
    vlm_concurrency = int(vlm_overrides.get("vlm_concurrency", DEFAULT_VLM_CONCURRENCY))
    prompt = vlm_overrides.get("vlm_prompt", ENRICH_VLM_PROMPT)

    sem = asyncio.Semaphore(vlm_concurrency)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    render_scale = 3.0

    tasks = []
    placeholder_map = {}

    for i, region in enumerate(regions_to_enrich):
        page_idx = region["page"]
        x1, y1, x2, y2 = region["bbox"]
        if page_idx >= len(doc):
            logger.warning(f"OCR SDK enrichment: page {page_idx} out of range, skipping")
            continue
        # OCR SDK отдаёт bbox_2d в нормализованных координатах 0-1000 по каждой оси
        # (x/image_width*1000, y/image_height*1000).
        # Обратно в пункты PDF переводим через реальные размеры страницы.
        page = doc[page_idx]
        pw, ph = page.rect.width, page.rect.height
        clip_rect = fitz.Rect(
            x1 / 1000.0 * pw,
            y1 / 1000.0 * ph,
            x2 / 1000.0 * pw,
            y2 / 1000.0 * ph,
        )
        mat = fitz.Matrix(render_scale, render_scale)
        try:
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)
            png_bytes_region = pix.tobytes("png")
        except Exception as e:
            logger.error(f"OCR SDK enrichment: render failed page={page_idx} bbox={region['bbox']}: {e}")
            continue
        placeholder = f"![](page={page_idx},bbox=[{x1}, {y1}, {x2}, {y2}])"
        placeholder_map[i] = placeholder
        max_tok = calculate_enrich_max_tokens(region["label"], region["bbox"])
        # Наблюдаемость: по логам калибруем пороги после выката.
        bw, bh = (x2 - x1), (y2 - y1)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        logger.info(
            f"OCR SDK enrichment: region p{page_idx} label={region['label']} "
            f"bbox={region['bbox']} norm_area={bw * bh} aspect={aspect:.1f} "
            f"max_tokens={max_tok}"
        )
        tasks.append(_describe_single_region(
            client, png_bytes_region, sem,
            vlm_url, vlm_api_key, vlm_model, vlm_timeout, prompt,
            max_tokens=max_tok,
        ))

    doc.close()

    if not tasks:
        return markdown

    _t = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    _elapsed = (time.time() - _t) * 1000

    matched = 0
    for i, result in enumerate(results):
        if i not in placeholder_map:
            continue
        placeholder = placeholder_map[i]
        if isinstance(result, Exception):
            logger.error(f"OCR SDK enrichment: VLM error for region {i}: {result}")
            description = f"[Ошибка описания: {type(result).__name__}]"
        else:
            description = result
        if placeholder in markdown:
            markdown = markdown.replace(placeholder, description, 1)
            matched += 1
        else:
            bbox = regions_to_enrich[i]["bbox"]
            page = regions_to_enrich[i]["page"]
            pattern = re.compile(
                r'!\[\]\(page=' + str(page) + r',\s*bbox=\[\s*' +
                str(bbox[0]) + r'\s*,\s*' + str(bbox[1]) + r'\s*,\s*' +
                str(bbox[2]) + r'\s*,\s*' + str(bbox[3]) + r'\s*\]\)'
            )
            new_md, count = pattern.subn(description, markdown, count=1)
            if count > 0:
                markdown = new_md
                matched += 1
            else:
                logger.warning(f"OCR SDK enrichment: placeholder not found for page={page} bbox={bbox}")
                markdown += f"\n\n{description}\n"
                matched += 1

    logger.info(f"OCR SDK enrichment: {matched}/{len(tasks)} descriptions inserted, {_elapsed:.0f}ms")
    return markdown


async def _describe_single_region(
    client: httpx.AsyncClient, png_bytes: bytes, sem: asyncio.Semaphore,
    vlm_url: str, vlm_api_key: str, vlm_model: str, vlm_timeout: int, prompt: str,
    max_tokens: int = 768,
) -> str:
    """Send a single image region to 122B VLM and get text description."""
    b64 = base64.b64encode(png_bytes).decode()
    body = {
        "model": vlm_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt + "\n/no_think"}
                ]
            }
        ],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {vlm_api_key}"
    }
    async with sem:
        resp = await client.post(vlm_url, json=body, headers=headers, timeout=float(vlm_timeout))
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content or "[Описание недоступно]"


async def convert_scan_via_ocr_sdk(
    client: httpx.AsyncClient, pdf_bytes: bytes, filename: str, vlm_overrides: dict
) -> bytes:
    """Process SCAN PDF via OCR SDK + optional enrichment via 122B.

    Returns docling-compatible JSON bytes, or None on failure (triggers fallback).
    """
    file_id = str(uuid.uuid4())
    inbox_path = os.path.join(OCR_SDK_INBOX_CONTAINER, f"{file_id}.pdf")
    _t_start = time.time()
    try:
        with open(inbox_path, "wb") as f:
            f.write(pdf_bytes)
        logger.info(f"OCR SDK: wrote {len(pdf_bytes)} bytes to {inbox_path}")

        sdk_payload = {"images": [f"{OCR_SDK_INBOX_CONTAINER}/{file_id}.pdf"]}
        _t_sdk = time.time()
        sdk_resp = await client.post(
            f"{OCR_SDK_URL}/glmocr/parse",
            json=sdk_payload,
            timeout=float(OCR_SDK_TIMEOUT)
        )
        _sdk_ms = (time.time() - _t_sdk) * 1000

        if sdk_resp.status_code != 200:
            logger.error(f"OCR SDK: HTTP {sdk_resp.status_code} after {_sdk_ms:.0f}ms")
            return None

        sdk_data = sdk_resp.json()
        markdown = sdk_data.get("markdown_result", "")
        json_result = sdk_data.get("json_result", [])

        if not markdown:
            logger.error(f"OCR SDK: empty markdown_result after {_sdk_ms:.0f}ms")
            return None

        logger.info(f"OCR SDK: {len(markdown)} chars markdown, {len(json_result)} pages, {_sdk_ms:.0f}ms")

        if ENRICH_PICTURES_WITH_122B and json_result:
            try:
                _t_enrich = time.time()
                markdown = await enrich_image_regions(
                    client, markdown, json_result, pdf_bytes, vlm_overrides
                )
                _enrich_ms = (time.time() - _t_enrich) * 1000
                logger.info(f"OCR SDK enrichment: {_enrich_ms:.0f}ms")
            except Exception as e:
                logger.error(f"OCR SDK enrichment ERROR (non-fatal): {e}")

        _total_ms = (time.time() - _t_start) * 1000
        response = {
            "document": {"filename": filename, "md_content": markdown},
            "status": "success",
            "errors": [],
            "processing_time": _total_ms / 1000,
        }
        logger.info(f"TIMING ocr_sdk total: {_total_ms:.0f}ms")
        return json.dumps(response, ensure_ascii=False).encode()

    except Exception as e:
        _total_ms = (time.time() - _t_start) * 1000
        logger.error(f"OCR SDK ERROR after {_total_ms:.0f}ms: {e}")
        return None

    finally:
        try:
            if os.path.exists(inbox_path):
                os.remove(inbox_path)
                logger.info(f"OCR SDK: cleaned up {inbox_path}")
        except Exception as e:
            logger.error(f"OCR SDK: cleanup failed {inbox_path}: {e}")


# ═══════════════════════════════════════════════════════════════
# VLM endpoint: /v1/chat/completions с инжекцией sampling и аналитикой
# ═══════════════════════════════════════════════════════════════
# Endpoint выступает gateway'ем между docling-serve и LiteLLM/SGLang.
# Регистрируется ВЫШЕ catch-all `/{path:path}` (FastAPI-роутинг идёт по
# порядку объявления). При ?profile=full_page|picture_desc прокси:
#   1) инжектирует sampling-параметры из ENV соответствующего профиля,
#      если клиент их не переопределил;
#   2) добавляет system-промпт, если в messages нет роли system;
#   3) добавляет chat_template_kwargs.enable_thinking=false;
#   4) подменяет Authorization на Bearer ${VLM_UPSTREAM_API_KEY};
#   5) форвардит на VLM_UPSTREAM_URL (LiteLLM);
#   6) пишет одну строку в vlm_requests_<DATE>.jsonl;
#   7) при finish_reason=length — асинхронно сохраняет дамп в
#      ${VLM_TRUNCATE_LOG_DIR}/<DATE>/<request_id>/.
# Тело ответа клиенту НЕ модифицируется (passthrough).

_VLM_OPENAI_SAMPLING_KEYS = {
    "temperature", "top_p", "top_k", "min_p",
    "presence_penalty", "frequency_penalty", "repetition_penalty",
    "max_tokens", "max_completion_tokens",
}


def _vlm_request_log_path_for(date_str: str) -> str:
    """`vlm_requests_<DATE>.jsonl` рядом с базовым именем VLM_REQUEST_LOG_FILE.

    Параметр VLM_REQUEST_LOG_FILE трактуется как путь-шаблон: каталог +
    базовое имя; реальные файлы — с дневным суффиксом.
    """
    base_dir = os.path.dirname(VLM_REQUEST_LOG_FILE) or "."
    base = os.path.basename(VLM_REQUEST_LOG_FILE) or "vlm_requests.jsonl"
    stem, ext = os.path.splitext(base)
    if not ext:
        ext = ".jsonl"
    return os.path.join(base_dir, f"{stem}_{date_str}{ext}")


def _vlm_log_request(record: dict) -> None:
    """Append-only запись одной строки в дневной JSONL.

    workers=1 в Dockerfile → один писатель, без блокировок. PIPE_BUF
    гарантирует атомарность строки до 4KB на ext4. Запись синхронная,
    но дешёвая (<1 мс при <1 МБ payload — у нас он сотни байт).
    """
    try:
        os.makedirs(os.path.dirname(VLM_REQUEST_LOG_FILE) or ".", exist_ok=True)
        date_str = dt.utcnow().strftime("%Y-%m-%d")
        path = _vlm_request_log_path_for(date_str)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.error(f"[vlm-log] failed to write: {e}")


def _vlm_strip_images_in_place(body: dict, save_to_dir: "str | None") -> int:
    """Извлечь все base64-картинки из messages[].content[] и подменить URL.

    Если save_to_dir задан — декодирует и сохраняет картинки как
    page_<i>.png рядом с request.json. URL в самом теле всегда
    подменяется на маркер, чтобы request.json не разрастался от base64.
    Возвращает количество найденных картинок.
    """
    count = 0
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        return 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict) or c.get("type") != "image_url":
                continue
            iu = c.get("image_url")
            if not isinstance(iu, dict):
                continue
            url = iu.get("url", "")
            if not (isinstance(url, str) and url.startswith("data:")):
                continue
            placeholder = f"<see page_{count}.png>"
            if save_to_dir is not None:
                try:
                    _, b64 = url.split(",", 1)
                    with open(os.path.join(save_to_dir, f"page_{count}.png"), "wb") as f:
                        f.write(base64.b64decode(b64))
                except Exception as e:
                    logger.warning(f"[vlm-truncate-dump] image decode failed: {e}")
                    placeholder = f"<base64 stripped (decode failed: {type(e).__name__})>"
            else:
                placeholder = "<base64 stripped>"
            iu["url"] = placeholder
            count += 1
    return count


def _vlm_save_truncate_dump(
    *,
    request_id: str,
    profile: str,
    model: str,
    request_body: dict,
    response_data,
    prompt_tokens,
    completion_tokens,
    elapsed_ms: int,
    sampling_used: dict,
    max_tokens_requested,
) -> None:
    """Синхронный дамп truncate-кейса. Вызывается из asyncio.to_thread.

    Создаёт ${VLM_TRUNCATE_LOG_DIR}/<YYYY-MM-DD>/<request_id>/ с
    request.json, response.json, meta.json и (опционально) page_*.png.
    """
    try:
        date_str = dt.utcnow().strftime("%Y-%m-%d")
        out_dir = os.path.join(VLM_TRUNCATE_LOG_DIR, date_str, request_id)
        os.makedirs(out_dir, exist_ok=True)

        # Глубокая копия — чтобы не трогать исходное тело, которое уже
        # отправлено upstream. Самый простой способ: round-trip через JSON.
        try:
            body_clean = json.loads(json.dumps(request_body, default=str))
        except Exception:
            body_clean = request_body

        if VLM_TRUNCATE_SAVE_PAYLOAD:
            _vlm_strip_images_in_place(body_clean, out_dir)
            try:
                with open(os.path.join(out_dir, "request.json"), "w", encoding="utf-8") as f:
                    json.dump(body_clean, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"[vlm-truncate-dump] request.json failed: {e}")
            if response_data is not None:
                try:
                    with open(os.path.join(out_dir, "response.json"), "w", encoding="utf-8") as f:
                        json.dump(response_data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.error(f"[vlm-truncate-dump] response.json failed: {e}")
        else:
            # Компактный режим: без request/response/картинок, только meta.
            _vlm_strip_images_in_place(body_clean, None)

        meta = {
            "request_id": request_id,
            "ts": dt.utcnow().isoformat() + "Z",
            "profile": profile,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "max_tokens_requested": max_tokens_requested,
            "max_tokens_used": completion_tokens,
            "finish_reason": "length",
            "elapsed_ms": elapsed_ms,
            "sampling_params_used": sampling_used,
            "save_payload": VLM_TRUNCATE_SAVE_PAYLOAD,
        }
        try:
            with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[vlm-truncate-dump] meta.json failed: {e}")
        logger.info(
            f"[vlm-truncate-dump] saved {out_dir} "
            f"(profile={profile} compl_tok={completion_tokens})"
        )
    except Exception as e:
        logger.error(f"[vlm-truncate-dump] failed for {request_id}: {e}")


def cleanup_old_truncate_dumps(retention_days: "int | None" = None) -> None:
    """Удаляет каталоги старше retention_days в VLM_TRUNCATE_LOG_DIR."""
    days = retention_days if retention_days is not None else VLM_TRUNCATE_RETENTION_DAYS
    if not os.path.isdir(VLM_TRUNCATE_LOG_DIR):
        return
    cutoff = time.time() - (days * 86400)
    removed = 0
    try:
        import shutil
        for entry in os.listdir(VLM_TRUNCATE_LOG_DIR):
            full = os.path.join(VLM_TRUNCATE_LOG_DIR, entry)
            if not os.path.isdir(full):
                continue
            try:
                if os.path.getmtime(full) < cutoff:
                    shutil.rmtree(full, ignore_errors=True)
                    removed += 1
            except Exception:
                pass
    except Exception as e:
        logger.error(f"[truncate cleanup] error: {e}")
    if removed > 0:
        logger.info(
            f"[truncate cleanup] removed {removed} day-dir(s) older than {days}d"
        )


async def _periodic_truncate_dumps_cleanup() -> None:
    """Раз в сутки чистит дампы старше VLM_TRUNCATE_RETENTION_DAYS."""
    while True:
        await asyncio.sleep(86400)
        try:
            cleanup_old_truncate_dumps()
        except Exception as e:
            logger.error(f"[truncate cleanup] error: {e}")


def _vlm_inject_sampling(body: dict, profile_sampling: dict) -> None:
    """Положить значения из profile_sampling в body, если клиент их не задал.

    Клиентское значение всегда приоритетнее. max_tokens — отдельный
    случай: если клиент задал max_completion_tokens, это эквивалент,
    второй ключ инжектировать не нужно.
    """
    for key, val in profile_sampling.items():
        if key == "max_tokens":
            if "max_tokens" in body or "max_completion_tokens" in body:
                continue
            body["max_tokens"] = val
        else:
            if key in body:
                continue
            body[key] = val


def _vlm_inject_system_prompt(body: dict, prompt: str) -> None:
    """Добавить system-сообщение в начало messages, если его там нет."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        body["messages"] = [{"role": "system", "content": prompt}]
        return
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            return
    body["messages"] = [{"role": "system", "content": prompt}] + messages


def _vlm_inject_chat_template_kwargs(body: dict) -> None:
    """Дублирующая защита от thinking mode: enable_thinking=false."""
    ctk = body.get("chat_template_kwargs")
    if not isinstance(ctk, dict):
        body["chat_template_kwargs"] = {"enable_thinking": False}
        return
    if "enable_thinking" not in ctk:
        ctk["enable_thinking"] = False


@app.post("/v1/chat/completions")
async def vlm_chat_completions(request: Request):
    profile = (request.query_params.get("profile") or "").strip().lower()
    if profile == "":
        logger.warning("VLM endpoint: profile not specified, defaulting to full_page")
        profile = "full_page"
    if profile not in ("full_page", "picture_desc"):
        return Response(
            content=json.dumps(
                {"error": {"message": f"unknown profile: {profile}",
                           "type": "invalid_request_error"}},
                ensure_ascii=False,
            ).encode("utf-8"),
            status_code=400,
            headers={"content-type": "application/json"},
        )

    try:
        body = await request.json()
    except Exception:
        return Response(
            content=json.dumps(
                {"error": {"message": "invalid JSON body",
                           "type": "invalid_request_error"}},
                ensure_ascii=False,
            ).encode("utf-8"),
            status_code=400,
            headers={"content-type": "application/json"},
        )

    if not isinstance(body, dict):
        return Response(
            content=json.dumps(
                {"error": {"message": "body must be a JSON object",
                           "type": "invalid_request_error"}},
                ensure_ascii=False,
            ).encode("utf-8"),
            status_code=400,
            headers={"content-type": "application/json"},
        )

    if body.get("stream"):
        return Response(
            content=json.dumps(
                {"error": {"message": "streaming is not supported by the VLM proxy",
                           "type": "invalid_request_error"}},
                ensure_ascii=False,
            ).encode("utf-8"),
            status_code=400,
            headers={"content-type": "application/json"},
        )

    sampling = (
        VLM_FULL_PAGE_SAMPLING if profile == "full_page" else VLM_PICTURE_DESC_SAMPLING
    )
    sys_prompt = (
        VLM_FULL_PAGE_PROMPT if profile == "full_page" else VLM_PICTURE_DESC_PROMPT
    )

    # Snapshot до инжекций — нужен для строки А, чтобы понять, какие
    # ключи прокси добавил, а какие пришли от клиента.
    client_keys = set(body.keys())
    client_messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    client_had_system = any(
        isinstance(m, dict) and m.get("role") == "system" for m in client_messages
    )
    msgs_count = len(client_messages)
    has_image = any(
        isinstance(c, dict) and c.get("type") == "image_url"
        for m in client_messages if isinstance(m, dict)
        for c in (m.get("content") if isinstance(m.get("content"), list) else [])
    )

    _vlm_inject_sampling(body, sampling)
    _vlm_inject_system_prompt(body, sys_prompt)
    _vlm_inject_chat_template_kwargs(body)

    sampling_used = {k: body.get(k) for k in _VLM_OPENAI_SAMPLING_KEYS if k in body}

    # Что именно прокси добавил (то, чего у клиента не было). max_tokens
    # отдельно показан полем `max_tokens=N (source)`, поэтому в этот
    # список не включается — иначе дублирование.
    sampling_injected: list = [
        k for k in sampling.keys()
        if k != "max_tokens" and k not in client_keys
    ]
    mt_value = body.get("max_tokens") or body.get("max_completion_tokens")
    mt_source = (
        "client" if ("max_tokens" in client_keys or "max_completion_tokens" in client_keys)
        else "default"
    )
    sys_source = "client" if client_had_system else "injected"

    request_id = str(uuid.uuid4())
    rid8 = request_id[:8]
    model = body.get("model", "")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {VLM_UPSTREAM_API_KEY}",
        "X-Request-Id": request_id,
    }

    # Строка А — приём, до форварда. При нагрузке даёт реалтайм-видимость
    # того, что прокси отправляет в upstream (sampling, prompt, картинки),
    # без необходимости ждать запись в JSONL.
    logger.info(
        f"[vlm rid={rid8}] received profile={profile} model={model} "
        f"msgs={msgs_count} has_image={'true' if has_image else 'false'} "
        f"max_tokens={mt_value if mt_value is not None else '-'} ({mt_source}) "
        f"sampling_injected={','.join(sampling_injected) or '(none)'} "
        f"system_prompt={sys_source}"
    )

    client: httpx.AsyncClient = request.app.state.client
    _t = time.time()
    try:
        upstream_resp = await client.post(
            VLM_UPSTREAM_URL, json=body, headers=headers,
            timeout=httpx.Timeout(1200.0, connect=10.0),
        )
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        elapsed_ms = int((time.time() - _t) * 1000)
        err_type = _classify_error_type(exc, None)
        http_returned = 504 if "timeout" in err_type else 502
        logger.error(
            f"[vlm rid={rid8}] upstream {http_returned} finish=- tokens=-/- "
            f"elapsed={elapsed_ms}ms status=error "
            f"msg=\"{err_type}: {type(exc).__name__}: {exc}\""
        )
        _vlm_log_request({
            "ts": dt.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
            "request_id": request_id,
            "profile": profile,
            "model": model,
            "prompt_tokens": None,
            "completion_tokens": None,
            "finish_reason": None,
            "elapsed_ms": elapsed_ms,
            "status": "error",
            "error": f"{err_type}: {type(exc).__name__}",
        })
        return Response(
            content=json.dumps(
                {"error": {"message": f"upstream {err_type}",
                           "type": "upstream_error"}},
                ensure_ascii=False,
            ).encode("utf-8"),
            status_code=http_returned,
            headers={"content-type": "application/json", "X-Request-ID": request_id},
        )

    elapsed_ms = int((time.time() - _t) * 1000)
    upstream_status = upstream_resp.status_code

    finish_reason = None
    prompt_tokens = None
    completion_tokens = None
    response_data = None
    if upstream_resp.headers.get("content-type", "").startswith("application/json"):
        try:
            response_data = upstream_resp.json()
        except Exception:
            response_data = None

    if isinstance(response_data, dict):
        choices = response_data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason")
        usage = response_data.get("usage") or {}
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")

    if upstream_status >= 400:
        status = "error"
    elif finish_reason == "length":
        status = "truncated"
    else:
        status = "ok"

    _vlm_log_request({
        "ts": dt.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z",
        "request_id": request_id,
        "profile": profile,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "elapsed_ms": elapsed_ms,
        "status": status,
        "upstream_status": upstream_status,
    })

    # Строка Б — ответ. WARNING при truncated, ERROR при upstream 4xx/5xx.
    _ct = completion_tokens if completion_tokens is not None else "-"
    _pt = prompt_tokens if prompt_tokens is not None else "-"
    _fr = finish_reason if finish_reason is not None else "-"
    if status == "truncated":
        dump_dir = os.path.join(
            VLM_TRUNCATE_LOG_DIR, dt.utcnow().strftime("%Y-%m-%d"), request_id
        )
        max_tokens_requested = body.get("max_tokens") or body.get("max_completion_tokens")
        # Дамп в фоне, чтобы не задерживать ответ клиенту.
        asyncio.create_task(asyncio.to_thread(
            _vlm_save_truncate_dump,
            request_id=request_id,
            profile=profile,
            model=model,
            request_body=body,
            response_data=response_data,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_ms=elapsed_ms,
            sampling_used=sampling_used,
            max_tokens_requested=max_tokens_requested,
        ))
        logger.warning(
            f"[vlm rid={rid8}] upstream {upstream_status} finish=length "
            f"tokens={_ct}/{_pt} elapsed={elapsed_ms}ms status=truncated "
            f"dump_dir={dump_dir}"
        )
    elif status == "error":
        # upstream вернул 4xx/5xx — текст ошибки берём из тела (если влезает).
        try:
            err_msg = upstream_resp.text[:200].replace("\n", " ").replace("\"", "'")
        except Exception:
            err_msg = ""
        logger.error(
            f"[vlm rid={rid8}] upstream {upstream_status} finish={_fr} "
            f"tokens={_ct}/{_pt} elapsed={elapsed_ms}ms status=error "
            f"msg=\"{err_msg}\""
        )
    else:
        logger.info(
            f"[vlm rid={rid8}] upstream {upstream_status} finish={_fr} "
            f"tokens={_ct}/{_pt} elapsed={elapsed_ms}ms status=ok"
        )

    resp_headers = {"X-Request-ID": request_id}
    ct = upstream_resp.headers.get("content-type")
    if ct:
        resp_headers["content-type"] = ct
    return Response(
        content=upstream_resp.content,
        status_code=upstream_status,
        headers=resp_headers,
    )


@app.get("/_stats/vlm")
async def stats_vlm(date: str = ""):
    """Агрегированная сводка по vlm_requests_<DATE>.jsonl за дату.

    Лимит 100k строк — для стабильного p99. Для больших объёмов —
    внешний агрегатор (Loki/ELK), здесь только быстрый sanity-check.
    """
    if not date:
        date = dt.utcnow().strftime("%Y-%m-%d")
    try:
        dt.strptime(date, "%Y-%m-%d")
    except ValueError:
        return Response(
            content=json.dumps(
                {"error": "invalid date, expected YYYY-MM-DD"},
                ensure_ascii=False,
            ).encode("utf-8"),
            status_code=400,
            headers={"content-type": "application/json"},
        )

    path = _vlm_request_log_path_for(date)
    if not os.path.isfile(path):
        return {
            "date": date,
            "total": 0,
            "by_profile": {},
            "by_model": {},
            "p50_elapsed_ms": None,
            "p95_elapsed_ms": None,
        }

    LIMIT = 100_000
    rows: list = []
    too_large = False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= LIMIT:
                    too_large = True
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception as e:
        return Response(
            content=json.dumps(
                {"error": f"failed to read log: {e}"}, ensure_ascii=False,
            ).encode("utf-8"),
            status_code=500,
            headers={"content-type": "application/json"},
        )

    if too_large:
        return {
            "date": date,
            "warning": f"file too large (>{LIMIT} lines), use external aggregator",
            "limit": LIMIT,
            "partial_total": len(rows),
        }

    by_profile: dict = {}
    by_model: dict = {}
    elapsed: list = []
    for r in rows:
        prof = r.get("profile") or "unknown"
        st = r.get("status") or "unknown"
        bp = by_profile.setdefault(
            prof, {"ok": 0, "truncated": 0, "error": 0, "truncate_rate": 0.0}
        )
        if st in bp:
            bp[st] += 1
        m = r.get("model")
        if m:
            by_model[m] = by_model.get(m, 0) + 1
        ms = r.get("elapsed_ms")
        if isinstance(ms, (int, float)):
            elapsed.append(ms)

    for prof, counts in by_profile.items():
        total_for_prof = counts["ok"] + counts["truncated"] + counts["error"]
        if total_for_prof > 0:
            counts["truncate_rate"] = round(
                counts["truncated"] / total_for_prof * 100, 2
            )

    elapsed.sort()
    p50 = elapsed[len(elapsed) // 2] if elapsed else None
    p95 = (
        elapsed[min(int(len(elapsed) * 0.95), len(elapsed) - 1)]
        if elapsed else None
    )

    return {
        "date": date,
        "total": len(rows),
        "by_profile": by_profile,
        "by_model": by_model,
        "p50_elapsed_ms": p50,
        "p95_elapsed_ms": p95,
    }


# ═══════════════════════════════════════════════════════════════
# Основной прокси
# ═══════════════════════════════════════════════════════════════

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):

    _t_total = time.time()
    target_url = f"{DOCLING_URL}/{path}"
    content_type = request.headers.get("content-type", "")
    client = request.app.state.client

    if "multipart/form-data" in content_type and "convert/file" in path:

        # Сквозной request_id для корреляции в логах, дампах и Langfuse.
        # Отправляется в docling как X-Request-Id (docling/LiteLLM могут
        # прокинуть его дальше в trace).
        _request_id = str(uuid.uuid4())
        _rid8 = _request_id[:8]
        # Синхронизируем с stats, чтобы в docling_requests.request_id
        # лежал тот же uuid, что и в dumps.
        _req_stats = getattr(request.state, "stats", None) if hasattr(request, "state") else None
        if isinstance(_req_stats, dict):
            _req_stats["request_id"] = uuid.UUID(_request_id)

        logger.info(f"[rid={_rid8}] START /convert/file")

        form = await request.form()

        do_pic_desc = form.get("do_picture_description", "").lower()
        do_pic_custom = form.get("do_picture_description_custom", "").lower()
        do_classification = form.get("do_picture_classification", "").lower()

        logger.info(f"РЕЖИМ do_pic_desc: {do_pic_desc}")
        logger.info(f"РЕЖИМ picture_description_custom: {do_pic_custom} и classification: {do_classification}")

        vlm_overrides = {}
        files = []
        data = []

        for key in form:
            field = form[key]
            if hasattr(field, "read"):
                content = await field.read()
                files.append(("files", (field.filename, content, field.content_type)))
            elif key.startswith("vlm_"):
                vlm_overrides[key] = str(field)
            else:
                data.append((key, str(field)))

        # ── Проверка поддерживаемых форматов ──
        for fi, (_, (fname, fbytes, ftype)) in enumerate(files):
            ext = os.path.splitext(fname)[1].lower() if fname else ""
            # Базовая разметка запроса для статистики — срабатывает только
            # если STATS_ENABLED=true (иначе request.state.stats отсутствует).
            _stats_set(request, filename=fname, file_size_bytes=len(fbytes) if fbytes else None)

            # Неподдерживаемый формат → дружелюбное сообщение
            if ext and ext not in SUPPORTED_EXTENSIONS:
                _stats_set(request, doc_type="UNSUPPORTED")
                logger.warning(f"UNSUPPORTED FORMAT: {fname} ({ext})")
                _total_ms = (time.time() - _t_total) * 1000
                logger.info(f"TIMING total: {_total_ms:.0f}ms  status: unsupported_format")
                resp_headers = {"content-type": "application/json"}
                return Response(
                    content=get_unsupported_response(fname),
                    status_code=422,
                    headers=resp_headers,
                )
            
            # .xls → конвертируем через xlrd в markdown и возвращаем сразу
            if ext == ".xls":
                _stats_set(request, doc_type="XLS", pipeline="xls_native")
                logger.info(f"XLS detected: {fname} -> converting via xlrd/pandas")
                _t_xls = time.time()
                xls_result = convert_xls_to_markdown(fbytes, fname)
                _xls_ms = (time.time() - _t_xls) * 1000
                if xls_result:
                    logger.info(f"TIMING xls_convert: {_xls_ms:.0f}ms")
                    _total_ms = (time.time() - _t_total) * 1000
                    logger.info(f"TIMING total: {_total_ms:.0f}ms  status: 200 (xls)")
                    resp_headers = {"content-type": "application/json"}
                    return Response(content=xls_result, status_code=200, headers=resp_headers)
                else:
                    logger.warning(f"XLS conversion failed, passing to docling")
            
            # .doc → Confluence MIME HTML или бинарный .doc
            if ext == ".doc":
                if is_confluence_doc(fbytes):
                    # Confluence export: MIME-encoded HTML → декодируем → подменяем на .html
                    _stats_set(request, doc_type="DOC_CONFLUENCE", pipeline="confluence_html")
                    logger.info(f"Confluence .doc detected: {fname}")
                    html_bytes, html_name = decode_confluence_doc(fbytes, fname)
                    if html_bytes:
                        files[fi] = ("files", (html_name, html_bytes, "text/html"))
                        logger.info(f"Confluence decode OK: {fname} -> {html_name} ({len(html_bytes)} bytes)")
                    else:
                        logger.error(f"Confluence decode FAILED: {fname} -> returning error")
                        _total_ms = (time.time() - _t_total) * 1000
                        error_msg = (
                            f"Не удалось извлечь HTML из файла «{fname}» (Confluence export). "
                            f"Попробуйте экспортировать документ из Confluence в формате PDF. "
                            f"Если возникнут вопросы — оставьте заявку: {SUPPORT_PORTAL_URL}"
                        )
                        return Response(
                            content=json.dumps({"detail": error_msg}, ensure_ascii=False).encode(),
                            status_code=422,
                            headers={"content-type": "application/json"},
                        )
                else:
                    # Бинарный .doc → Gotenberg → PDF → PyMuPDF → markdown
                    _stats_set(request, doc_type="DOC", pipeline="gotenberg")
                    logger.info(f"Binary .doc detected: {fname} -> converting via Gotenberg+PyMuPDF")
                    _t_doc = time.time()
                    doc_result = await convert_doc_to_markdown(client, fbytes, fname)
                    _doc_ms = (time.time() - _t_doc) * 1000
                    if doc_result:
                        logger.info(f"TIMING doc_convert: {_doc_ms:.0f}ms")
                        _total_ms = (time.time() - _t_total) * 1000
                        logger.info(f"TIMING total: {_total_ms:.0f}ms  status: 200 (doc)")
                        return Response(content=doc_result, status_code=200, headers={"content-type": "application/json"})
                    else:
                        logger.warning(f"DOC conversion failed, passing to docling")

        # ── Определяем pipeline (auto / vlm / standard) ──
        pipeline_value = None
        for key, val in data:
            if key == "pipeline":
                pipeline_value = val
                break

        # ── Автодетекция: если pipeline не задан или "auto" ──
        if pipeline_value in (None, "auto", ""):
            
            # Проверяем: есть ли PDF среди файлов?
            pdf_bytes_list = []
            for _, (fname, fbytes, ftype) in files:
                if fname and fname.lower().endswith(".pdf"):
                    pdf_bytes_list.append((fname, fbytes))

            if pdf_bytes_list:
                fname, fbytes = pdf_bytes_list[0]
                _t_detect = time.time()
                _is_scan = is_scan_pdf(fbytes)
                _detect_ms = (time.time() - _t_detect) * 1000
                logger.info(f"TIMING auto-detect: {_detect_ms:.0f}ms")
                # Подсчёт страниц для маршрутизации
                _page_count = 0
                try:
                    import fitz
                    _pdf_doc = fitz.open(stream=fbytes, filetype="pdf")
                    _page_count = len(_pdf_doc)
                    _pdf_doc.close()
                except Exception as e:
                    logger.warning(f"could not count pages: {e}")
                # Подсчёт картинок для предупреждения
                _image_count = count_pdf_images(fbytes) if not _is_scan else 0
                if _image_count > 0:
                    logger.info(f"PDF images: {_image_count} images in {_page_count} pages")
                
                pdf_type = "SCAN" if _is_scan else "TEXT PDF"
                _stats_set(request, file_pages=_page_count or None)
                _base_fname = fname.rsplit("/", 1)[-1] if "/" in fname else fname
                # Убираем UUID-префикс из имени файла для пользователя
                if "_" in _base_fname and len(_base_fname.split("_")[0]) == 36:
                    _base_fname = _base_fname.split("_", 1)[1]
                _processing_warning = get_processing_warning(_base_fname, _page_count, _image_count, _is_scan)
                if _processing_warning:
                    logger.warning(f"user-facing: {_processing_warning}")
                
                # ── SCAN PDF + OCR SDK ENABLED → новый путь v4.0 ──
                if _is_scan and OCR_SDK_ENABLED:
                    _stats_set(request, doc_type="SCAN", pipeline="ocr-sdk")
                    logger.info(f"Auto-detect: {fname} -> SCAN ({_page_count} pages) -> OCR SDK path (v4.0)")
                    sdk_result = await convert_scan_via_ocr_sdk(client, fbytes, fname, vlm_overrides)
                    if sdk_result is not None:
                        fixed_result = fix_katex_compatibility(sdk_result)
                        _total_ms = (time.time() - _t_total) * 1000
                        logger.info(f"TIMING total: {_total_ms:.0f}ms  status: 200 (ocr-sdk)")
                        return Response(
                            content=fixed_result,
                            status_code=200,
                            headers={"content-type": "application/json"},
                        )
                    else:
                        logger.warning("OCR SDK FALLBACK: SDK failed, falling back to VLM 122B full-page")

                # Маршрутизация: SCAN → всегда VLM, TEXT PDF > N стр. → standard.
                # Порог N настраивается из .env (TEXT_PDF_VLM_THRESHOLD) или
                # перекрывается per-request через form-field vlm_page_threshold.
                try:
                    VLM_PAGE_LIMIT = int(vlm_overrides.get("vlm_page_threshold", TEXT_PDF_VLM_THRESHOLD))
                except (TypeError, ValueError):
                    VLM_PAGE_LIMIT = TEXT_PDF_VLM_THRESHOLD
                if _is_scan:
                    pipeline_value = "vlm"
                    _stats_set(request, doc_type="SCAN", pipeline=pipeline_value)
                    logger.info(f"Auto-detect: {fname} -> {pdf_type} ({_page_count} pages) -> pipeline=vlm (scans always use VLM)")
                elif _page_count > VLM_PAGE_LIMIT:
                    pipeline_value = "standard"
                    _stats_set(request, doc_type="TEXT_LONG", pipeline=pipeline_value)
                    logger.info(f"Auto-detect: {fname} -> {pdf_type} ({_page_count} pages) -> pipeline=standard (>{VLM_PAGE_LIMIT} pages, text extractable)")
                else:
                    pipeline_value = "vlm"
                    _stats_set(request, doc_type="TEXT_SHORT", pipeline=pipeline_value)
                    logger.info(f"Auto-detect: {fname} -> {pdf_type} ({_page_count} pages) -> pipeline=vlm")
            else:
                # Не PDF (docx, xlsx и т.д.)
                _processing_warning = ""
                file_names = [fname for _, (fname, _, _) in files]
                
                # Проверяем: есть ли OLE-объекты (MathType формулы)?
                _has_ole = False
                _ole_file_idx = -1
                for fi, (_, (fname, fbytes, ftype)) in enumerate(files):
                    if has_ole_objects(fbytes, fname):
                        _has_ole = True
                        _ole_file_idx = fi
                        break
                
                if _has_ole:
                    # DOCX с OLE → Gotenberg (DOCX→PDF) → VLM pipeline
                    ole_fname = files[_ole_file_idx][1][0]
                    ole_bytes = files[_ole_file_idx][1][1]
                    logger.info(f"Auto-detect: {ole_fname} -> has OLE objects -> converting via Gotenberg")
                    try:
                        _t_gotenberg = time.time()
                        pdf_bytes = await convert_via_gotenberg(client, ole_bytes, ole_fname)
                        _gotenberg_ms = (time.time() - _t_gotenberg) * 1000
                        logger.info(f"TIMING gotenberg: {_gotenberg_ms:.0f}ms ({len(pdf_bytes)} bytes PDF)")
                        # Подменяем файл на сконвертированный PDF
                        pdf_name = ole_fname.rsplit(".", 1)[0] + ".pdf"
                        files[_ole_file_idx] = ("files", (pdf_name, pdf_bytes, "application/pdf"))
                        pipeline_value = "vlm"
                        _stats_set(request, doc_type="DOCX_OLE", pipeline=pipeline_value)
                        logger.info(f"Auto-detect: {ole_fname} -> OLE -> Gotenberg -> {pdf_name} -> pipeline=vlm")
                    except Exception as e:
                        logger.error(f"Gotenberg ERROR: {e} -> fallback to standard pipeline")
                        pipeline_value = "standard"
                        _stats_set(request, doc_type="DOCX_OLE", pipeline=pipeline_value)
                else:
                    pipeline_value = "standard"
                    _stats_set(request, doc_type="OTHER", pipeline=pipeline_value)
                    logger.info(f"Auto-detect: non-PDF {file_names} -> no OLE -> pipeline=standard")

        # ── Обновляем pipeline в data для docling ──
        data = [(k, v) for k, v in data if k != "pipeline"]
        data.append(("pipeline", pipeline_value))

        # ── Standard pipeline: отключаем OCR (Qwen3.5 VLM вместо PaddleOCR) ──
        if pipeline_value == "standard":
            data = [(k, v) for k, v in data if k != "do_ocr"]
            data.append(("do_ocr", "false"))
            logger.info("Standard Pipeline: OCR disabled, images via Qwen3.5 VLM")

            # images_scale — единственный параметр, реально влияющий на разрешение
            # картинок, отправляемых в picture_description_api (scale внутри API-блока
            # docling игнорирует). Приоритет:
            #   1) клиентский form-field images_scale (лежит в data, без префикса vlm_)
            #   2) vlm_overrides["images_scale"] (обратная совместимость, префикс vlm_)
            #   3) DEFAULT_IMAGES_SCALE из env — fallback.
            _client_images_scale = None
            for _k, _v in data:
                if _k == "images_scale":
                    _client_images_scale = _v
                    break
            if _client_images_scale is not None:
                _images_scale = _client_images_scale
                _scale_source = "client"
            elif "images_scale" in vlm_overrides:
                _images_scale = vlm_overrides["images_scale"]
                _scale_source = "vlm_overrides"
            else:
                _images_scale = str(DEFAULT_IMAGES_SCALE)
                _scale_source = "env_default"
            data = [(k, v) for k, v in data if k != "images_scale"]
            data.append(("images_scale", str(_images_scale)))
            logger.info(f"Standard Pipeline: images_scale={_images_scale} (source={_scale_source})")

        # ── VLM Pipeline: страница целиком -> Qwen3-VL -> markdown ──
        if pipeline_value == "vlm":
            
            keys_data = [k for k, _ in data]
            
            if "vlm_pipeline_model_api" not in keys_data:
                data.append(("vlm_pipeline_model_api", build_vlm_pipeline_model_api(vlm_overrides)))
                logger.info("VLM Pipeline: injected vlm_pipeline_model_api (Qwen3-VL full-page OCR)")
                
            # Workaround: VLM pipeline + embedded images = Pillow crash on some PDFs
            if "image_export_mode" not in keys_data:
                data.append(("image_export_mode", "placeholder"))
                logger.info("VLM Pipeline: выбран image_export_mode=placeholder")
                
            # VLM уже извлекает всё — picture description избыточен
            do_pic_desc = "false"
            
            data = [(k, v) for k, v in data if k not in ("do_picture_description", "do_picture_description_custom", "do_picture_classification")]
            data.append(("do_picture_description", "false"))
            data.append(("do_picture_description_custom", "false"))
            data.append(("do_picture_classification", "false"))
            logger.info("VLM Pipeline: suppressed picture_description and picture_classification (redundant with full-page VLM)")

        # ── Picture Description: описание картинок через VLM ──
        if do_pic_desc == "true":
            keys_data = [k for k, _ in data]
            # Всегда используем picture_description_api — поддерживает concurrency
            # picture_description_custom_config НЕ поддерживает параллельную обработку
            # (переход на custom_config был из-за thinking mode, но агентная инстанция + /no_think решает это)
            if "picture_description_api" not in keys_data:
                api_json = build_picture_description_api(vlm_overrides)
                _conc = int(vlm_overrides.get("vlm_concurrency", DEFAULT_VLM_CONCURRENCY))
                data.append(("picture_description_api", api_json))
                logger.info(f"Режим: picture_description_api (concurrency={_conc})")

        # Сохранение параметров для отладки
        save(data, files)

        max_docs = int(vlm_overrides.get(
            "vlm_max_concurrent_docs", DEFAULT_VLM_MAX_CONCURRENT_DOCS
        ))
        sem = get_semaphore(max_docs)

        multipart = []
        for key, val in data:
            multipart.append((key, (None, val)))
        multipart.extend(files)

        _docling_headers = {"X-Request-Id": _request_id}
        _t_queue = time.time()
        _queue_ms = 0.0
        _docling_ms = 0.0
        _attempts_made = 0
        _retry_reasons: list = []
        _last_exc = None
        resp = None
        # Retry ВНУТРИ семафора на каждой попытке: не пробиваем общий
        # cap=vlm_max_concurrent_docs, зато держим слот дольше при ретраях.
        # Повторяем только на httpx-исключениях и 502/503/504 (upstream-сбой);
        # 500 и 4xx — не ретраим (часто repeatable).
        for _attempt in range(1, DOCLING_RETRY_MAX_ATTEMPTS + 1):
            _attempts_made = _attempt
            try:
                async with sem:
                    if _attempt == 1:
                        _queue_ms = (time.time() - _t_queue) * 1000
                    _t_docling = time.time()
                    resp = await client.post(
                        target_url,
                        files=multipart,
                        headers=_docling_headers,
                        timeout=1200.0,
                    )
                    _docling_ms = (time.time() - _t_docling) * 1000
                # Ретраим на 502/503/504
                if resp.status_code in (502, 503, 504) and _attempt < DOCLING_RETRY_MAX_ATTEMPTS:
                    _reason = f"docling_{resp.status_code}"
                    _retry_reasons.append(_reason)
                    logger.warning(
                        f"[rid={_rid8}] attempt {_attempt}/{DOCLING_RETRY_MAX_ATTEMPTS}: "
                        f"{_reason}, retrying after {DOCLING_RETRY_BACKOFF_SEC}s"
                    )
                    await asyncio.sleep(DOCLING_RETRY_BACKOFF_SEC)
                    continue
                break  # success/4xx/500/нечего ретраить — выходим
            except (httpx.TimeoutException, httpx.RequestError) as _exc:
                _last_exc = _exc
                _reason = _classify_error_type(_exc, None)
                _retry_reasons.append(_reason)
                if _attempt < DOCLING_RETRY_MAX_ATTEMPTS:
                    logger.warning(
                        f"[rid={_rid8}] attempt {_attempt}/{DOCLING_RETRY_MAX_ATTEMPTS}: "
                        f"{_reason} ({type(_exc).__name__}: {_exc}), "
                        f"retrying after {DOCLING_RETRY_BACKOFF_SEC}s"
                    )
                    await asyncio.sleep(DOCLING_RETRY_BACKOFF_SEC)
                    continue
                # Попытки кончились — выпадаем ниже и отдадим дамп + заглушку
                resp = None
                break

        # Если все попытки упали с httpx-исключением — нет resp, отдаём дамп.
        if resp is None and _last_exc is not None:
            _total_ms = (time.time() - _t_total) * 1000
            _err_type = _classify_error_type(_last_exc, None)
            _http_returned = 504 if "timeout" in _err_type else 502
            _params_dict = {k: v for k, v in data}
            _file_names = [fn for _, (fn, _, _) in files]
            _client_ip = request.client.host if request.client else None
            _log_path = _save_error_response_log(
                error_type=_err_type,
                http_status_returned=_http_returned,
                docling_status_code=None,
                docling_response_body=None,
                exception=_last_exc,
                params_dict=_params_dict,
                file_names=_file_names,
                client_ip=_client_ip,
                duration_ms=int(_total_ms),
                request_id=_request_id,
                attempts_made=_attempts_made,
                retry_reasons=_retry_reasons,
            )
            logger.error(
                f"[ERROR-RESPONSE] [rid={_rid8}] saved to {_log_path}  "
                f"http={_http_returned}  error_type={_err_type}  "
                f"attempts={_attempts_made}  duration_ms={int(_total_ms)}  "
                f"exc={type(_last_exc).__name__}: {_last_exc}"
            )
            _stats_set(
                request,
                http_status=_http_returned,
                error_message=f"{_err_type}: {type(_last_exc).__name__}: {_last_exc}",
                duration_total_ms=int(_total_ms),
            )
            return Response(
                content=json.dumps(
                    {
                        "status": "failure",
                        "document": {"md_content": make_null_response_markdown(_request_id)},
                        "proxy_diagnostics": {
                            "error_type": _err_type,
                            "error_response_log": _log_path,
                            "reason": f"{type(_last_exc).__name__}: {_last_exc}",
                            "request_id": _request_id,
                            "attempts_made": _attempts_made,
                            "retry_reasons": _retry_reasons,
                        },
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
                status_code=_http_returned,
                headers={"content-type": "application/json"},
            )

        logger.info(
            f"TIMING queue_wait: {_queue_ms:.0f}ms  "
            f"docling_request: {_docling_ms:.0f}ms  "
            f"attempts={_attempts_made}"
        )

        _total_ms = (time.time() - _t_total) * 1000
        logger.info(f"[rid={_rid8}] TIMING total: {_total_ms:.0f}ms  status: {resp.status_code}  attempts={_attempts_made}")

        # Дамп для docling 5xx — чтобы видеть upstream-сбои
        if 500 <= resp.status_code < 600:
            _err_type = _classify_error_type(None, resp.status_code)
            _params_dict = {k: v for k, v in data}
            _file_names = [fn for _, (fn, _, _) in files]
            _client_ip = request.client.host if request.client else None
            _log_path = _save_error_response_log(
                error_type=_err_type,
                http_status_returned=resp.status_code,
                docling_status_code=resp.status_code,
                docling_response_body=resp.content,
                exception=None,
                params_dict=_params_dict,
                file_names=_file_names,
                client_ip=_client_ip,
                duration_ms=int(_total_ms),
                request_id=_request_id,
                attempts_made=_attempts_made,
                retry_reasons=_retry_reasons,
            )
            logger.error(
                f"[ERROR-RESPONSE] [rid={_rid8}] saved to {_log_path}  "
                f"http={resp.status_code}  error_type={_err_type}  "
                f"attempts={_attempts_made}  duration_ms={int(_total_ms)}"
            )

        # Финальная разметка для статистики (только docling-ветка)
        _stats_set(
            request,
            duration_queue_wait_ms=int(_queue_ms),
            duration_docling_ms=int(_docling_ms),
        )
        if resp.status_code == 200:
            try:
                _docling_body = json.loads(resp.content)
                if isinstance(_docling_body, dict) and "timings" in _docling_body:
                    _stats_set(request, timings_json=_docling_body["timings"])
            except Exception:
                pass
        else:
            try:
                _stats_set(request, error_message=resp.content[:500].decode("utf-8", "replace"))
            except Exception:
                pass
        
        # Пост-обработка: (1) fix_vlm_truncation на сыром ответе — обрезка по
        # [END_OF_PAGE] / fuzzy-варианту / хвосту пустых строк. (2) Детекция
        # null/partial_success → дамп в null_response_*.json + подстановка
        # заглушки, чтобы OWUI не падал на Pydantic-ошибке. (3) KaTeX fix.
        # При не-200 — пропускаем.
        if resp.status_code == 200:
            _vlm_fixed, _vlm_info = fix_vlm_truncation(resp.content)

            # Диагностика + заглушка для null/partial_success ответов.
            try:
                _d_data = json.loads(_vlm_fixed)
            except Exception:
                _d_data = None
            _d_doc = _d_data.get("document") if isinstance(_d_data, dict) else None
            _d_md = _d_doc.get("md_content") if isinstance(_d_doc, dict) else None
            _d_status = _d_data.get("status") if isinstance(_d_data, dict) else None
            _d_errors = _d_data.get("errors") if isinstance(_d_data, dict) else []

            _null_trigger = (
                _d_md is None
                or (isinstance(_d_md, str) and _d_md == "")
                or _d_status in ("partial_success", "failure", "skipped")
            )
            if _null_trigger:
                _params_dict = {k: v for k, v in data}
                _file_names = [fn for _, (fn, _, _) in files]
                _client_ip = request.client.host if request.client else None
                _log_path = _save_null_response_log(
                    resp_content=resp.content,
                    http_status=resp.status_code,
                    status_field=_d_status,
                    md_content=_d_md,
                    errors=_d_errors or [],
                    fix_info=_vlm_info,
                    params_dict=_params_dict,
                    file_names=_file_names,
                    client_ip=_client_ip,
                    total_ms=int(_total_ms),
                    request_id=_request_id,
                    attempts_made=_attempts_made,
                    retry_reasons=_retry_reasons,
                )
                _md_len_desc = "null" if _d_md is None else str(len(_d_md)) if isinstance(_d_md, str) else "?"
                _err_count = len(_d_errors) if isinstance(_d_errors, list) else 0
                logger.warning(
                    f"[NULL-RESPONSE] [rid={_rid8}] saved to {_log_path}  "
                    f"status={_d_status}  md_len={_md_len_desc}  "
                    f"errors_count={_err_count}  attempts={_attempts_made}"
                )
                # Подменяем ответ на валидную заглушку
                if isinstance(_d_data, dict) and isinstance(_d_doc, dict):
                    _d_doc["md_content"] = make_null_response_markdown(_request_id)
                    _d_data["document"] = _d_doc
                    _d_data["status"] = "success"
                    _d_data["proxy_diagnostics"] = {
                        "original_status": _d_status,
                        "null_response_log": _log_path,
                        "reason": "md_content was null or empty",
                        "request_id": _request_id,
                        "attempts_made": _attempts_made,
                        "retry_reasons": _retry_reasons,
                    }
                    try:
                        _vlm_fixed = json.dumps(_d_data, ensure_ascii=False).encode("utf-8")
                    except Exception as _e:
                        logger.error(f"[NULL-RESPONSE] failed to re-serialize stub response: {_e}")
                # Для статистики — пометим, но не ломаем http_status
                _stats_set(request, error_message=f"null_response: original_status={_d_status}")

            fixed_content = fix_katex_compatibility(_vlm_fixed)
        else:
            fixed_content = resp.content
        
        # Убираем Content-Length — он мог измениться после KaTeX fix
        resp_headers = dict(resp.headers)
        resp_headers.pop("content-length", None)
        resp_headers.pop("Content-Length", None)
        
        return Response(
            content=fixed_content,
            status_code=resp.status_code,
            headers=resp_headers,
        )

    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)

    resp = await client.request(
        method=request.method,
        url=target_url,
        headers=headers,
        content=body,
        timeout=660.0,
    )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )