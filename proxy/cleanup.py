"""Filesystem cleanup: inbox, vlm_requests, null/error responses, truncate dumps."""
import os, time, glob, asyncio, logging
from datetime import datetime as dt
from proxy.config import (
    OCR_SDK_INBOX_CONTAINER, LOG_DIR,
    VLM_REQUEST_LOG_FILE, VLM_REQUEST_LOG_RETENTION_DAYS,
    VLM_REQUEST_LOG_MAX_SIZE_MB,
    NULL_RESPONSE_LOG_RETENTION_DAYS, ERROR_RESPONSE_LOG_RETENTION_DAYS,
    VLM_TRUNCATE_LOG_DIR, VLM_TRUNCATE_RETENTION_DAYS,
)

logger = logging.getLogger("docling_proxy")

_NULL_RESPONSE_LOG_PATTERN = "null_response_*.json"
_ERROR_RESPONSE_LOG_PATTERN = "error_response_*.json"


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
