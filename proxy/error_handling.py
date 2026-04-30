"""Error handling: null responses, error dumps, friendly stubs."""
import os, json, traceback, logging, httpx
from datetime import datetime as dt
from proxy.config import LOG_DIR

logger = logging.getLogger("docling_proxy")


def make_null_response_markdown(request_id: str = "") -> str:
    """Дружелюбная заглушка для OWUI с указанием времени и request_id."""
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
