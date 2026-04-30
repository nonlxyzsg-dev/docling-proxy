"""docling-serve dispatch: retry, response post-processing, null/error logging.

Extracted from proxy_handler.py to keep individual modules pushable through
GitHub MCP without hitting stream-idle timeout. Behaviour identical to inline.
"""
import os, json, time, asyncio, logging, httpx
from fastapi import Request, Response
from proxy.config import (
    DOCLING_RETRY_MAX_ATTEMPTS, DOCLING_RETRY_BACKOFF_SEC,
)
from proxy.error_handling import (
    _classify_error_type, _save_null_response_log,
    _save_error_response_log, make_null_response_markdown,
)
from proxy.post_process import fix_vlm_truncation, fix_katex_compatibility
from proxy.stats import _stats_set
from proxy.http_client import get_semaphore

logger = logging.getLogger("docling_proxy")


async def run_docling_request(
    *,
    client: httpx.AsyncClient,
    request: Request,
    target_url: str,
    multipart: list,
    data: list,
    files: list,
    _request_id: str,
    _rid8: str,
    _t_total: float,
    max_docs: int,
) -> Response:
    """Run docling-serve request with retry and full response post-processing.

    Identical behaviour to the inline version it replaces in proxy_handler.proxy().
    """
    sem = get_semaphore(max_docs)

    _docling_headers = {"X-Request-Id": _request_id}
    _t_queue = time.time()
    _queue_ms = 0.0
    _docling_ms = 0.0
    _attempts_made = 0
    _retry_reasons: list = []
    _last_exc = None
    resp = None
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
            if resp.status_code in (502, 503, 504) and _attempt < DOCLING_RETRY_MAX_ATTEMPTS:
                _reason = f"docling_{resp.status_code}"
                _retry_reasons.append(_reason)
                logger.warning(
                    f"[rid={_rid8}] attempt {_attempt}/{DOCLING_RETRY_MAX_ATTEMPTS}: "
                    f"{_reason}, retrying after {DOCLING_RETRY_BACKOFF_SEC}s"
                )
                await asyncio.sleep(DOCLING_RETRY_BACKOFF_SEC)
                continue
            break
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
            resp = None
            break

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

    if resp.status_code == 200:
        _vlm_fixed, _vlm_info = fix_vlm_truncation(resp.content)

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
            _stats_set(request, error_message=f"null_response: original_status={_d_status}")

        fixed_content = fix_katex_compatibility(_vlm_fixed)
    else:
        fixed_content = resp.content

    resp_headers = dict(resp.headers)
    resp_headers.pop("content-length", None)
    resp_headers.pop("Content-Length", None)

    return Response(
        content=fixed_content,
        status_code=resp.status_code,
        headers=resp_headers,
    )
