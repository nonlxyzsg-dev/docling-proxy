"""VLM endpoint: /v1/chat/completions with sampling injection and JSONL logging."""
import os, json, base64, asyncio, time, uuid, logging, httpx
from datetime import datetime as dt
from fastapi import Request, Response, APIRouter
from proxy.config import (
    VLM_FULL_PAGE_SAMPLING, VLM_PICTURE_DESC_SAMPLING,
    VLM_UPSTREAM_URL, VLM_UPSTREAM_API_KEY,
    VLM_REQUEST_LOG_FILE, VLM_TRUNCATE_LOG_DIR, VLM_TRUNCATE_SAVE_PAYLOAD,
)
from proxy.prompts import VLM_FULL_PAGE_PROMPT, VLM_PICTURE_DESC_PROMPT
from proxy.error_handling import _classify_error_type

logger = logging.getLogger("docling_proxy")

_VLM_OPENAI_SAMPLING_KEYS = {
    "temperature", "top_p", "top_k", "min_p",
    "presence_penalty", "frequency_penalty", "repetition_penalty",
    "max_tokens", "max_completion_tokens",
}


def _vlm_request_log_path_for(date_str: str) -> str:
    """`vlm_requests_<DATE>.jsonl` рядом с базовым именем VLM_REQUEST_LOG_FILE."""
    base_dir = os.path.dirname(VLM_REQUEST_LOG_FILE) or "."
    base = os.path.basename(VLM_REQUEST_LOG_FILE) or "vlm_requests.jsonl"
    stem, ext = os.path.splitext(base)
    if not ext:
        ext = ".jsonl"
    return os.path.join(base_dir, f"{stem}_{date_str}{ext}")


def _vlm_log_request(record: dict) -> None:
    """Append-only запись одной строки в дневной JSONL."""
    try:
        os.makedirs(os.path.dirname(VLM_REQUEST_LOG_FILE) or ".", exist_ok=True)
        date_str = dt.utcnow().strftime("%Y-%m-%d")
        path = _vlm_request_log_path_for(date_str)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.error(f"[vlm-log] failed to write: {e}")


def _vlm_strip_images_in_place(body: dict, save_to_dir: "str | None") -> int:
    """Извлечь все base64-картинки из messages[].content[] и подменить URL."""
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
    """Синхронный дамп truncate-кейса."""
    try:
        date_str = dt.utcnow().strftime("%Y-%m-%d")
        out_dir = os.path.join(VLM_TRUNCATE_LOG_DIR, date_str, request_id)
        os.makedirs(out_dir, exist_ok=True)

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


def _vlm_inject_sampling(body: dict, profile_sampling: dict) -> None:
    """Положить значения из profile_sampling в body, если клиент их не задал."""
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


router = APIRouter()


@router.post("/v1/chat/completions")
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

    _ct = completion_tokens if completion_tokens is not None else "-"
    _pt = prompt_tokens if prompt_tokens is not None else "-"
    _fr = finish_reason if finish_reason is not None else "-"
    if status == "truncated":
        dump_dir = os.path.join(
            VLM_TRUNCATE_LOG_DIR, dt.utcnow().strftime("%Y-%m-%d"), request_id
        )
        max_tokens_requested = body.get("max_tokens") or body.get("max_completion_tokens")
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
