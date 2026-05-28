"""Main proxy handler for /convert/file (and catch-all forwarder)."""
import os, json, time, uuid, asyncio, logging, fitz, httpx
from fastapi import Request, Response, APIRouter
from proxy.config import (
    DOCLING_URL, TEXT_PDF_VLM_THRESHOLD, TEXT_PDF_VLM_THRESHOLD_SOURCE,
    SCAN_PDF_FULL_PAGE, SCAN_PDF_FULL_PAGE_SOURCE,
    DEFAULT_VLM_MAX_CONCURRENT_DOCS, DEFAULT_VLM_CONCURRENCY,
    DEFAULT_IMAGES_SCALE,
    DOCLING_RETRY_MAX_ATTEMPTS, DOCLING_RETRY_BACKOFF_SEC,
    OCR_SDK_ENABLED, ARCHIVE_PROCESSING_ENABLED,
    _resolve_int_threshold, _resolve_bool_flag,
)
from proxy.routing import (
    is_scan_pdf, count_pdf_images, has_ole_objects,
    is_confluence_doc, decode_confluence_doc,
    convert_via_gotenberg, get_processing_warning,
)
from proxy.pipelines import (
    SUPPORTED_EXTENSIONS, SUPPORT_PORTAL_URL, get_unsupported_response,
    convert_xls_to_markdown, convert_doc_to_markdown,
    convert_scan_via_ocr_sdk, save,
)
from proxy.builders import (
    build_picture_description_api, build_vlm_pipeline_model_api,
)
from proxy.post_process import fix_katex_compatibility
from proxy.stats import _stats_set
from proxy.dispatch import run_docling_request
from proxy.archive_extract import is_archive
from proxy.archive import handle_archive


logger = logging.getLogger("docling_proxy")


router = APIRouter()


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):

    _t_total = time.time()
    target_url = f"{DOCLING_URL}/{path}"
    content_type = request.headers.get("content-type", "")
    client = request.app.state.client

    if "multipart/form-data" in content_type and "convert/file" in path:

        _request_id = str(uuid.uuid4())
        _rid8 = _request_id[:8]
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
        routing_overrides = {}
        files = []
        data = []

        for key in form:
            field = form[key]
            if hasattr(field, "read"):
                content = await field.read()
                files.append(("files", (field.filename, content, field.content_type)))
            elif key.startswith("vlm_"):
                vlm_overrides[key] = str(field)
            elif key == "scan_pdf_full_page":
                routing_overrides[key] = str(field)
            else:
                data.append((key, str(field)))

        # proxy_skip: клиент явно говорит «не обрабатывай файл, в docling не ходи».
        # Принимаем truthy-значения (true/1/yes/on), сам параметр в docling не пробрасываем.
        # Ответ — стандартный docling-формат с пустым md_content, чтобы OWUI не ломался.
        _proxy_skip_raw = ""
        for _k, _v in data:
            if _k == "proxy_skip":
                _proxy_skip_raw = _v
                break
        data = [(k, v) for k, v in data if k != "proxy_skip"]
        if _proxy_skip_raw.strip().lower() in ("true", "1", "yes", "on"):
            _fname = files[0][1][0] if files else "<no file>"
            _stats_set(request, doc_type="SKIPPED", pipeline="skip")
            _total_ms = (time.time() - _t_total) * 1000
            logger.info(f"[rid={_rid8}] Proxy skip: {_fname} -> not forwarded to docling")
            logger.info(f"TIMING total: {_total_ms:.0f}ms  status: 200 (proxy_skip)")
            return Response(
                content=json.dumps(
                    {
                        "status": "success",
                        "document": {"md_content": ""},
                        "proxy_diagnostics": {
                            "proxy_skipped": True,
                            "request_id": _request_id,
                        },
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
                status_code=200,
                headers={"content-type": "application/json"},
            )

        # Распаковка архивов: если среди загруженных файлов есть архив
        # (zip/tar/7z/rar) — разворачиваем его рекурсивно и обрабатываем каждый
        # вложенный документ как отдельную загрузку, склеивая результат в один
        # markdown. Подробно — proxy/archive.py, README.md раздел «Архивы».
        if ARCHIVE_PROCESSING_ENABLED and any(
            is_archive(f[1][0], f[1][1]) for f in files
        ):
            return await handle_archive(
                client=client, request=request, target_url=target_url,
                files=files, data=data,
                vlm_overrides=vlm_overrides, routing_overrides=routing_overrides,
                rid8=_rid8, request_id=_request_id, t_total=_t_total,
            )

        for fi, (_, (fname, fbytes, ftype)) in enumerate(files):
            ext = os.path.splitext(fname)[1].lower() if fname else ""
            _stats_set(request, filename=fname, file_size_bytes=len(fbytes) if fbytes else None)

            if ext and ext not in SUPPORTED_EXTENSIONS:
                _stats_set(request, doc_type="UNSUPPORTED")
                logger.warning(f"UNSUPPORTED FORMAT: {fname} ({ext})")
                _total_ms = (time.time() - _t_total) * 1000
                logger.info(f"TIMING total: {_total_ms:.0f}ms  status: unsupported_format")
                return Response(
                    content=get_unsupported_response(fname),
                    status_code=422,
                    headers={"content-type": "application/json"},
                )

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
                    return Response(content=xls_result, status_code=200, headers={"content-type": "application/json"})
                else:
                    logger.warning(f"XLS conversion failed, passing to docling")

            if ext == ".doc":
                if is_confluence_doc(fbytes):
                    _stats_set(request, doc_type="DOC_CONFLUENCE", pipeline="confluence_html")
                    logger.info(f"Confluence .doc detected: {fname}")
                    html_bytes, html_name = decode_confluence_doc(fbytes, fname)
                    if html_bytes:
                        files[fi] = ("files", (html_name, html_bytes, "text/html"))
                        logger.info(f"Confluence decode OK: {fname} -> {html_name} ({len(html_bytes)} bytes)")
                    else:
                        logger.error(f"Confluence decode FAILED: {fname} -> returning error")
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

        pipeline_value = None
        for key, val in data:
            if key == "pipeline":
                pipeline_value = val
                break

        if pipeline_value in (None, "auto", ""):
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
                _page_count = 0
                try:
                    _pdf_doc = fitz.open(stream=fbytes, filetype="pdf")
                    _page_count = len(_pdf_doc)
                    _pdf_doc.close()
                except Exception as e:
                    logger.warning(f"could not count pages: {e}")
                _image_count = count_pdf_images(fbytes) if not _is_scan else 0
                if _image_count > 0:
                    logger.info(f"PDF images: {_image_count} images in {_page_count} pages")

                pdf_type = "SCAN" if _is_scan else "TEXT PDF"
                _stats_set(request, file_pages=_page_count or None)
                _base_fname = fname.rsplit("/", 1)[-1] if "/" in fname else fname
                if "_" in _base_fname and len(_base_fname.split("_")[0]) == 36:
                    _base_fname = _base_fname.split("_", 1)[1]
                _processing_warning = get_processing_warning(_base_fname, _page_count, _image_count, _is_scan)
                if _processing_warning:
                    logger.warning(f"user-facing: {_processing_warning}")

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

                _vpt, _vpt_src = _resolve_int_threshold(
                    vlm_overrides.get("vlm_page_threshold"),
                    TEXT_PDF_VLM_THRESHOLD, TEXT_PDF_VLM_THRESHOLD_SOURCE,
                    "vlm_page_threshold", _rid8,
                )
                _scan_full, _scan_full_src = _resolve_bool_flag(
                    routing_overrides.get("scan_pdf_full_page"),
                    SCAN_PDF_FULL_PAGE, SCAN_PDF_FULL_PAGE_SOURCE,
                    "scan_pdf_full_page", _rid8,
                )

                if _is_scan:
                    pipeline_value = "vlm" if _scan_full else "standard"
                    _stats_set(request, doc_type="SCAN", pipeline=pipeline_value)
                elif _vpt > 0 and _page_count <= _vpt:
                    pipeline_value = "vlm"
                    _stats_set(request, doc_type="TEXT_SHORT", pipeline=pipeline_value)
                else:
                    pipeline_value = "standard"
                    _stats_set(request, doc_type="TEXT_LONG", pipeline=pipeline_value)
                logger.info(
                    f"Auto-detect: {fname} -> {pdf_type} ({_page_count} pages) -> "
                    f"pipeline={pipeline_value} "
                    f"(scan_pdf_full_page={_scan_full} source={_scan_full_src}, "
                    f"vlm_page_threshold={_vpt} source={_vpt_src})"
                )
            else:
                file_names = [fname for _, (fname, _, _) in files]
                _has_ole = False
                _ole_file_idx = -1
                for fi, (_, (fname, fbytes, ftype)) in enumerate(files):
                    if has_ole_objects(fbytes, fname):
                        _has_ole = True
                        _ole_file_idx = fi
                        break

                if _has_ole:
                    ole_fname = files[_ole_file_idx][1][0]
                    ole_bytes = files[_ole_file_idx][1][1]
                    logger.info(f"Auto-detect: {ole_fname} -> has OLE objects -> converting via Gotenberg")
                    try:
                        _t_gotenberg = time.time()
                        pdf_bytes = await convert_via_gotenberg(client, ole_bytes, ole_fname)
                        _gotenberg_ms = (time.time() - _t_gotenberg) * 1000
                        logger.info(f"TIMING gotenberg: {_gotenberg_ms:.0f}ms ({len(pdf_bytes)} bytes PDF)")
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

        data = [(k, v) for k, v in data if k != "pipeline"]
        data.append(("pipeline", pipeline_value))

        if pipeline_value == "standard":
            data = [(k, v) for k, v in data if k != "do_ocr"]
            data.append(("do_ocr", "false"))
            logger.info("Standard Pipeline: OCR disabled, images via Qwen3.5 VLM")

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

        if pipeline_value == "vlm":
            keys_data = [k for k, _ in data]
            if "vlm_pipeline_model_api" not in keys_data:
                data.append(("vlm_pipeline_model_api", build_vlm_pipeline_model_api(vlm_overrides)))
                logger.info("VLM Pipeline: injected vlm_pipeline_model_api (Qwen3-VL full-page OCR)")
            if "image_export_mode" not in keys_data:
                data.append(("image_export_mode", "placeholder"))
                logger.info("VLM Pipeline: выбран image_export_mode=placeholder")
            do_pic_desc = "false"
            data = [(k, v) for k, v in data if k not in ("do_picture_description", "do_picture_description_custom", "do_picture_classification")]
            data.append(("do_picture_description", "false"))
            data.append(("do_picture_description_custom", "false"))
            data.append(("do_picture_classification", "false"))
            logger.info("VLM Pipeline: suppressed picture_description and picture_classification (redundant with full-page VLM)")

        if do_pic_desc == "true":
            keys_data = [k for k, _ in data]
            if "picture_description_api" not in keys_data:
                api_json = build_picture_description_api(vlm_overrides)
                _conc = int(vlm_overrides.get("vlm_concurrency", DEFAULT_VLM_CONCURRENCY))
                data.append(("picture_description_api", api_json))
                logger.info(f"Режим: picture_description_api (concurrency={_conc})")

        save(data, files)

        max_docs = int(vlm_overrides.get(
            "vlm_max_concurrent_docs", DEFAULT_VLM_MAX_CONCURRENT_DOCS
        ))

        multipart = []
        for key, val in data:
            multipart.append((key, (None, val)))
        multipart.extend(files)

        return await run_docling_request(
            client=client,
            request=request,
            target_url=target_url,
            multipart=multipart,
            data=data,
            files=files,
            _request_id=_request_id,
            _rid8=_rid8,
            _t_total=_t_total,
            max_docs=max_docs,
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
