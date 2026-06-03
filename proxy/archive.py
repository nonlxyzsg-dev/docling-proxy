"""Обработка загруженных архивов.

Разворачивает архив (рекурсивно, включая вложенные — см. archive_extract.py),
прогоняет каждый извлечённый документ через тот же роутинг, что и обычная
загрузка в чат (XLS/DOC/PDF-scan/DOCX-OLE/standard/vlm), и склеивает markdown
всех файлов в один ответ docling-формата.

Workaround поверх docling: docling-serve принимает только одиночные документы,
не архивы. Когда форк docling научится разворачивать архивы нативно — этот
модуль можно будет удалить.
"""
import os
import json
import time
import mimetypes
import logging

import fitz
from fastapi import Request, Response

from proxy.config import (
    DEFAULT_IMAGES_SCALE, OCR_SDK_ENABLED,
    TEXT_PDF_VLM_THRESHOLD, TEXT_PDF_VLM_THRESHOLD_SOURCE,
    SCAN_PDF_FULL_PAGE, SCAN_PDF_FULL_PAGE_SOURCE,
    DEFAULT_VLM_MAX_CONCURRENT_DOCS,
    _resolve_int_threshold, _resolve_bool_flag,
)
from proxy.routing import (
    is_scan_pdf, has_ole_objects, convert_via_gotenberg,
    is_confluence_doc, decode_confluence_doc,
)
from proxy.pipelines import (
    SUPPORTED_EXTENSIONS, convert_xls_to_markdown, convert_doc_to_markdown,
    convert_scan_via_ocr_sdk,
)
from proxy.builders import (
    build_picture_description_api, build_vlm_pipeline_model_api,
)
from proxy.post_process import fix_katex_compatibility
from proxy.stats import _stats_set
from proxy.dispatch import run_docling_request
from proxy.archive_extract import is_archive, extract_archive, archive_kind

logger = logging.getLogger("docling_proxy")

# Поля формы, которые прокси инжектирует сам (зависят от выбранного пайплайна).
# Снимаем их из исходного payload и пересобираем под каждый извлечённый файл.
_CONTROL_KEYS = {
    "pipeline", "do_ocr", "images_scale",
    "vlm_pipeline_model_api", "image_export_mode", "picture_description_api",
    "do_picture_description", "do_picture_description_custom",
    "do_picture_classification",
}


def _md_from_response(resp: Response) -> str | None:
    """Достать document.md_content из ответа run_docling_request."""
    try:
        body = json.loads(bytes(resp.body))
        doc = body.get("document") if isinstance(body, dict) else None
        if isinstance(doc, dict):
            return doc.get("md_content")
    except Exception:
        pass
    return None


def _md_from_json_bytes(raw: bytes) -> str | None:
    """Достать md_content из JSON-байтов локальных конвертеров (xls/doc/ocr-sdk)."""
    try:
        body = json.loads(raw)
        doc = body.get("document") if isinstance(body, dict) else None
        if isinstance(doc, dict):
            return doc.get("md_content")
    except Exception:
        pass
    return None


def _inject_pipeline_params(data: list, pipeline_value: str,
                            vlm_overrides: dict, do_pic_desc: str) -> list:
    """Пересобрать form-поля под выбранный пайплайн (зеркало proxy_handler)."""
    data = [(k, v) for k, v in data if k != "pipeline"]
    data.append(("pipeline", pipeline_value))

    if pipeline_value == "standard":
        data = [(k, v) for k, v in data if k != "do_ocr"]
        data.append(("do_ocr", "false"))
        _scale = None
        for k, v in data:
            if k == "images_scale":
                _scale = v
                break
        if _scale is None:
            _scale = vlm_overrides.get("images_scale", str(DEFAULT_IMAGES_SCALE))
        data = [(k, v) for k, v in data if k != "images_scale"]
        data.append(("images_scale", str(_scale)))

    if pipeline_value == "vlm":
        keys = [k for k, _ in data]
        if "vlm_pipeline_model_api" not in keys:
            data.append(("vlm_pipeline_model_api", build_vlm_pipeline_model_api(vlm_overrides)))
        if "image_export_mode" not in keys:
            data.append(("image_export_mode", "placeholder"))
        data = [(k, v) for k, v in data
                if k not in ("do_picture_description",
                             "do_picture_description_custom",
                             "do_picture_classification")]
        data.append(("do_picture_description", "false"))
        data.append(("do_picture_description_custom", "false"))
        data.append(("do_picture_classification", "false"))
        do_pic_desc = "false"

    if do_pic_desc == "true":
        keys = [k for k, _ in data]
        if "picture_description_api" not in keys:
            data.append(("picture_description_api", build_picture_description_api(vlm_overrides)))

    return data


async def _convert_member(
    *, client, request: Request, target_url: str,
    fname: str, fbytes: bytes,
    base_data: list, do_pic_desc: str,
    vlm_overrides: dict, routing_overrides: dict,
    rid8: str, request_id: str, idx: int, max_docs: int,
) -> dict:
    """Обработать один извлечённый файл как обычную загрузку.

    Возвращает dict: filename, status (ok|unsupported|error), md, note,
    size (исходный размер в байтах), pages (число страниц, где дёшево —
    PDF и docx→PDF, иначе None).
    Зеркалит роутинг proxy_handler.proxy(), но отдаёт markdown, а не Response.
    """
    ext = os.path.splitext(fname)[1].lower() if fname else ""
    _orig_size = len(fbytes) if fbytes else 0
    _pages = None

    def _r(status, md=None, note=None):
        return {"filename": fname, "status": status, "md": md, "note": note,
                "size": _orig_size, "pages": _pages}

    if ext and ext not in SUPPORTED_EXTENSIONS:
        logger.info(f"[rid={rid8}] archive member {fname}: unsupported ({ext}), skip")
        return _r("unsupported")

    # XLS — нативная конвертация.
    if ext == ".xls":
        res = convert_xls_to_markdown(fbytes, fname)
        md = _md_from_json_bytes(res) if res else None
        if md is not None:
            return _r("ok", md=md)
        logger.warning(f"[rid={rid8}] archive member {fname}: XLS convert failed")

    # DOC — Confluence-HTML или бинарный .doc через Gotenberg.
    if ext == ".doc":
        if is_confluence_doc(fbytes):
            html_bytes, html_name = decode_confluence_doc(fbytes, fname)
            if html_bytes:
                fname, fbytes, ext = html_name, html_bytes, ".html"
            else:
                return _r("error", note="не удалось извлечь HTML из Confluence-экспорта.")
        else:
            res = await convert_doc_to_markdown(client, fbytes, fname)
            md = _md_from_json_bytes(res) if res else None
            if md is not None:
                return _r("ok", md=md)
            logger.warning(f"[rid={rid8}] archive member {fname}: DOC convert failed")

    # Выбор пайплайна.
    pipeline_value = "standard"
    if fname.lower().endswith(".pdf"):
        _is_scan = is_scan_pdf(fbytes)
        _page_count = 0
        try:
            _pdf = fitz.open(stream=fbytes, filetype="pdf")
            _page_count = len(_pdf)
            _pdf.close()
        except Exception as e:
            logger.warning(f"[rid={rid8}] archive member {fname}: page count failed: {e}")
        _pages = _page_count or None

        if _is_scan and OCR_SDK_ENABLED:
            sdk_res = await convert_scan_via_ocr_sdk(client, fbytes, fname, vlm_overrides)
            if sdk_res is not None:
                md = _md_from_json_bytes(fix_katex_compatibility(sdk_res))
                if md is not None:
                    return _r("ok", md=md)
            logger.warning(f"[rid={rid8}] archive member {fname}: OCR SDK fallback")

        _vpt, _ = _resolve_int_threshold(
            vlm_overrides.get("vlm_page_threshold"),
            TEXT_PDF_VLM_THRESHOLD, TEXT_PDF_VLM_THRESHOLD_SOURCE,
            "vlm_page_threshold", rid8,
        )
        _scan_full, _ = _resolve_bool_flag(
            routing_overrides.get("scan_pdf_full_page"),
            SCAN_PDF_FULL_PAGE, SCAN_PDF_FULL_PAGE_SOURCE,
            "scan_pdf_full_page", rid8,
        )
        if _is_scan:
            pipeline_value = "vlm" if _scan_full else "standard"
        elif _vpt > 0 and _page_count <= _vpt:
            pipeline_value = "vlm"
        else:
            pipeline_value = "standard"
        logger.info(
            f"[rid={rid8}] archive member {fname}: "
            f"{'SCAN' if _is_scan else 'TEXT'} {_page_count}p -> {pipeline_value}"
        )
    else:
        # DOCX/PPTX с OLE-объектами -> Gotenberg -> PDF -> vlm.
        if has_ole_objects(fbytes, fname):
            try:
                pdf_bytes = await convert_via_gotenberg(client, fbytes, fname)
                fname = fname.rsplit(".", 1)[0] + ".pdf"
                fbytes = pdf_bytes
                pipeline_value = "vlm"
                try:
                    _pdf = fitz.open(stream=fbytes, filetype="pdf")
                    _pages = len(_pdf) or None
                    _pdf.close()
                except Exception:
                    pass
                logger.info(f"[rid={rid8}] archive member {fname}: OLE -> Gotenberg -> vlm")
            except Exception as e:
                pipeline_value = "standard"
                logger.error(f"[rid={rid8}] archive member {fname}: Gotenberg failed: {e}")

    # Форвард в docling-serve с тем же retry/null-handling, что у обычной загрузки.
    data_m = _inject_pipeline_params(list(base_data), pipeline_value, vlm_overrides, do_pic_desc)
    ftype = mimetypes.guess_type(fname)[0] or "application/octet-stream"
    files_m = [("files", (fname, fbytes, ftype))]
    multipart = [(k, (None, v)) for k, v in data_m]
    multipart.extend(files_m)

    resp = await run_docling_request(
        client=client, request=request, target_url=target_url,
        multipart=multipart, data=data_m, files=files_m,
        _request_id=f"{request_id}-m{idx}", _rid8=f"{rid8}.{idx}",
        _t_total=time.time(), max_docs=max_docs,
    )
    md = _md_from_response(resp)
    if resp.status_code == 200 and md is not None:
        return _r("ok", md=md)
    return _r("error", note=f"docling вернул статус {resp.status_code}.")


_STATUS_ICON = {"ok": "✅", "unsupported": "⏭️", "error": "⚠️"}


def _human_size(n: int) -> str:
    """Человекочитаемый размер: 8 КБ, 1.4 МБ и т.п."""
    if not n:
        return ""
    units = ["Б", "КБ", "МБ", "ГБ", "ТБ"]
    s = float(n)
    i = 0
    while s >= 1024 and i < len(units) - 1:
        s /= 1024
        i += 1
    if i == 0:
        return f"{int(s)} {units[i]}"
    return f"{s:.1f}".rstrip("0").rstrip(".") + f" {units[i]}"


def _meta_str(r: dict) -> str:
    """Строка метаданных файла: страницы + размер (+ причина для skip/error)."""
    bits = []
    if r.get("pages"):
        bits.append(f"{r['pages']} стр.")
    size = _human_size(r.get("size") or 0)
    if size:
        bits.append(size)
    if r["status"] == "unsupported":
        bits.append("формат не поддерживается")
    elif r["status"] == "error":
        bits.append(r.get("note") or "ошибка обработки")
    return " · ".join(bits)


def _build_tree(results: list) -> dict:
    """Собрать дерево из плоских путей вида archive.zip/dir/file.pdf.

    Промежуточные узлы (архивы/папки) — dict, листья — ("leaf", result).
    """
    tree: dict = {}
    for r in results:
        parts = r["filename"].split("/")
        node = tree
        for p in parts[:-1]:
            sub = node.get(p)
            if not isinstance(sub, dict):
                sub = {}
                node[p] = sub
            node = sub
        node[parts[-1]] = ("leaf", r)
    return tree


def _render_tree(tree: dict, depth: int, lines: list):
    for name, val in tree.items():
        indent = "  " * depth
        if isinstance(val, tuple):  # лист — файл
            r = val[1]
            icon = _STATUS_ICON.get(r["status"], "•")
            meta = _meta_str(r)
            lines.append(f"{indent}- {icon} {name}" + (f" — {meta}" if meta else ""))
        else:  # ветка — архив или папка
            node_icon = "📦" if archive_kind(name) else "📁"
            lines.append(f"{indent}- {node_icon} {name}")
            _render_tree(val, depth + 1, lines)


def _merge_markdown(top_name: str, results: list, notes: list) -> str:
    """Склеить markdown всех файлов: сводный манифест сверху + документы ниже.

    Манифест (дерево состава, статусы, страницы/объёмы, нераспакованное) даёт
    модели/человеку и человеку карту архива; под каждым документом — мета-строка
    с источником для провенанса в векторном хранилище.
    """
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "unsupported")
    errored = sum(1 for r in results if r["status"] == "error")

    parts = [f"# Содержимое архива: {top_name}", ""]
    parts.append(
        f"**Итог:** обработано {ok}, пропущено {skipped}, с ошибкой {errored}; "
        f"всего {len(results)} файл(ов)."
    )

    # Сводный манифест — дерево состава архива.
    parts.extend(["", "## Состав архива", ""])
    if results:
        tree_lines: list = []
        _render_tree(_build_tree(results), 0, tree_lines)
        parts.extend(tree_lines)
    else:
        parts.append("_(нет файлов для обработки)_")

    # Нераспакованное (битые/недоступные субархивы, сработавшие лимиты).
    if notes:
        parts.extend(["", "**Не удалось распаковать:**", ""])
        parts.extend(f"- {n}" for n in notes)

    # Сами документы с разделителями и мета-строкой-источником.
    for r in results:
        meta = _meta_str(r)
        parts.extend(["", "---", "", f"## 📄 {r['filename']}", ""])
        if meta:
            parts.extend([f"_Источник: {r['filename']} · {meta}_", ""])
        if r["status"] == "ok" and r["md"]:
            parts.append(r["md"])
        elif r["status"] == "ok":
            parts.append("_(пустой результат)_")
        elif r["status"] == "unsupported":
            parts.append("_⚠️ Формат файла не поддерживается — пропущен._")
        else:
            parts.append(f"_⚠️ {r.get('note') or 'не удалось обработать файл.'}_")

    return "\n".join(parts)


async def handle_archive(
    *, client, request: Request, target_url: str,
    files: list, data: list,
    vlm_overrides: dict, routing_overrides: dict,
    rid8: str, request_id: str, t_total: float,
) -> Response:
    """Развернуть архив(ы) среди загруженных файлов и обработать содержимое."""
    _t = time.time()

    # Снимаем управляющие поля — пересоберём их индивидуально под каждый файл.
    base_data = [(k, v) for k, v in data if k not in _CONTROL_KEYS]
    do_pic_desc = ""
    for k, v in data:
        if k == "do_picture_description":
            do_pic_desc = str(v).lower()
            break

    max_docs = int(vlm_overrides.get(
        "vlm_max_concurrent_docs", DEFAULT_VLM_MAX_CONCURRENT_DOCS
    ))

    # Имя архива(ов) — для заголовка ответа.
    archive_names = [f[1][0] for f in files if is_archive(f[1][0], f[1][1])]
    top_name = archive_names[0] if archive_names else (files[0][1][0] if files else "archive")

    # Разворачиваем верхнеуровневые файлы: архивы — рекурсивно, остальное — как есть.
    leaves: list = []
    notes: list = []
    for _, (fname, fbytes, _ftype) in files:
        if is_archive(fname, fbytes):
            sub_leaves, sub_notes = extract_archive(fname, fbytes)
            leaves.extend(sub_leaves)
            notes.extend(sub_notes)
        else:
            leaves.append((fname, fbytes))

    logger.info(
        f"[rid={rid8}] ARCHIVE {top_name}: extracted {len(leaves)} file(s), "
        f"{len(notes)} note(s)"
    )
    _stats_set(request, doc_type="ARCHIVE", pipeline="archive",
               filename=top_name)

    if not leaves:
        notes.append("Архив не содержит файлов для обработки.")

    results: list = []
    for idx, (fname, fbytes) in enumerate(leaves):
        try:
            res = await _convert_member(
                client=client, request=request, target_url=target_url,
                fname=fname, fbytes=fbytes,
                base_data=base_data, do_pic_desc=do_pic_desc,
                vlm_overrides=vlm_overrides, routing_overrides=routing_overrides,
                rid8=rid8, request_id=request_id, idx=idx, max_docs=max_docs,
            )
        except Exception as e:
            logger.error(f"[rid={rid8}] archive member {fname} crashed: {e}")
            res = {"filename": fname, "status": "error", "md": None,
                   "note": f"внутренняя ошибка обработки: {e}"}
        results.append(res)

    merged = _merge_markdown(top_name, results, notes)
    _ok = sum(1 for r in results if r["status"] == "ok")
    _skipped = sum(1 for r in results if r["status"] == "unsupported")
    _elapsed = time.time() - _t
    _total_ms = (time.time() - t_total) * 1000
    logger.info(
        f"[rid={rid8}] ARCHIVE done: {_ok} ok, {_skipped} skipped, "
        f"{len(results)} total, {_total_ms:.0f}ms"
    )

    response = {
        "document": {"filename": top_name, "md_content": merged},
        "status": "success",
        "errors": [],
        "processing_time": _elapsed,
        "proxy_diagnostics": {
            "archive": True,
            "request_id": request_id,
            "files_total": len(results),
            "files_ok": _ok,
            "files_skipped": _skipped,
            "notes": notes,
        },
    }
    return Response(
        content=json.dumps(response, ensure_ascii=False).encode("utf-8"),
        status_code=200,
        headers={"content-type": "application/json"},
    )
