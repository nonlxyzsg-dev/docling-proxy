"""Pipelines: XLS/DOC conversion, OCR SDK enrichment, supported extensions."""
import os, json, time, base64, asyncio, fitz, re, uuid, httpx, logging
from datetime import datetime as dt
from proxy.config import (
    LOG_DIR, GOTENBERG_URL,
    OCR_SDK_URL, OCR_SDK_INBOX_CONTAINER, OCR_SDK_ENABLED, OCR_SDK_TIMEOUT,
    DEFAULT_VLM_URL, DEFAULT_VLM_API_KEY, DEFAULT_VLM_MODEL,
    DEFAULT_VLM_TIMEOUT, DEFAULT_VLM_CONCURRENCY,
    ENRICH_PICTURES_WITH_122B, ENRICH_LABELS,
)
from proxy.prompts import ENRICH_VLM_PROMPT

logger = logging.getLogger("docling_proxy")


def save(data: list, files: list):
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    file_names = [f[1][0] for f in files] if files else []
    params = {"data": {k: v for k, v in data}, "files": file_names}
    filename = f"params_{dt.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    with open(os.path.join(LOG_DIR, filename), 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm",
    ".md", ".csv", ".png", ".jpg", ".jpeg", ".tiff", ".tif",
    ".bmp", ".gif", ".webp", ".asciidoc", ".adoc",
    ".xls",
    ".doc",
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
    """Convert binary .doc to markdown via Gotenberg (doc→PDF) + PyMuPDF (PDF→text)."""
    try:
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

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        all_md = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                all_md.append(text)
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


def calculate_enrich_max_tokens(label: str, bbox: list) -> int:
    """Динамический max_tokens для обогащения регионов OCR SDK."""
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

    if norm_area < 10_000:
        area_mult = 0.4
    elif norm_area < 100_000:
        area_mult = 0.7
    else:
        area_mult = 1.0

    aspect = max(w, h) / max(min(w, h), 1)
    shape_mult = 0.5 if aspect > 5 else 1.0

    result = int(base * area_mult * shape_mult)
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
    """Process SCAN PDF via OCR SDK + optional enrichment via 122B."""
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
