import time
import fitz  # PyMuPDF
from fastapi import FastAPI, Request, Response
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from datetime import datetime as dt
import os, json, httpx, asyncio, zipfile
import xml.etree.ElementTree as ET

load_dotenv()

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
DEFAULT_VLM_MAX_CONCURRENT_DOCS = os.getenv("DEFAULT_VLM_MAX_CONCURRENT_DOCS")
DEFAULT_VLM_MAX_COMPLETION_TOKENS = int(os.getenv("DEFAULT_VLM_MAX_COMPLETION_TOKENS", "2048"))

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

DEFAULT_VLM_PIPELINE_PROMPT = (
    "Обработай эту страницу документа и верни результат в формате markdown.\n\n"
    "ПРАВИЛА ИЗВЛЕЧЕНИЯ ТЕКСТА:\n"
    "- Извлеки весь текст точно на языке оригинала\n"
    "- Сохрани структуру: заголовки, списки, таблицы, форматирование\n"
    "- Не переводи и не перефразируй текст\n\n"
    "ПРАВИЛА ОПИСАНИЯ ВИЗУАЛЬНОГО СОДЕРЖИМОГО:\n"
    "Если на странице есть изображения, фотографии, рисунки, диаграммы, графики, схемы или иллюстрации — "
    "опиши каждый элемент в квадратных скобках, адаптируя уровень детализации:\n\n"
    "• ИНЖЕНЕРНЫЕ ЧЕРТЕЖИ и ТЕХНИЧЕСКИЕ СХЕМЫ: максимально подробно — размеры, допуски, масштаб, "
    "материалы, обозначения, содержимое штампа (организация, номер чертежа, дата, подписи), "
    "спецификация, позиции деталей, сечения, виды, проекции.\n"
    "• ГРАФИКИ и ДИАГРАММЫ: оси, единицы измерения, значения ключевых точек, легенда, тренды, "
    "подписи данных, заголовок графика.\n"
    "• ЭЛЕКТРИЧЕСКИЕ/ГИДРАВЛИЧЕСКИЕ/ПНЕВМАТИЧЕСКИЕ СХЕМЫ: компоненты, соединения, номиналы, "
    "обозначения по ГОСТ/ISO, направления потоков.\n"
    "• ФОТОГРАФИИ и ИЛЛЮСТРАЦИИ: что изображено, ключевые объекты, их расположение, цвета, "
    "текст на изображении, контекст.\n"
    "• ЛОГОТИПЫ и ШТАМПЫ: краткое описание в одну строку.\n\n"
    "Отвечай только markdown, без вступлений и пояснений."
)


# ═══════════════════════════════════════════════════════════════
# httpx AsyncClient
# ═══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(1200.0, connect=10.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        follow_redirects=False,
    )
    yield
    await app.state.client.aclose()

app = FastAPI(lifespan=lifespan)


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
                        print(f"Confluence decode: found HTML part ({len(payload)} bytes)")
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
                print(f"Confluence decode: fallback quopri ({len(decoded)} bytes)")
                return decoded, html_name
        
        print(f"Confluence decode: could not extract HTML from {filename}")
        return None, None
    except Exception as e:
        print(f"Confluence decode ERROR: {e}")
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
        # Скан: каждая страница → VLM запрос, параллельно по vlm_concurrency
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



def build_picture_description_api(vlm_overrides: dict) -> str:
    params = {"model": vlm_overrides.get("vlm_model", DEFAULT_VLM_MODEL), "chat_template_kwargs": {"enable_thinking": False}}
    if "vlm_temperature" in vlm_overrides:
        params["temperature"] = float(vlm_overrides["vlm_temperature"])
    if "vlm_max_tokens" in vlm_overrides:
        params["max_tokens"] = int(vlm_overrides["vlm_max_tokens"])
    api_config = {
        "url": vlm_overrides.get("vlm_url", DEFAULT_VLM_URL),
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
        "scale": 2.0,
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
    """VlmModelApi flat format for vlm_pipeline_model_api."""
    config = {
        "url": vlm_overrides.get("vlm_url", DEFAULT_VLM_URL),
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
        "scale": 2.0,
        "temperature": 0.0
    }
    return json.dumps(config)



# ═══════════════════════════════════════════════════════════════
# Пост-обработка LaTeX для KaTeX-совместимости
# ═══════════════════════════════════════════════════════════════

import re

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


        
        if len(md) != original_len:
            print(f"KaTeX fix: {original_len} -> {len(md)} chars")
        
        doc["md_content"] = md
        data["document"] = doc
        return json.dumps(data, ensure_ascii=False).encode()
    except Exception as e:
        print(f"KaTeX fix error: {e}")
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
        print(f"XLS conversion error: {e}")
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
            print(f"DOC→PDF Gotenberg failed: HTTP {resp.status_code}")
            return None
        pdf_bytes = resp.content
        _gotenberg_ms = (time.time() - _t) * 1000
        print(f"TIMING doc→pdf (Gotenberg): {_gotenberg_ms:.0f}ms ({len(pdf_bytes)} bytes)")
        
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
        print(f"TIMING doc→markdown total: {_total_ms:.0f}ms ({len(md_content)} chars)")
        
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
        print(f"DOC conversion error: {e}")
        return None



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

        form = await request.form()

        do_pic_desc = form.get("do_picture_description", "").lower()
        do_pic_custom = form.get("do_picture_description_custom", "").lower()
        do_classification = form.get("do_picture_classification", "").lower()

        print(f"РЕЖИМ do_pic_desc: {do_pic_desc}")
        print(f"РЕЖИМ picture_description_custom: {do_pic_custom} и classification: {do_classification}")

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
            
            # Неподдерживаемый формат → дружелюбное сообщение
            if ext and ext not in SUPPORTED_EXTENSIONS:
                print(f"UNSUPPORTED FORMAT: {fname} ({ext})")
                _total_ms = (time.time() - _t_total) * 1000
                print(f"TIMING total: {_total_ms:.0f}ms  status: unsupported_format")
                resp_headers = {"content-type": "application/json"}
                return Response(
                    content=get_unsupported_response(fname),
                    status_code=422,
                    headers=resp_headers,
                )
            
            # .xls → конвертируем через xlrd в markdown и возвращаем сразу
            if ext == ".xls":
                print(f"XLS detected: {fname} -> converting via xlrd/pandas")
                _t_xls = time.time()
                xls_result = convert_xls_to_markdown(fbytes, fname)
                _xls_ms = (time.time() - _t_xls) * 1000
                if xls_result:
                    print(f"TIMING xls_convert: {_xls_ms:.0f}ms")
                    _total_ms = (time.time() - _t_total) * 1000
                    print(f"TIMING total: {_total_ms:.0f}ms  status: 200 (xls)")
                    resp_headers = {"content-type": "application/json"}
                    return Response(content=xls_result, status_code=200, headers=resp_headers)
                else:
                    print(f"XLS conversion failed, passing to docling")
            
            # .doc → Confluence MIME HTML или бинарный .doc
            if ext == ".doc":
                if is_confluence_doc(fbytes):
                    # Confluence export: MIME-encoded HTML → декодируем → подменяем на .html
                    print(f"Confluence .doc detected: {fname}")
                    html_bytes, html_name = decode_confluence_doc(fbytes, fname)
                    if html_bytes:
                        files[fi] = ("files", (html_name, html_bytes, "text/html"))
                        print(f"Confluence decode OK: {fname} -> {html_name} ({len(html_bytes)} bytes)")
                    else:
                        print(f"Confluence decode FAILED: {fname} -> returning error")
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
                    print(f"Binary .doc detected: {fname} -> converting via Gotenberg+PyMuPDF")
                    _t_doc = time.time()
                    doc_result = await convert_doc_to_markdown(client, fbytes, fname)
                    _doc_ms = (time.time() - _t_doc) * 1000
                    if doc_result:
                        print(f"TIMING doc_convert: {_doc_ms:.0f}ms")
                        _total_ms = (time.time() - _t_total) * 1000
                        print(f"TIMING total: {_total_ms:.0f}ms  status: 200 (doc)")
                        return Response(content=doc_result, status_code=200, headers={"content-type": "application/json"})
                    else:
                        print(f"DOC conversion failed, passing to docling")

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
                print(f"TIMING auto-detect: {_detect_ms:.0f}ms")
                # Подсчёт страниц для маршрутизации
                _page_count = 0
                try:
                    import fitz
                    _pdf_doc = fitz.open(stream=fbytes, filetype="pdf")
                    _page_count = len(_pdf_doc)
                    _pdf_doc.close()
                except Exception as e:
                    print(f"WARNING: could not count pages: {e}")
                # Подсчёт картинок для предупреждения
                _image_count = count_pdf_images(fbytes) if not _is_scan else 0
                if _image_count > 0:
                    print(f"PDF images: {_image_count} images in {_page_count} pages")
                
                pdf_type = "SCAN" if _is_scan else "TEXT PDF"
                _base_fname = fname.rsplit("/", 1)[-1] if "/" in fname else fname
                # Убираем UUID-префикс из имени файла для пользователя
                if "_" in _base_fname and len(_base_fname.split("_")[0]) == 36:
                    _base_fname = _base_fname.split("_", 1)[1]
                _processing_warning = get_processing_warning(_base_fname, _page_count, _image_count, _is_scan)
                if _processing_warning:
                    print(f"WARNING for user: {_processing_warning}")
                
                # Маршрутизация: SCAN → всегда VLM, TEXT PDF > N стр. → standard
                VLM_PAGE_LIMIT = 20
                if _is_scan:
                    pipeline_value = "vlm"
                    print(f"Auto-detect: {fname} -> {pdf_type} ({_page_count} pages) -> pipeline=vlm (scans always use VLM)")
                elif _page_count > VLM_PAGE_LIMIT:
                    pipeline_value = "standard"
                    print(f"Auto-detect: {fname} -> {pdf_type} ({_page_count} pages) -> pipeline=standard (>{VLM_PAGE_LIMIT} pages, text extractable)")
                else:
                    pipeline_value = "vlm"
                    print(f"Auto-detect: {fname} -> {pdf_type} ({_page_count} pages) -> pipeline=vlm")
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
                    print(f"Auto-detect: {ole_fname} -> has OLE objects -> converting via Gotenberg")
                    try:
                        _t_gotenberg = time.time()
                        pdf_bytes = await convert_via_gotenberg(client, ole_bytes, ole_fname)
                        _gotenberg_ms = (time.time() - _t_gotenberg) * 1000
                        print(f"TIMING gotenberg: {_gotenberg_ms:.0f}ms ({len(pdf_bytes)} bytes PDF)")
                        # Подменяем файл на сконвертированный PDF
                        pdf_name = ole_fname.rsplit(".", 1)[0] + ".pdf"
                        files[_ole_file_idx] = ("files", (pdf_name, pdf_bytes, "application/pdf"))
                        pipeline_value = "vlm"
                        print(f"Auto-detect: {ole_fname} -> OLE -> Gotenberg -> {pdf_name} -> pipeline=vlm")
                    except Exception as e:
                        print(f"Gotenberg ERROR: {e} -> fallback to standard pipeline")
                        pipeline_value = "standard"
                else:
                    pipeline_value = "standard"
                    print(f"Auto-detect: non-PDF {file_names} -> no OLE -> pipeline=standard")

        # ── Обновляем pipeline в data для docling ──
        data = [(k, v) for k, v in data if k != "pipeline"]
        data.append(("pipeline", pipeline_value))

        # ── VLM Pipeline: страница целиком -> Qwen3-VL -> markdown ──
        if pipeline_value == "vlm":
            
            keys_data = [k for k, _ in data]
            
            if "vlm_pipeline_model_api" not in keys_data:
                data.append(("vlm_pipeline_model_api", build_vlm_pipeline_model_api(vlm_overrides)))
                print("VLM Pipeline: injected vlm_pipeline_model_api (Qwen3-VL full-page OCR)")
                
            # Workaround: VLM pipeline + embedded images = Pillow crash on some PDFs
            if "image_export_mode" not in keys_data:
                data.append(("image_export_mode", "placeholder"))
                print("VLM Pipeline: выбран image_export_mode=placeholder")
                
            # VLM уже извлекает всё — picture description избыточен
            do_pic_desc = "false"
            
            data = [(k, v) for k, v in data if k not in ("do_picture_description", "do_picture_description_custom")]
            data.append(("do_picture_description", "false"))
            data.append(("do_picture_description_custom", "false"))
            print("VLM Pipeline: suppressed picture_description (redundant with VLM)")

        # ── Picture Description: описание картинок через VLM ──
        if do_pic_desc == "true":
            keys_data = [k for k, _ in data]
            if do_pic_custom == "true":
                if "picture_description_custom_config" not in keys_data:
                    data.append(("picture_description_custom_config", build_custom_model(vlm_overrides, classification=do_classification)))
                    print("Режим: picture_description_custom_config")
            else:
                if "picture_description_api" not in keys_data:
                    api_json = build_picture_description_api(vlm_overrides)
                    data.append(("picture_description_api", api_json))
                    print("Режим: picture_description_api")

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

        _t_queue = time.time()
        async with sem:
            _queue_ms = (time.time() - _t_queue) * 1000
            _t_docling = time.time()
            resp = await client.post(target_url, files=multipart, timeout=1200.0)
            _docling_ms = (time.time() - _t_docling) * 1000
            print(f"TIMING queue_wait: {_queue_ms:.0f}ms  docling_request: {_docling_ms:.0f}ms")

        _total_ms = (time.time() - _t_total) * 1000
        print(f"TIMING total: {_total_ms:.0f}ms  status: {resp.status_code}")
        
        # Инъекция предупреждения о большом документе
        _resp_content = resp.content
        if resp.status_code == 200 and '_processing_warning' in dir() and _processing_warning:
            try:
                _resp_data = json.loads(_resp_content)
                _doc = _resp_data.get("document", {})
                _md = _doc.get("md_content", "")
                if _md:
                    _warning_block = f"> ⚠️ {_processing_warning}\n\n"
                    _doc["md_content"] = _warning_block + _md
                    _resp_data["document"] = _doc
                    _resp_content = json.dumps(_resp_data, ensure_ascii=False).encode()
                    print(f"Injected processing warning into response")
            except Exception as e:
                print(f"Warning injection error: {e}")
        
        # Пост-обработка: KaTeX-совместимость
        fixed_content = fix_katex_compatibility(_resp_content) if resp.status_code == 200 else _resp_content
        
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