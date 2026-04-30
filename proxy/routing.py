"""Routing helpers: PDF type detection, Confluence/OLE handling, processing warnings."""
import os, math, fitz, zipfile, logging, httpx
from proxy.config import GOTENBERG_URL, OCR_SDK_ENABLED

logger = logging.getLogger("docling_proxy")

def is_scan_pdf(pdf_bytes: bytes, min_chars_per_page: int = 100, pages_to_check: int = 3) -> bool:
    """Check if PDF is a scan (no/little extractable text)."""
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


def is_confluence_doc(file_bytes: bytes) -> bool:
    """Check if .doc file is actually a Confluence MIME HTML export."""
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
    """Decode Confluence MIME HTML .doc to plain HTML."""
    import email
    import quopri

    try:
        msg = email.message_from_bytes(file_bytes)

        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                if ct == 'text/html':
                    payload = part.get_payload(decode=True)
                    if payload:
                        html_name = filename.rsplit('.', 1)[0] + '.html'
                        logger.info(f"Confluence decode: found HTML part ({len(payload)} bytes)")
                        return payload, html_name

        payload = msg.get_payload(decode=True)
        if payload:
            html_name = filename.rsplit('.', 1)[0] + '.html'
            return payload, html_name

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
    parts = []
    if page_count > 20:
        parts.append(f"{page_count} страниц")
    if image_count > 10:
        parts.append(f"{image_count} изображений")
    if is_scan:
        parts.append("отсканированный документ")

    if not parts:
        return ""

    est_seconds = 0
    if is_scan:
        if OCR_SDK_ENABLED:
            est_seconds = page_count * 0.5 + 10
        else:
            batches = math.ceil(page_count / vlm_concurrency)
            est_seconds = batches * 20
    else:
        est_seconds = page_count * 0.2
        if image_count > 0:
            img_batches = math.ceil(image_count / vlm_concurrency)
            est_seconds += img_batches * 20

    detail = ", ".join(parts)

    if est_seconds >= 60:
        est_min = math.ceil(est_seconds / 60)
        time_str = f"~{est_min} мин"
    elif est_seconds >= 10:
        time_str = f"~{int(est_seconds)} сек"
    else:
        return ""

    return (
        f"Документ «{filename}» содержит {detail}. "
        f"Ориентировочное время обработки: {time_str}."
    )
