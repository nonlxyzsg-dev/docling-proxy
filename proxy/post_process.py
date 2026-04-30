"""Markdown post-processing: VLM truncation fixup, KaTeX compatibility."""
import json, re, logging

logger = logging.getLogger("docling_proxy")


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
    """Пост-обработка md_content докла."""
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

    strict_matches = list(_END_MARKER_STRICT_RE.finditer(md))
    if strict_matches:
        info["had_end_marker"] = True
        info["markers_found_count"] = len(strict_matches)
        tail_cut = md[: strict_matches[-1].end()]
        new_md = _END_MARKER_STRICT_RE.sub("", tail_cut).rstrip()
        info["markers_removed_count"] = len(strict_matches)
        info["action"] = "end_marker_strict"
        info["trimmed"] = True
        changed = True
    else:
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
