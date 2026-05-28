"""Распаковка архивов: детект + рекурсивное извлечение (архив внутри архива).

Чистый helper без сетевых вызовов и без зависимостей от FastAPI — только
разбор байтов архива в плоский список листовых файлов. Орхестрация обработки
извлечённых документов — в proxy/archive.py.

Форматы:
- zip, tar/tar.gz/tgz/tar.bz2/tar.xz, а также одиночные gz/bz2/xz — stdlib.
- 7z — через py7zr (ленивый импорт).
- rar — через rarfile (ленивый импорт) + системный бинарник unrar/bsdtar.

Если библиотека/бинарник недоступны или архив битый — соответствующий узел
помечается note'ом и пропускается, исключение наружу не пробрасывается.
"""
import io
import os
import bz2
import gzip
import lzma
import tarfile
import zipfile
import logging

from proxy.config import (
    ARCHIVE_MAX_DEPTH, ARCHIVE_MAX_FILES, ARCHIVE_MAX_TOTAL_BYTES,
    ARCHIVE_MAX_TOTAL_MB,
)

logger = logging.getLogger("docling_proxy")


def archive_kind(filename: str) -> str | None:
    """Определить тип архива по расширению. None — не архив (обычный файл)."""
    n = (filename or "").lower()
    if n.endswith(".zip"):
        return "zip"
    if n.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2",
                   ".tar.xz", ".txz")):
        return "tar"
    if n.endswith(".7z"):
        return "7z"
    if n.endswith(".rar"):
        return "rar"
    # Одиночные сжатые файлы (file.pdf.gz и т.п.) — НЕ .tar.* (их поймали выше).
    if n.endswith((".gz", ".bz2", ".xz")):
        return "single"
    return None


def _sniff_kind(fbytes: bytes) -> str | None:
    """Детект по magic-байтам — fallback для файлов без расширения внутри архива.

    ВАЖНО: zip-сигнатуру (PK..) специально НЕ матчим, т.к. её имеют docx/xlsx/
    pptx. На верхнем уровне расширения от OWUI корректны, а вложенные офисные
    файлы должны идти в обычный пайплайн, а не распаковываться как zip.
    """
    if fbytes[:6] == b"Rar!\x1a\x07" or fbytes[:7] == b"Rar!\x1a\x07\x01":
        return "rar"
    if fbytes[:6] == b"7z\xbc\xaf\x27\x1c":
        return "7z"
    return None


def is_archive(filename: str, fbytes: bytes) -> bool:
    """Является ли файл архивом, который умеем разворачивать."""
    return archive_kind(filename) is not None


def _is_junk(name: str) -> bool:
    """Служебный мусор архиваторов, который не нужно обрабатывать."""
    base = name.rsplit("/", 1)[-1]
    if not base:
        return True
    if name.startswith("__MACOSX/") or "/__MACOSX/" in name:
        return True
    if base in (".DS_Store", "Thumbs.db", "desktop.ini"):
        return True
    if base.startswith("._"):  # AppleDouble
        return True
    return False


def _strip_compress_ext(name: str) -> str:
    """Имя после снятия одного слоя сжатия: report.pdf.gz -> report.pdf."""
    low = name.lower()
    for ext in (".gz", ".bz2", ".xz"):
        if low.endswith(ext):
            return name[: -len(ext)]
    return name


def _extract_zip(fbytes: bytes) -> list[tuple[str, bytes]]:
    out = []
    with zipfile.ZipFile(io.BytesIO(fbytes)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            if _is_junk(info.filename):
                continue
            out.append((info.filename, z.read(info)))
    return out


def _extract_tar(fbytes: bytes) -> list[tuple[str, bytes]]:
    out = []
    # mode "r:*" — автодетект сжатия (gz/bz2/xz/несжатый).
    with tarfile.open(fileobj=io.BytesIO(fbytes), mode="r:*") as t:
        for m in t.getmembers():
            if not m.isfile():
                continue
            if _is_junk(m.name):
                continue
            f = t.extractfile(m)
            if f is None:
                continue
            out.append((m.name, f.read()))
    return out


def _extract_single(name: str, fbytes: bytes) -> list[tuple[str, bytes]]:
    """Одиночный сжатый файл (gz/bz2/xz) -> [(внутреннее_имя, байты)]."""
    low = name.lower()
    if low.endswith(".gz"):
        data = gzip.decompress(fbytes)
    elif low.endswith(".bz2"):
        data = bz2.decompress(fbytes)
    elif low.endswith(".xz"):
        data = lzma.decompress(fbytes)
    else:
        return []
    return [(_strip_compress_ext(name), data)]


def _extract_7z(fbytes: bytes) -> list[tuple[str, bytes]]:
    import py7zr  # ленивый импорт: либа может отсутствовать в окружении
    out = []
    with py7zr.SevenZipFile(io.BytesIO(fbytes), mode="r") as z:
        for fname, bio in z.readall().items():
            if _is_junk(fname):
                continue
            out.append((fname, bio.read()))
    return out


def _extract_rar(fbytes: bytes) -> list[tuple[str, bytes]]:
    import rarfile  # ленивый импорт; требует системный unrar/bsdtar
    out = []
    with rarfile.RarFile(io.BytesIO(fbytes)) as r:
        for info in r.infolist():
            if info.isdir():
                continue
            if _is_junk(info.filename):
                continue
            out.append((info.filename, r.read(info)))
    return out


def _extract_by_kind(kind: str, name: str, fbytes: bytes) -> list[tuple[str, bytes]]:
    if kind == "zip":
        return _extract_zip(fbytes)
    if kind == "tar":
        return _extract_tar(fbytes)
    if kind == "single":
        return _extract_single(name, fbytes)
    if kind == "7z":
        return _extract_7z(fbytes)
    if kind == "rar":
        return _extract_rar(fbytes)
    return []


def extract_archive(top_name: str, top_bytes: bytes) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Рекурсивно развернуть архив в плоский список листовых (не-архивных) файлов.

    Возвращает (leaves, notes):
    - leaves: list[(display_path, bytes)] — извлечённые обычные файлы. display_path
      включает путь внутри архива (с вложенностью) для заголовков-разделителей.
    - notes: list[str] — предупреждения (пропущенные форматы, ошибки распаковки,
      сработавшие лимиты). Идут пользователю в раздел «Примечания».

    Лимиты (защита от zip-бомб): ARCHIVE_MAX_DEPTH / MAX_FILES / MAX_TOTAL_BYTES.
    """
    leaves: list[tuple[str, bytes]] = []
    notes: list[str] = []
    state = {"files": 0, "bytes": 0, "stopped": False}

    def _stop_limits() -> bool:
        return state["stopped"]

    def _walk(name: str, data: bytes, depth: int, prefix: str):
        if state["stopped"]:
            return

        kind = archive_kind(name)
        if kind is None:
            kind = _sniff_kind(data)

        # Лист — обычный файл.
        if kind is None:
            if state["files"] >= ARCHIVE_MAX_FILES:
                if not state["stopped"]:
                    notes.append(
                        f"⚠️ Достигнут лимит в {ARCHIVE_MAX_FILES} файлов — "
                        f"остальное содержимое архива пропущено."
                    )
                    state["stopped"] = True
                return
            if state["bytes"] + len(data) > ARCHIVE_MAX_TOTAL_BYTES:
                if not state["stopped"]:
                    notes.append(
                        f"⚠️ Достигнут лимит распакованного объёма "
                        f"({ARCHIVE_MAX_TOTAL_MB} МБ) — остальное пропущено."
                    )
                    state["stopped"] = True
                return
            state["files"] += 1
            state["bytes"] += len(data)
            leaves.append((prefix + name, data))
            return

        # Вложенный архив.
        if depth >= ARCHIVE_MAX_DEPTH:
            notes.append(
                f"⚠️ Архив «{prefix + name}» пропущен: превышена глубина "
                f"вложенности ({ARCHIVE_MAX_DEPTH})."
            )
            return

        try:
            members = _extract_by_kind(kind, name, data)
        except ImportError:
            lib = "py7zr" if kind == "7z" else "rarfile/unrar"
            notes.append(
                f"⚠️ Архив «{prefix + name}» ({kind}) пропущен: на сервере не "
                f"установлен {lib}."
            )
            logger.warning(f"archive: {kind} support unavailable for {name}")
            return
        except Exception as e:
            notes.append(f"⚠️ Не удалось распаковать «{prefix + name}»: {e}")
            logger.warning(f"archive: extract failed for {name}: {e}")
            return

        # Для одиночных сжатых файлов (gz/bz2/xz) имя внутреннего файла уже
        # содержит полный путь — не добавляем имя контейнера в префикс.
        child_prefix = prefix if kind == "single" else prefix + name + "/"
        for mname, mdata in members:
            if _stop_limits():
                return
            _walk(mname, mdata, depth + 1, child_prefix)

    _walk(top_name, top_bytes, depth=0, prefix="")
    return leaves, notes
