"""Adaptive image resize for VLM endpoint payload (before upstream forward).

Pure helper. Decodes a data:URL with a base64-encoded image, downsamples it
to roughly target_pixels (preserving aspect ratio) when above threshold, and
re-encodes to PNG. Images already at or below target_pixels are returned
unchanged. Errors are non-fatal: the original data:URL is returned with
reason='error' so the caller can keep forwarding.
"""
import base64
import logging
import math
from io import BytesIO
from typing import Tuple

from PIL import Image

logger = logging.getLogger("docling_proxy")

_SUPPORTED_MIMES = ("image/png", "image/jpeg", "image/jpg")


def resize_image_in_data_url(
    data_url: str,
    target_pixels: int,
    min_pixels: int = 200_704,
) -> Tuple[str, dict]:
    """Resize a base64 image inside a data:URL to ~target_pixels area.

    Returns (new_data_url, stats). Stats is always populated; reason is one of
    'in_range', 'too_small', 'too_large', 'unsupported_mime', 'error', 'disabled'.

    Behaviour:
      - target_pixels <= 0 -> 'disabled', no decode, returned as-is.
      - was_px <= target_pixels (including was_px < min_pixels) -> returned as-is
        with reason 'too_small' or 'in_range'.
      - was_px > target_pixels -> LANCZOS-downscaled to keep aspect ratio so that
        new_w * new_h ~ target_pixels, re-encoded to PNG.
      - any decode/encode exception -> returned as-is with reason 'error'.
    """
    stats = {
        "was_w": None,
        "was_h": None,
        "was_px": None,
        "new_w": None,
        "new_h": None,
        "new_px": None,
        "resized": False,
        "reason": "",
    }

    if target_pixels is None or target_pixels <= 0:
        stats["reason"] = "disabled"
        return data_url, stats

    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        stats["reason"] = "error"
        return data_url, stats

    # Parse data:<mime>;base64,<payload>
    try:
        head, payload_b64 = data_url.split(",", 1)
        # head looks like "data:image/png;base64"
        mime_part = head[len("data:"):]
        if ";base64" not in mime_part:
            stats["reason"] = "error"
            return data_url, stats
        mime = mime_part.split(";", 1)[0].strip().lower()
    except Exception as e:
        logger.warning(f"image_resize: cannot parse data:URL header: {e}")
        stats["reason"] = "error"
        return data_url, stats

    if mime not in _SUPPORTED_MIMES:
        stats["reason"] = "unsupported_mime"
        return data_url, stats

    try:
        raw = base64.b64decode(payload_b64)
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception as e:
        logger.warning(f"image_resize: PIL decode failed: {e}")
        stats["reason"] = "error"
        return data_url, stats

    w, h = img.size
    was_px = w * h
    stats["was_w"] = w
    stats["was_h"] = h
    stats["was_px"] = was_px

    if was_px < min_pixels:
        stats["new_w"], stats["new_h"], stats["new_px"] = w, h, was_px
        stats["reason"] = "too_small"
        return data_url, stats

    if was_px <= target_pixels:
        stats["new_w"], stats["new_h"], stats["new_px"] = w, h, was_px
        stats["reason"] = "in_range"
        return data_url, stats

    # Downscale, preserving aspect ratio so that new_w * new_h ~ target_pixels
    ratio = math.sqrt(target_pixels / was_px)
    new_w = max(1, int(round(w * ratio)))
    new_h = max(1, int(round(h * ratio)))

    try:
        # Drop alpha for PNG-encoding stability; preserve mode otherwise
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        out_io = BytesIO()
        resized.save(out_io, "PNG")
        new_b64 = base64.b64encode(out_io.getvalue()).decode("utf-8")
        new_url = f"data:image/png;base64,{new_b64}"
    except Exception as e:
        logger.warning(f"image_resize: PIL resize/encode failed: {e}")
        stats["reason"] = "error"
        return data_url, stats

    stats["new_w"] = new_w
    stats["new_h"] = new_h
    stats["new_px"] = new_w * new_h
    stats["resized"] = True
    stats["reason"] = "too_large"
    return new_url, stats


def resize_images_in_messages(
    messages: list,
    target_pixels: int,
    min_pixels: int = 200_704,
) -> dict:
    """Walk messages[].content[] in place, resize each image_url with data:URL.

    Mutates messages in place. Returns aggregate stats:
      {imgs, resized, was_pixels: list, new_pixels: list, reasons: list}
    Useful for logging into the VLM JSONL request log and truncate dump meta.

    Empty result (imgs=0) means there were no inline images - skip logging.
    """
    agg = {
        "imgs": 0,
        "resized": 0,
        "was_pixels": [],
        "new_pixels": [],
        "reasons": [],
    }
    if not isinstance(messages, list):
        return agg
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
            if not isinstance(url, str) or not url.startswith("data:"):
                continue
            new_url, stats = resize_image_in_data_url(
                url, target_pixels=target_pixels, min_pixels=min_pixels
            )
            iu["url"] = new_url
            agg["imgs"] += 1
            if stats.get("resized"):
                agg["resized"] += 1
            if stats.get("was_px") is not None:
                agg["was_pixels"].append(stats["was_px"])
            if stats.get("new_px") is not None:
                agg["new_pixels"].append(stats["new_px"])
            agg["reasons"].append(stats.get("reason", ""))
    return agg


def summarize_resize_stats(agg: dict) -> dict:
    """Compact aggregate suitable for JSONL log/meta. Skip if imgs==0."""
    imgs = agg.get("imgs", 0)
    if imgs <= 0:
        return {}
    was = agg.get("was_pixels") or []
    new = agg.get("new_pixels") or []
    avg_was = int(round(sum(was) / len(was))) if was else None
    avg_new = int(round(sum(new) / len(new))) if new else None
    return {
        "imgs": imgs,
        "resized": agg.get("resized", 0),
        "avg_was_px": avg_was,
        "avg_new_px": avg_new,
    }
