"""Detect text regions in an image and compute their area ratio."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from image_io import load_preview_bgr
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


@dataclass
class TextDetection:
    ratio: float
    box_count: int
    texts: List[str] = field(default_factory=list)
    boxes: List[List[float]] = field(default_factory=list)  # normalized [x1,y1,x2,y2]
    width: int = 0
    height: int = 0
    engine: str = "none"


_engine = None
_engine_error: Optional[str] = None


def is_image_name(name: str) -> bool:
    from pathlib import Path

    return Path(name).suffix.lower() in IMAGE_EXTS


def warmup() -> str:
    """Load the OCR engine once. Returns the engine name."""
    _get_engine()
    if _engine is not None:
        return "rapidocr"
    return f"opencv-fallback:{_engine_error or 'unavailable'}"


def _get_engine():
    global _engine, _engine_error
    if _engine is not None or _engine_error == "__failed__":
        return _engine
    try:
        from rapidocr import RapidOCR

        _engine = RapidOCR()
        return _engine
    except Exception as exc:  # noqa: BLE001
        _engine_error = "__failed__"
        # Keep the real message for status.
        globals()["_engine_error_detail"] = str(exc)
        return None


def analyze_image(path: str, max_side: int = 1600, min_score: float = 0.35) -> TextDetection:
    image_bgr, width, height = load_preview_bgr(path, max_side)
    if image_bgr is None:
        return TextDetection(ratio=0.0, box_count=0, width=width, height=height)

    work, scale = image_bgr, 1.0
    engine = _get_engine()
    if engine is not None:
        try:
            return _analyze_with_ocr(engine, work, width, height, min_score)
        except Exception:
            pass
    return _analyze_with_opencv(work, width, height)


def _load_bgr(path: str) -> Tuple[Optional[np.ndarray], int, int]:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            width, height = im.size
            rgb = np.array(im)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), width, height
    except Exception:
        return None, 0, 0


def _downscale(image_bgr: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image_bgr, 1.0
    scale = max_side / float(longest)
    resized = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _analyze_with_ocr(engine, work: np.ndarray, orig_w: int, orig_h: int, min_score: float) -> TextDetection:
    boxes_xy, texts = _run_ocr(engine, work, min_score)
    h, w = work.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    norm_boxes: List[List[float]] = []
    kept_texts: List[str] = []

    for poly, text in zip(boxes_xy, texts):
        cleaned = (text or "").strip()
        # Drop 1-char glyph guesses (sun/circles often become "O"/"0").
        if len(cleaned) <= 1:
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        if pts.shape[0] < 2:
            continue
        cv2.fillPoly(mask, [pts], 1)
        xs, ys = pts[:, 0], pts[:, 1]
        x1, y1 = float(xs.min()) / w, float(ys.min()) / h
        x2, y2 = float(xs.max()) / w, float(ys.max()) / h
        norm_boxes.append([_clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2)])
        kept_texts.append(cleaned)

    area = int(h * w)
    ratio = float(mask.sum()) / area if area else 0.0
    return TextDetection(
        ratio=ratio,
        box_count=len(norm_boxes),
        texts=kept_texts[:12],
        boxes=norm_boxes,
        width=orig_w,
        height=orig_h,
        engine="rapidocr",
    )


def _run_ocr(engine, work: np.ndarray, min_score: float) -> Tuple[List, List[str]]:
    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    out = engine(rgb)
    boxes: List = []
    texts: List[str] = []

    # RapidOCR >= 2/3 object API
    if hasattr(out, "boxes"):
        raw_boxes = out.boxes if out.boxes is not None else []
        raw_txts = list(getattr(out, "txts", None) or [])
        raw_scores = list(getattr(out, "scores", None) or [])
        for i, box in enumerate(raw_boxes):
            score = raw_scores[i] if i < len(raw_scores) else 1.0
            if score is not None and float(score) < min_score:
                continue
            boxes.append(box)
            texts.append(raw_txts[i] if i < len(raw_txts) else "")
        return boxes, texts

    # Older tuple API: (result, elapse)
    result = out[0] if isinstance(out, tuple) else out
    if not result:
        return [], []
    for item in result:
        if not item:
            continue
        box = item[0]
        text = item[1] if len(item) > 1 else ""
        score = item[2] if len(item) > 2 else 1.0
        if score is not None and float(score) < min_score:
            continue
        boxes.append(box)
        texts.append(str(text))
    return boxes, texts


def _analyze_with_opencv(work: np.ndarray, orig_w: int, orig_h: int) -> TextDetection:
    """Heuristic text-like region detector used if OCR models are unavailable."""
    h, w = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 40), 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros((h, w), dtype=np.uint8)
    boxes: List[List[float]] = []
    min_area = max(40.0, 0.0004 * h * w)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_area:
            continue
        aspect = bw / float(bh or 1)
        if aspect < 1.2 or aspect > 25:
            continue
        if bh > h * 0.35:
            continue
        cv2.rectangle(mask, (x, y), (x + bw, y + bh), 1, -1)
        boxes.append(
            [
                _clamp01(x / w),
                _clamp01(y / h),
                _clamp01((x + bw) / w),
                _clamp01((y + bh) / h),
            ]
        )

    area = int(h * w)
    ratio = float(mask.sum()) / area if area else 0.0
    return TextDetection(
        ratio=ratio,
        box_count=len(boxes),
        texts=[],
        boxes=boxes,
        width=orig_w,
        height=orig_h,
        engine="opencv",
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def union_histogram(ratios: Sequence[float], bins: int = 10) -> List[dict]:
    edges = [i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for ratio in ratios:
        idx = min(bins - 1, max(0, int(ratio * bins)))
        counts[idx] += 1
    return [
        {"from": edges[i], "to": edges[i + 1], "count": counts[i]}
        for i in range(bins)
    ]
