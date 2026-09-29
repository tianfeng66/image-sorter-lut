"""Trim uniform black/white borders and save a new file with original encoding."""

from __future__ import annotations

import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from image_io import load_preview_bgr


@dataclass
class CropPlan:
    width: int
    height: int
    box: Tuple[int, int, int, int]
    cropped: bool
    border: str
    trim: Tuple[int, int, int, int]

    def as_public(self) -> dict:
        left, top, right, bottom = self.box
        t, b, l, r = self.trim
        return {
            "width": self.width,
            "height": self.height,
            "crop_box": [left, top, right, bottom],
            "crop_width": max(0, right - left),
            "crop_height": max(0, bottom - top),
            "cropped": self.cropped,
            "border": self.border,
            "trim_top": t,
            "trim_bottom": b,
            "trim_left": l,
            "trim_right": r,
        }


def analyze_borders(path: str, max_side: int = 1600) -> CropPlan:
    preview, file_w, file_h = load_preview_bgr(path, max_side)
    if preview is None or file_w <= 0 or file_h <= 0:
        return CropPlan(0, 0, (0, 0, 0, 0), False, "none", (0, 0, 0, 0))
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im) or im
            width, height = im.size
    except Exception:
        width, height = file_w, file_h
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    bgr = preview.astype(np.float32) / 255.0
    ph, pw = gray.shape[:2]
    top_p, bottom_p, left_p, right_p, kind = _content_bounds(gray, bgr)
    sx, sy = width / pw, height / ph
    left = min(width - 1, max(0, int(np.ceil(left_p * sx))))
    top = min(height - 1, max(0, int(np.ceil(top_p * sy))))
    right = min(width, max(left + 1, int(np.floor(right_p * sx))))
    bottom = min(height, max(top + 1, int(np.floor(bottom_p * sy))))
    box = (left, top, right, bottom)
    if not _box_sane(width, height, box):
        box = (0, 0, width, height)
        kind = "none"
    trimmed = (box != (0, 0, width, height))
    t, b, l, r = top, height - bottom, left, width - right
    return CropPlan(width, height, box, trimmed, kind if trimmed else "none", (t, b, l, r))


def save_cropped(src: str, dest: str, box: Optional[Tuple[int, int, int, int]] = None) -> str:
    src_path = Path(src)
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if box is None or box == (0, 0, 0, 0):
        if src_path.resolve() != dest_path.resolve():
            shutil.copy2(src_path, dest_path)
        return str(dest_path.resolve())
    meta = _inspect_image(src_path)
    crop_box = _clamp_box(tuple(int(v) for v in box[:4]), meta["disp_w"], meta["disp_h"])
    if crop_box == (0, 0, meta["disp_w"], meta["disp_h"]):
        if src_path.resolve() != dest_path.resolve():
            shutil.copy2(src_path, dest_path)
        return str(dest_path.resolve())
    if meta["fmt"] == "JPEG":
        _lossless_jpeg_crop(src_path, dest_path, crop_box, meta)
        return str(dest_path.resolve())
    if meta["fmt"] == "PNG":
        _lossless_png_crop(src_path, dest_path, crop_box, meta)
        return str(dest_path.resolve())
    with Image.open(src_path) as im:
        im = ImageOps.exif_transpose(im) or im
        cropped = im.crop(crop_box)
        if cropped.mode != meta["enc"]["mode"] and meta["enc"]["mode"] in ("RGB", "CMYK", "L"):
            cropped = cropped.convert(meta["enc"]["mode"])
        _save_like_original(cropped, dest_path, meta["fmt"], meta["enc"])
    return str(dest_path.resolve())


def _inspect_image(path: Path) -> Dict[str, Any]:
    with Image.open(path) as im:
        fmt = (im.format or path.suffix.lstrip(".")).upper()
        if fmt == "JPG":
            fmt = "JPEG"
        stored_w, stored_h = im.size
        orientation = _exif_orientation(im)
        enc = _capture_encoding(im, fmt)
        oriented = ImageOps.exif_transpose(im) or im
        disp_w, disp_h = oriented.size
    return {
        "fmt": fmt,
        "stored_w": stored_w,
        "stored_h": stored_h,
        "disp_w": disp_w,
        "disp_h": disp_h,
        "orientation": orientation,
        "enc": enc,
    }


def _content_bounds(gray: np.ndarray, bgr: np.ndarray) -> Tuple[int, int, int, int, str]:
    h, w = gray.shape[:2]
    top_n, top_k = _flat_run(gray, bgr, "row", 0, h, 0, w, reverse=False)
    bot_n, bot_k = _flat_run(gray, bgr, "row", 0, h, 0, w, reverse=True)
    top, bottom, kind_tb = _paired_trim(top_n, top_k, bot_n, bot_k, h)
    left_n, left_k = _flat_run(gray, bgr, "col", top, bottom, 0, w, reverse=False)
    right_n, right_k = _flat_run(gray, bgr, "col", top, bottom, 0, w, reverse=True)
    left, right, kind_lr = _paired_trim(left_n, left_k, right_n, right_k, w)
    if top == 0 and bottom == h and left == 0 and right == w:
        return 0, h, 0, w, "none"
    if bottom - top < max(8, int(h * 0.35)) or right - left < max(8, int(w * 0.35)):
        return 0, h, 0, w, "none"
    if not _inner_looks_like_frame(gray, bgr, top, bottom, left, right):
        return 0, h, 0, w, "none"
    kinds = [k for k in (kind_tb, kind_lr) if k in ("black", "white")]
    if not kinds:
        border = "none"
    elif all(k == kinds[0] for k in kinds):
        border = kinds[0]
    else:
        border = "mixed"
    return top, bottom, left, right, border


def _flat_run(gray, bgr, axis: str, y0: int, y1: int, x0: int, x1: int, reverse: bool) -> Tuple[int, str]:
    if axis == "row":
        indices = range(y1 - 1, y0 - 1, -1) if reverse else range(y0, y1)
    else:
        indices = range(x1 - 1, x0 - 1, -1) if reverse else range(x0, x1)
    count = 0
    kind = ""
    for idx in indices:
        if axis == "row":
            g = gray[idx, x0:x1]
            c = bgr[idx, x0:x1]
        else:
            g = gray[y0:y1, idx]
            c = bgr[y0:y1, idx]
        hit = _letterbox_strip(g, c)
        if not hit:
            break
        if kind and hit != kind:
            break
        kind = hit
        count += 1
    return count, kind


def _paired_trim(a_n: int, a_k: str, b_n: int, b_k: str, size: int) -> Tuple[int, int, str]:
    min_bar = max(6, int(round(size * 0.012)))
    if a_n < min_bar or b_n < min_bar or not a_k or not b_k or a_k != b_k:
        return 0, size, "none"
    if max(a_n, b_n) / max(1, min(a_n, b_n)) > 2.6:
        return 0, size, "none"
    if a_n + b_n > size * 0.55:
        return 0, size, "none"
    return a_n, size - b_n, a_k


def _inner_looks_like_frame(gray, bgr, top, bottom, left, right) -> bool:
    """Reject object-on-black crops: a film frame keeps picture in the corners."""
    h = bottom - top
    w = right - left
    if h < 8 or w < 8:
        return False
    ph, pw = max(2, int(h * 0.08)), max(2, int(w * 0.08))
    patches = [
        (top, top + ph, left, left + pw),
        (top, top + ph, right - pw, right),
        (bottom - ph, bottom, left, left + pw),
        (bottom - ph, bottom, right - pw, right),
    ]
    bar_corners = 0
    for y0, y1, x0, x1 in patches:
        if _letterbox_strip(gray[y0:y1, x0:x1].ravel(), bgr[y0:y1, x0:x1].reshape(-1, 3)):
            bar_corners += 1
    return bar_corners <= 1


def _letterbox_strip(gray: np.ndarray, bgr: Optional[np.ndarray] = None) -> str:
    strip = np.asarray(gray, dtype=np.float32).ravel()
    if strip.size < 8:
        return ""
    med = float(np.median(strip))
    std = float(strip.std())
    span = float(strip.max() - strip.min())
    struct = float(np.mean(np.abs(np.diff(strip)))) if strip.size >= 3 else 0.0
    if std > 0.028 or span > 0.10 or struct > 0.012:
        return ""
    if float(np.mean(np.abs(strip - med) <= 0.04)) < 0.985:
        return ""
    if bgr is not None:
        pixels = np.asarray(bgr, dtype=np.float32).reshape(-1, 3)
        chroma = float(np.mean(pixels.max(axis=1) - pixels.min(axis=1)))
        if chroma > 0.06:
            return ""
    if med <= 0.06:
        return "black"
    if med >= 0.94:
        return "white"
    return ""


def _box_sane(width: int, height: int, box: Tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = box
    if right - left < max(8, int(width * 0.2)):
        return False
    if bottom - top < max(8, int(height * 0.2)):
        return False
    if (right - left) * (bottom - top) < width * height * 0.22:
        return False
    return True


def _capture_encoding(im: Image.Image, fmt: str) -> Dict[str, Any]:
    info = dict(im.info or {})
    qtables = None
    raw_q = getattr(im, "quantization", None)
    if raw_q:
        qtables = {int(k): list(v) for k, v in raw_q.items()}
    return {
        "mode": im.mode,
        "info": info,
        "icc": info.get("icc_profile"),
        "dpi": info.get("dpi"),
        "exif": info.get("exif"),
        "progressive": bool(info.get("progressive")),
        "comment": info.get("comment"),
        "qtables": qtables,
        "subsampling": _jpeg_subsampling(im) if fmt == "JPEG" else None,
        "webp_lossless": bool(info.get("lossless")),
        "compress_level": info.get("compress_level"),
    }


def _jpeg_subsampling(im: Image.Image) -> int:
    """Pillow: 0=4:4:4, 1=4:2:2, 2=4:2:0."""
    layer = getattr(im, "layer", None) or []
    if not layer:
        return 2
    try:
        _cid, h_samp, v_samp, _q = layer[0]
    except (TypeError, ValueError, IndexError):
        return 2
    if h_samp >= 2 and v_samp >= 2:
        return 2
    if h_samp >= 2:
        return 1
    return 0


def _exif_orientation(im: Image.Image) -> int:
    try:
        value = im.getexif().get(0x0112, 1)
        return int(value or 1)
    except Exception:
        return 1


def _display_to_stored(x: int, y: int, sw: int, sh: int, orientation: int) -> Tuple[int, int]:
    """Map a display-space pixel to the JPEG's stored pixel (Pillow exif_transpose inverse)."""
    o = orientation
    if o == 2:
        return sw - 1 - x, y
    if o == 3:
        return sw - 1 - x, sh - 1 - y
    if o == 4:
        return x, sh - 1 - y
    if o == 5:
        return y, x
    if o == 6:
        return y, sh - 1 - x
    if o == 7:
        return sw - 1 - y, sh - 1 - x
    if o == 8:
        return sw - 1 - y, x
    return x, y


def _display_box_to_stored(
    box: Tuple[int, int, int, int],
    stored_w: int,
    stored_h: int,
    orientation: int,
) -> Tuple[int, int, int, int]:
    left, top, right, bottom = box
    pts = [
        _display_to_stored(left, top, stored_w, stored_h, orientation),
        _display_to_stored(right - 1, top, stored_w, stored_h, orientation),
        _display_to_stored(left, bottom - 1, stored_w, stored_h, orientation),
        _display_to_stored(right - 1, bottom - 1, stored_w, stored_h, orientation),
    ]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    sl, st, sr, sb = min(xs), min(ys), max(xs) + 1, max(ys) + 1
    sl = min(stored_w - 1, max(0, sl))
    st = min(stored_h - 1, max(0, st))
    sr = min(stored_w, max(sl + 1, sr))
    sb = min(stored_h, max(st + 1, sb))
    return sl, st, sr, sb


def _exif_reset_orientation(exif: Any) -> Any:
    if not exif:
        return None
    try:
        im = Image.new("RGB", (1, 1))
        loaded = im.getexif()
        loaded.load(exif)
        loaded[0x0112] = 1
        return loaded.tobytes()
    except Exception:
        return None


def _clamp_box(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    left, top, right, bottom = box
    left = min(width - 1, max(0, left))
    top = min(height - 1, max(0, top))
    right = min(width, max(left + 1, right))
    bottom = min(height, max(top + 1, bottom))
    return left, top, right, bottom


def _expand_to_mcu(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
    mx: int = 16,
    my: int = 16,
) -> Tuple[int, int, int, int]:
    left, top, right, bottom = box
    left = max(0, (left // mx) * mx)
    top = max(0, (top // my) * my)
    right = min(width, ((right + mx - 1) // mx) * mx)
    bottom = min(height, ((bottom + my - 1) // my) * my)
    return _clamp_box((left, top, right, bottom), width, height)


def _find_jpegtran() -> str:
    for candidate in (
        "/usr/local/bin/jpegtran",
        "/opt/homebrew/bin/jpegtran",
        shutil.which("jpegtran"),
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError("未找到 jpegtran，无法无损裁切 JPEG。请安装 libjpeg-turbo。")


def _run_jpegtran(src: Path, dest: Path, extra: list[str]) -> None:
    jpegtran = _find_jpegtran()
    tmp = dest.with_name(dest.name + ".jpegtran.tmp")
    if tmp.exists():
        tmp.unlink()
    cmd = [jpegtran, "-copy", "all", *extra, "-outfile", str(tmp), str(src)]
    try:
        done = subprocess.run(cmd, check=False, capture_output=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"jpegtran 无法运行：{exc}") from exc
    err = (done.stderr or b"").decode("utf-8", "ignore").strip()
    if done.returncode != 0 or not tmp.is_file() or tmp.stat().st_size < 32:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(err or f"jpegtran 失败（{done.returncode}）")
    tmp.replace(dest)


def _lossless_jpeg_crop(
    src: Path,
    dest: Path,
    display_box: Tuple[int, int, int, int],
    meta: Dict[str, Any],
) -> None:
    orientation = meta["orientation"]
    stored_w, stored_h = meta["stored_w"], meta["stored_h"]
    stored_box = (
        display_box
        if orientation <= 1
        else _display_box_to_stored(display_box, stored_w, stored_h, orientation)
    )
    stored_box = _clamp_box(stored_box, stored_w, stored_h)
    last_error = None
    for box in (stored_box, _expand_to_mcu(stored_box, stored_w, stored_h)):
        left, top, right, bottom = box
        crop = f"{right - left}x{bottom - top}+{left}+{top}"
        try:
            _run_jpegtran(src, dest, ["-crop", crop])
            return
        except RuntimeError as exc:
            last_error = exc
    raise RuntimeError(f"JPEG 无损裁切失败，未重新编码。{last_error}")


_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_PNG_ANCILLARY = {b"iCCP", b"gAMA", b"cHRM", b"sRGB", b"pHYs", b"tEXt", b"zTXt", b"iTXt"}


def _png_header(path: Path) -> Optional[Dict[str, int]]:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 33 or not data.startswith(_PNG_SIG):
        return None
    length = int.from_bytes(data[8:12], "big")
    if data[12:16] != b"IHDR" or length < 13:
        return None
    width, height, bit_depth, color_type = struct.unpack(">IIBB", data[16:26])
    return {"width": width, "height": height, "bit_depth": bit_depth, "color_type": color_type}


def _iter_png_chunks(data: bytes) -> List[Tuple[bytes, bytes]]:
    chunks: List[Tuple[bytes, bytes]] = []
    pos = 8
    while pos + 12 <= len(data):
        length = int.from_bytes(data[pos : pos + 4], "big")
        typ = data[pos + 4 : pos + 8]
        end = pos + 12 + length
        if end > len(data):
            break
        chunks.append((typ, data[pos:end]))
        pos = end
        if typ == b"IEND":
            break
    return chunks


def _png_graft_ancillary(src: Path, dest: Path) -> None:
    extras = [(typ, chunk) for typ, chunk in _iter_png_chunks(src.read_bytes()) if typ in _PNG_ANCILLARY]
    if not extras:
        return
    dest_chunks = _iter_png_chunks(dest.read_bytes())
    if not dest_chunks:
        return
    have = {typ for typ, _ in dest_chunks}
    extras = [chunk for typ, chunk in extras if typ not in have]
    if not extras:
        return
    out = bytearray(_PNG_SIG)
    for typ, chunk in dest_chunks:
        out.extend(chunk)
        if typ == b"IHDR":
            for extra in extras:
                out.extend(extra)
    dest.write_bytes(out)


def _numpy_exif_transpose(arr: np.ndarray, orientation: int) -> np.ndarray:
    if orientation <= 1:
        return arr
    ops = {
        2: lambda a: np.fliplr(a),
        3: lambda a: np.rot90(a, 2),
        4: lambda a: np.flipud(a),
        5: lambda a: np.transpose(a, (1, 0, 2) if a.ndim == 3 else (1, 0)),
        6: lambda a: np.rot90(a, 1),
        7: lambda a: np.rot90(np.fliplr(a), 1),
        8: lambda a: np.rot90(a, 3),
    }
    fn = ops.get(orientation)
    return fn(arr) if fn else arr


def _lossless_png_crop(
    src: Path,
    dest: Path,
    display_box: Tuple[int, int, int, int],
    meta: Dict[str, Any],
) -> None:
    header = _png_header(src)
    if header and header["color_type"] == 3:
        with Image.open(src) as im:
            im.load()
            cropped = im.crop(display_box)
            kwargs: Dict[str, Any] = {}
            if meta["enc"].get("icc"):
                kwargs["icc_profile"] = meta["enc"]["icc"]
            if meta["enc"].get("dpi"):
                kwargs["dpi"] = meta["enc"]["dpi"]
            cropped.save(dest, format="PNG", compress_level=1, **kwargs)
        _png_graft_ancillary(src, dest)
        return
    arr = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError("无法按原位深读取 PNG。")
    if meta.get("orientation", 1) > 1:
        arr = _numpy_exif_transpose(arr, meta["orientation"])
    left, top, right, bottom = display_box
    cropped = arr[top:bottom, left:right]
    if cropped.size == 0:
        raise RuntimeError("裁切框为空。")
    channels = 1 if cropped.ndim == 2 else cropped.shape[2]
    sample = 2 if cropped.dtype == np.uint16 else 1
    raw = max(1, meta["disp_w"] * meta["disp_h"] * channels * sample)
    level = 0 if src.stat().st_size / raw > 0.55 else 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest), cropped, [int(cv2.IMWRITE_PNG_COMPRESSION), int(level)]):
        raise RuntimeError("写入 PNG 失败。")
    _png_graft_ancillary(src, dest)


def _save_like_original(image: Image.Image, dest: Path, fmt: str, enc: Dict[str, Any]) -> None:
    if fmt == "JPEG":
        raise RuntimeError("JPEG 必须走 jpegtran 无损裁切，禁止重新编码。")
    kwargs: Dict[str, Any] = {}
    if enc.get("icc"):
        kwargs["icc_profile"] = enc["icc"]
    if enc.get("dpi"):
        kwargs["dpi"] = enc["dpi"]
    if enc.get("comment"):
        kwargs["comment"] = enc["comment"]
    if fmt == "PNG":
        if enc.get("compress_level") is not None:
            kwargs["compress_level"] = enc["compress_level"]
        image.save(dest, format="PNG", **kwargs)
        return
    if fmt == "WEBP":
        if enc.get("webp_lossless"):
            kwargs["lossless"] = True
            image.save(dest, format="WEBP", **kwargs)
            return
        kwargs["quality"] = 98
        kwargs["method"] = 6
        image.save(dest, format="WEBP", **kwargs)
        return
    image.save(dest, format=fmt or dest.suffix.lstrip(".").upper() or "PNG", **kwargs)
