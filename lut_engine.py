"""Build a 3D LUT from reference photos and/or a text description, then apply it."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from detector import is_image_name

CUBE_SIZE = 65
HALD_CUBE = 64
HUE_BANDS = (
    ("red", 0.0),
    ("yellow", 60.0),
    ("green", 120.0),
    ("cyan", 180.0),
    ("blue", 240.0),
    ("magenta", 300.0),
)
HUE_BAND_WIDTH = 78.0


def _max3(rgb: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])


def _min3(rgb: np.ndarray) -> np.ndarray:
    return np.minimum(np.minimum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])


def rgb_to_hsv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.ascontiguousarray(np.clip(rgb, 0.0, 1.0), dtype=np.float32)
    shape = rgb.shape[:-1]
    hsv = cv2.cvtColor(rgb if rgb.ndim == 3 else rgb.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(shape + (3,))
    return hsv[..., 0], hsv[..., 1], hsv[..., 2]


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    hsv = np.empty(np.shape(h) + (3,), dtype=np.float32)
    hsv[..., 0] = np.mod(h, 360.0)
    hsv[..., 1] = np.clip(s, 0.0, 1.5)
    hsv[..., 2] = v
    if hsv.ndim != 3:
        flat = cv2.cvtColor(hsv.reshape(1, -1, 3), cv2.COLOR_HSV2RGB)
        return np.clip(flat.reshape(hsv.shape), 0.0, 1.0)
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hue_weight(h: np.ndarray, center: float, width: float) -> np.ndarray:
    d = np.abs(np.mod(h - center + 180.0, 360.0) - 180.0)
    return _smoothstep(1.0 - d / max(width, 1e-6))


def _chroma_gate(sat: np.ndarray) -> np.ndarray:
    """Near-neutral pixels have unstable hue; don't apply hue-specific moves there."""
    return _smoothstep((sat - 0.03) / 0.13)


def _stable_color_gate(y: np.ndarray, sat: np.ndarray) -> np.ndarray:
    """0 on near-black hair / pale wash — HSV moves there turn into JPEG mosaic."""
    not_dark = _smoothstep((y - 0.055) / 0.11)
    not_pale = 1.0 - _smoothstep((y - 0.84) / 0.10) * _smoothstep(1.0 - sat / 0.16)
    return not_dark * not_pale * _chroma_gate(sat)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _odd_k(n: int, lo: int = 3, hi: int = 81) -> int:
    n = int(np.clip(int(n), lo, hi))
    return n if n % 2 else n + 1


def _source_stats(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mx = _max3(rgb)
    chroma = mx - _min3(rgb)
    y = _luma(rgb)
    sat = np.where(mx > 1e-5, chroma / np.maximum(mx, 1e-5), 0.0)
    return y, sat, chroma


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float = 1e-3) -> np.ndarray:
    """Edge-aware smooth: weights follow hair/sky edges instead of a blurred blob."""
    r = _odd_k(radius, 3, 61)
    g = guide.astype(np.float32)
    p = src.astype(np.float32)
    k = (r, r)
    mean_i = cv2.blur(g, k)
    mean_p = cv2.blur(p, k)
    mean_ii = cv2.blur(g * g, k)
    mean_ip = cv2.blur(g * p, k)
    var_i = np.maximum(mean_ii - mean_i * mean_i, 0.0)
    a = (mean_ip - mean_i * mean_p) / (var_i + eps)
    b = mean_p - a * mean_i
    return cv2.blur(a, k) * g + cv2.blur(b, k)


def estimate_adjust_room(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel limiter, then guided-filtered so protection hugs real edges."""
    y, sat, chroma = _source_stats(rgb)
    if rgb.ndim != 3 or min(int(rgb.shape[0]), int(rgb.shape[1])) < 16:
        return np.full(y.shape, 0.75, dtype=np.float32)

    short = min(int(rgb.shape[0]), int(rgb.shape[1]))
    k_var = _odd_k(round(short * 0.05), 17, 81)
    y32 = y.astype(np.float32)
    ch32 = chroma.astype(np.float32)
    mu = cv2.blur(y32, (k_var, k_var))
    var = np.maximum(cv2.blur(y32 * y32, (k_var, k_var)) - mu * mu, 0.0)
    smooth = _smoothstep(1.0 - var / 0.0055)
    sky = np.clip(
        smooth
        * np.maximum(
            _smoothstep((y32 - 0.40) / 0.22),
            _smoothstep((y32 - 0.44) / 0.24) * _smoothstep(1.0 - ch32 / 0.30),
        ),
        0.0,
        1.0,
    )
    hair = _smoothstep((0.20 - y32) / 0.13) * _smoothstep(1.0 - np.minimum(ch32, sat) / 0.16)
    void = _smoothstep((0.10 - y32) / 0.07)
    fragile = np.clip(np.maximum(np.maximum(sky, hair), void), 0.0, 1.0)
    crush = _smoothstep((sat - 0.32) / 0.22) * _smoothstep((0.44 - y) / 0.16)
    room = (1.0 - 0.92 * fragile) * (1.0 - 0.28 * crush)
    room = _guided_filter(y32, room.astype(np.float32), radius=max(7, short // 28), eps=4e-4)
    return np.clip(room, 0.05, 1.0)


def _adjust_room(rgb: np.ndarray) -> np.ndarray:
    return estimate_adjust_room(rgb)


def _hue_dist(h1: np.ndarray, h2) -> np.ndarray:
    return np.abs(np.mod(h1 - h2 + 180.0, 360.0) - 180.0)


def _cast_hsv(cast: np.ndarray) -> Tuple[float, float]:
    c = np.asarray(cast, dtype=np.float32).reshape(3)
    c = c - float(c.mean())
    probe = np.clip(0.5 + c * 2.4, 0.0, 1.0).reshape(1, 1, 3)
    h, s, _v = rgb_to_hsv(probe)
    return float(h[0, 0]), float(s[0, 0])


def _look_tint_hsv(src: np.ndarray, graded: np.ndarray, look_cast: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Dominant hue/sat of the LUT cast (gray probe first, else image neutrals)."""
    if look_cast is not None:
        return _cast_hsv(look_cast)
    y = _luma(src)
    _h, s, _v = rgb_to_hsv(src)
    mid = _smoothstep((y - 0.12) / 0.14) * _smoothstep((0.88 - y) / 0.14)
    w_neu = mid * _smoothstep(1.0 - s / 0.16)
    w = w_neu if float(np.sum(w_neu)) > 400.0 else mid
    tint = (graded - src) - _luma(graded - src)[..., None]
    den = float(np.sum(w)) + 1e-6
    t = (tint * w[..., None]).reshape(-1, 3).sum(axis=0) / den
    return _cast_hsv(t)


def _complement_gate(
    src: np.ndarray,
    graded: np.ndarray,
    y: np.ndarray,
    sat: np.ndarray,
    look_cast: Optional[np.ndarray] = None,
) -> np.ndarray:
    """1 = source hue fights the LUT (e.g. warm skin vs teal/green)."""
    look_h, look_s = _look_tint_hsv(src, graded, look_cast)
    h_src, s_src, _v = rgb_to_hsv(src)
    d = graded - src
    cool_pull = _smoothstep(((0.5 * (d[..., 1] + d[..., 2]) - d[..., 0]) - 0.006) / 0.024)
    skin = (
        _smoothstep(1.0 - _hue_dist(h_src, 24.0) / 40.0)
        * _smoothstep((s_src - 0.08) / 0.10)
        * _smoothstep((0.58 - s_src) / 0.22)
        * _smoothstep((y - 0.16) / 0.12)
        * _smoothstep((0.84 - y) / 0.14)
    )
    if look_s < 0.014:
        return np.clip(0.85 * skin * cool_pull, 0.0, 1.0).astype(np.float32)
    opp = _hue_dist(h_src, look_h)
    comp = _smoothstep((opp - 78.0) / 52.0) * _smoothstep((s_src - 0.07) / 0.11)
    cool_look = _smoothstep(1.0 - _hue_dist(np.float32(look_h), 155.0) / 75.0)
    return np.clip(np.maximum(0.65 * comp, 0.70 * skin * np.maximum(cool_look, cool_pull)), 0.0, 1.0).astype(np.float32)


def _analogous_gate(
    src: np.ndarray,
    graded: np.ndarray,
    y: np.ndarray,
    sat: np.ndarray,
    look_cast: Optional[np.ndarray] = None,
) -> np.ndarray:
    """1 = source hue already sits next to the LUT, so extra pull is wasted."""
    look_h, look_s = _look_tint_hsv(src, graded, look_cast)
    if look_s < 0.014:
        return np.zeros(y.shape, dtype=np.float32)
    h_src, s_src, _v = rgb_to_hsv(src)
    near = _hue_dist(h_src, look_h)
    analog = _smoothstep((50.0 - near) / 34.0) * _smoothstep((s_src - 0.10) / 0.12)
    sky = _smoothstep((y.astype(np.float32) - 0.42) / 0.22) * _smoothstep(1.0 - sat / 0.30)
    analog = analog * (1.0 - 0.92 * sky)
    return np.clip(analog, 0.0, 1.0).astype(np.float32)


def apply_adjust_room(
    src: np.ndarray,
    graded: np.ndarray,
    look_cast: Optional[np.ndarray] = None,
    skin_only: bool = False,
    room: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Whole-frame look; ease complementary and already-close hues."""
    src = np.clip(src.astype(np.float32), 0.0, 1.0)
    graded = np.clip(graded.astype(np.float32), 0.0, 1.0)
    y, sat, _chroma = _source_stats(src)
    if skin_only:
        luma_w = np.ones_like(y, dtype=np.float32)
        chroma_w = np.ones_like(y, dtype=np.float32)
    else:
        if room is None:
            room = estimate_adjust_room(src)
        crush = _smoothstep((sat - 0.28) / 0.20) * _smoothstep((0.48 - y) / 0.18)
        pop = _smoothstep((room - 0.40) / 0.32)
        luma_w = np.clip(0.18 + 0.82 * room * (1.0 - 0.55 * crush) * (1.0 + 0.20 * pop), 0.06, 1.0)
        chroma_w = np.clip(0.16 + 0.84 * room * (1.0 + 0.24 * pop), 0.06, 1.0)
    comp = _complement_gate(src, graded, y, sat, look_cast)
    analog = _analogous_gate(src, graded, y, sat, look_cast)
    luma_w = luma_w * (1.0 - 0.08 * comp) * (1.0 - 0.10 * analog)
    chroma_w = chroma_w * (1.0 - 0.40 * comp) * (1.0 - 0.36 * analog)
    if src.ndim == 3 and min(int(src.shape[0]), int(src.shape[1])) >= 16:
        guide = y.astype(np.float32)
        rad = max(7, min(int(src.shape[0]), int(src.shape[1])) // 32)
        luma_w = np.clip(_guided_filter(guide, luma_w.astype(np.float32), rad, 4e-4), 0.0, 1.0)
        chroma_w = np.clip(_guided_filter(guide, chroma_w.astype(np.float32), rad, 4e-4), 0.0, 1.0)
    src_y = y[..., None]
    gr_y = _luma(graded)[..., None]
    out = src_y + (gr_y - src_y) * luma_w[..., None]
    out = out + (src - src_y) + ((graded - gr_y) - (src - src_y)) * chroma_w[..., None]
    mix = np.minimum(luma_w, chroma_w)
    return np.clip(out, 0.0, 1.0).astype(np.float32), mix


def default_tune() -> dict:
    tune = {
        "hue": 0.0,
        "sat": 0.0,
        "gamma": 1.0,
        "contrast": 1.0,
        "black": 0.0,
        "mid_tint": [0.0, 0.0, 0.0],
    }
    for name, _center in HUE_BANDS:
        tune[f"hue_{name}"] = 0.0
    return tune


def _bands_from_moves(moves) -> dict:
    bands = {name: 0.0 for name, _center in HUE_BANDS}
    for center, width, amount in moves or []:
        cover = min(1.0, float(width) / 80.0)
        for name, band_c in HUE_BANDS:
            w = float(hue_weight(np.array([band_c], dtype=np.float32), float(center), float(width))[0])
            bands[name] += float(amount) * w * cover
    return {name: float(np.clip(val, -60.0, 60.0)) for name, val in bands.items()}


def tune_from_params(params: Optional[dict]) -> dict:
    """Map a LUT's color bias to banding-safe hue / curve knobs."""
    p = {**default_params(), **(params or {})}
    hue = 0.0
    for center, width, amount in p.get("hue_moves") or []:
        hue += float(amount) * min(1.0, float(width) / 90.0)
    sat = 0.0
    for center, width, amount in p.get("sat_moves") or []:
        sat += float(amount) * (1.0 if float(width) >= 160.0 else 0.4)
    tint = _as_arr3(p.get("mid_tint") or [0, 0, 0])
    if abs(float(tint[1] - tint[0])) + abs(float(tint[2] - tint[0])) > 0.02:
        th, ts, _tv = rgb_to_hsv(np.clip(0.5 + tint * 1.8, 0, 1).reshape(1, 1, 3))
        hue += (float(th[0, 0]) - 90.0) * float(ts[0, 0]) * 0.12
    bands = _bands_from_moves(p.get("hue_moves"))
    master = 0.0
    if max(abs(v) for v in bands.values()) < 0.35 and abs(hue) > 0.35:
        master = float(np.clip(hue, -36.0, 36.0))
    out = {
        "hue": master,
        "sat": float(np.clip(sat, -0.52, 0.32)),
        "gamma": float(np.clip(p.get("gamma") or 1.0, 0.78, 1.26)),
        "contrast": float(np.clip(p.get("contrast") or 1.0, 0.86, 1.18)),
        "black": float(np.clip(p.get("black") or 0.0, 0.0, 0.08)),
        "mid_tint": [float(x) for x in tint.tolist()],
    }
    for name, val in bands.items():
        out[f"hue_{name}"] = val
    return out


def mix_tunes(tunes: Sequence[Tuple[dict, float]]) -> dict:
    acc = {"hue": 0.0, "sat": 0.0, "gamma": 0.0, "contrast": 0.0, "black": 0.0}
    for name, _center in HUE_BANDS:
        acc[f"hue_{name}"] = 0.0
    tint = np.zeros(3, dtype=np.float32)
    wsum = 0.0
    for tune, weight in tunes:
        w = max(0.0, float(weight))
        if w <= 1e-6:
            continue
        t = {**default_tune(), **(tune or {})}
        acc["hue"] += t["hue"] * w
        acc["sat"] += t["sat"] * w
        acc["gamma"] += t["gamma"] * w
        acc["contrast"] += t["contrast"] * w
        acc["black"] += t["black"] * w
        for name, _center in HUE_BANDS:
            acc[f"hue_{name}"] += float(t.get(f"hue_{name}") or 0.0) * w
        tint += np.asarray(t["mid_tint"], dtype=np.float32)[:3] * w
        wsum += w
    if wsum < 1e-6:
        return default_tune()
    return {
        "hue": float(acc["hue"] / wsum),
        "sat": float(acc["sat"] / wsum),
        "gamma": float(acc["gamma"] / wsum),
        "contrast": float(acc["contrast"] / wsum),
        "black": float(acc["black"] / wsum),
        "mid_tint": [float(x) for x in (tint / wsum).tolist()],
        **{f"hue_{name}": float(acc[f"hue_{name}"] / wsum) for name, _center in HUE_BANDS},
    }


def apply_tune(rgb: np.ndarray, tune: dict, strength: float = 1.0) -> np.ndarray:
    """Hue + tone + sat only — no 3D LUT, so sky/hair stay free of mosaic blocks."""
    src = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    t = {**default_tune(), **(tune or {})}
    s = float(np.clip(strength, 0.0, 1.5))
    if s <= 1e-6:
        return src
    y = _luma(src)
    gamma = 1.0 + (float(t["gamma"]) - 1.0) * s
    contrast = 1.0 + (float(t["contrast"]) - 1.0) * s
    black = float(t["black"]) * s
    y2 = tone_curve(y, black, gamma, contrast, 0.0)
    scale = np.where(y > 1e-5, y2 / np.maximum(y, 1e-5), 1.0)
    out = np.clip(src * scale[..., None], 0.0, 1.0)
    mid_w = (1.0 - np.clip(1.0 - y2 / 0.42, 0.0, 1.0)) * (1.0 - np.clip((y2 - 0.62) / 0.38, 0.0, 1.0))
    out = np.clip(out + mid_w[..., None] * _as_arr3(t["mid_tint"]) * s, 0.0, 1.0)
    h, sv, v = rgb_to_hsv(out)
    dh = np.full(h.shape, float(t["hue"]) * s, dtype=np.float32)
    sat_gate = _chroma_gate(sv)
    for name, center in HUE_BANDS:
        amount = float(t.get(f"hue_{name}") or 0.0) * s
        if abs(amount) < 1e-6:
            continue
        dh = dh + hue_weight(h, center, HUE_BAND_WIDTH) * sat_gate * amount
    out = hsv_to_rgb(h + dh, sv, v)
    y3 = _luma(out)
    sat_scale = 1.0 + float(t["sat"]) * s
    out = y3[..., None] + (out - y3[..., None]) * sat_scale
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def analyze_grade_space(rgb: np.ndarray, room: Optional[np.ndarray] = None) -> dict:
    """Does this photo have enough room for a 3D LUT, or should we use curves?"""
    y, sat, chroma = _source_stats(rgb)
    if room is None:
        room = estimate_adjust_room(rgb)
    if rgb.ndim != 3 or min(int(rgb.shape[0]), int(rgb.shape[1])) < 16:
        return {
            "mode": "lut",
            "score": 0.7,
            "reason": "可调整空间足够，套 LUT",
            "room_mean": float(room.mean()) if hasattr(room, "mean") else 0.7,
            "sky_frac": 0.0,
            "hair_frac": 0.0,
        }
    short = min(int(rgb.shape[0]), int(rgb.shape[1]))
    k_var = _odd_k(round(short * 0.05), 17, 81)
    y32 = y.astype(np.float32)
    mu = cv2.blur(y32, (k_var, k_var))
    var = np.maximum(cv2.blur(y32 * y32, (k_var, k_var)) - mu * mu, 0.0)
    smooth = _smoothstep(1.0 - var / 0.0055)
    sky = smooth * np.maximum(
        _smoothstep((y32 - 0.40) / 0.22),
        _smoothstep((y32 - 0.44) / 0.24) * _smoothstep(1.0 - chroma / 0.30),
    )
    hair = _smoothstep((0.20 - y32) / 0.13) * _smoothstep(1.0 - np.minimum(chroma, sat) / 0.16)
    sky_frac = float((sky > 0.45).mean())
    hair_frac = float((hair > 0.45).mean())
    room_mean = float(room.mean())
    score = room_mean * (1.0 - 0.58 * sky_frac) * (1.0 - 0.18 * hair_frac)
    if sky_frac > 0.30 or score < 0.40:
        mode = "tune"
        reason = "平滑高光/天空占比高，套 LUT 容易出马赛克，改为色相和曲线"
    else:
        mode = "lut"
        reason = "可调整空间足够，套 LUT，对比色和邻近色会减轻"
    return {
        "mode": mode,
        "score": float(score),
        "reason": reason,
        "room_mean": room_mean,
        "sky_frac": sky_frac,
        "hair_frac": hair_frac,
    }


def default_params() -> dict:
    return {
        "black": 0.0,
        "gamma": 1.0,
        "contrast": 1.0,
        "sparkle": 0.0,
        "sh_tint": [0.0, 0.0, 0.0],
        "hi_tint": [0.0, 0.0, 0.0],
        "mid_tint": [0.0, 0.0, 0.0],
        "hue_moves": [],
        "sat_moves": [],
        "val_moves": [],
        "gray_protect": 0.12,
    }


def clone_params(params: Optional[dict]) -> dict:
    return copy.deepcopy(params or default_params())


def _as_arr3(value: Sequence[float]) -> np.ndarray:
    arr = np.array(list(value) + [0, 0, 0], dtype=np.float32)[:3]
    return arr


def tone_curve(y: np.ndarray, black: float, gamma: float, contrast: float, sparkle: float) -> np.ndarray:
    x = (y - black) / max(1.0 - black, 1e-6)
    x = np.clip(x, 0.0, 1.0) ** max(gamma, 0.05)
    x = np.clip((x - 0.48) * contrast + 0.50, 0.0, 1.0)
    if sparkle:
        hi = np.clip((x - 0.78) / 0.22, 0.0, 1.0)
        x = x + hi * sparkle
    return np.clip(x, 0.0, 1.0)


def apply_params(rgb: np.ndarray, params: dict, protect_wash: bool = True) -> np.ndarray:
    """Pure color transform. Spatial room is applied later on real photos only."""
    src = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    p = {**default_params(), **(params or {})}

    y = _luma(src)
    y2 = tone_curve(y, float(p["black"]), float(p["gamma"]), float(p["contrast"]), float(p["sparkle"]))
    scale = np.where(y > 1e-5, y2 / np.maximum(y, 1e-5), 1.0)
    graded = np.clip(src * scale[..., None], 0.0, 1.0)
    sh_w = np.clip(1.0 - y2 / 0.42, 0.0, 1.0)
    hi_w = np.clip((y2 - 0.62) / 0.38, 0.0, 1.0)
    mid_w = (1.0 - sh_w) * (1.0 - hi_w)
    tint_ok = _smoothstep((y - 0.05) / 0.10)
    sh_w = (sh_w * tint_ok)[..., None]
    hi_w = hi_w[..., None]
    mid_w = mid_w[..., None]
    graded = np.clip(
        graded + sh_w * _as_arr3(p["sh_tint"]) + hi_w * _as_arr3(p["hi_tint"]) + mid_w * _as_arr3(p["mid_tint"]),
        0.0,
        1.0,
    )

    h, s, v = rgb_to_hsv(graded)
    y0, s0, _c0 = _source_stats(src)
    gate = _stable_color_gate(y0, s0) if protect_wash else np.ones_like(y0, dtype=np.float32)
    dh_g = np.zeros_like(h)
    ds = np.zeros_like(s)
    dv = np.zeros_like(v)
    for center, width, amount in p.get("hue_moves") or []:
        wt = hue_weight(h, float(center), float(width)) * gate
        dh_g += wt * float(amount)
    for center, width, amount in p.get("sat_moves") or []:
        wt = hue_weight(h, float(center), float(width)) * gate
        ds += wt * float(amount)
    for center, width, amount in p.get("val_moves") or []:
        wt = hue_weight(h, float(center), float(width)) * gate
        dv += wt * float(amount)
    graded = hsv_to_rgb(h + dh_g, s, np.clip(v * (1.0 + dv), 0.0, 1.0))
    y3 = _luma(graded)
    sat_scale = np.clip(1.0 + ds, 0.0, 1.85)[..., None]
    graded = y3[..., None] + (graded - y3[..., None]) * sat_scale

    protect = float(p.get("gray_protect") or 0.0)
    if protect > 1e-6:
        _h2, s2, v2 = rgb_to_hsv(graded)
        gray = _smoothstep(1.0 - s2 / 0.18)
        w = protect * gray
        graded = graded * (1.0 - w[..., None]) + v2[..., None] * w[..., None]

    return np.clip(graded, 0.0, 1.0).astype(np.float32)


def identity_hald(cube: int = HALD_CUBE) -> np.ndarray:
    side = int(round((cube**3) ** 0.5))
    idx = np.arange(side * side, dtype=np.int32).reshape(side, side)
    r = idx % cube
    g = (idx // cube) % cube
    b = idx // (cube * cube)
    return np.stack([r, g, b], axis=-1).astype(np.float32) / float(cube - 1)


def identity_cube_points(size: int = CUBE_SIZE) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, size, dtype=np.float32)
    b, g, r = np.meshgrid(xs, xs, xs, indexing="ij")
    return np.stack([r, g, b], axis=-1).reshape(-1, 3)


def build_hald(params: dict) -> np.ndarray:
    return apply_params(identity_hald(), params)


def build_cube_rgb(params: dict, size: int = CUBE_SIZE) -> np.ndarray:
    pts = identity_cube_points(size)
    graded = apply_params(pts.reshape(-1, 1, 3), params).reshape(-1, 3)
    return graded.reshape(size, size, size, 3)


def cube_points_from_volume(volume: np.ndarray) -> np.ndarray:
    return volume.reshape(-1, 3)


def save_hald_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = (np.clip(rgb[..., ::-1], 0, 1) * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(str(path), bgr)


def save_cube_file(path: Path, volume: np.ndarray, title: str) -> None:
    size = int(volume.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'TITLE "{title}"',
        f"LUT_3D_SIZE {size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
        "",
    ]
    for r, g, b in volume.reshape(-1, 3):
        lines.append(f"{float(r):.6f} {float(g):.6f} {float(b):.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_cube_file(path: Path) -> np.ndarray:
    size = None
    values: List[List[float]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("LUT_3D_SIZE"):
            size = int(line.split()[-1])
            continue
        if line.upper().startswith("TITLE") or line.upper().startswith("DOMAIN"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 3:
            values.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not size:
        n = len(values)
        size = int(round(n ** (1.0 / 3.0)))
    arr = np.array(values, dtype=np.float32)
    return arr.reshape(size, size, size, 3)


def apply_look(rgb: np.ndarray, params: dict, strength: float = 1.0, protect_wash: bool = True) -> np.ndarray:
    """Apply the parametric look in float, without sampling a 3D lattice."""
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    strength = float(np.clip(strength, 0.0, 1.5))
    if strength <= 1e-6:
        return rgb
    graded = apply_params(rgb, params, protect_wash=protect_wash)
    if 0.999 <= strength <= 1.001:
        return graded
    return np.clip(rgb * (1.0 - strength) + graded * strength, 0.0, 1.0).astype(np.float32)


def apply_cube(rgb: np.ndarray, cube: np.ndarray, strength: float = 1.0) -> np.ndarray:
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    strength = float(np.clip(strength, 0.0, 1.5))
    if strength <= 1e-6:
        return rgb
    n = int(cube.shape[0])
    h, w, _ = rgb.shape
    out = np.empty_like(rgb)
    step = 96
    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)
        sl = rgb[y0:y1]
        x = sl * (n - 1)
        i0 = np.floor(x).astype(np.int32)
        np.clip(i0, 0, n - 1, out=i0)
        i1 = np.minimum(i0 + 1, n - 1)
        f = x - i0
        r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
        r1, g1, b1 = i1[..., 0], i1[..., 1], i1[..., 2]
        c000 = cube[b0, g0, r0]
        c001 = cube[b0, g0, r1]
        c010 = cube[b0, g1, r0]
        c011 = cube[b0, g1, r1]
        c100 = cube[b1, g0, r0]
        c101 = cube[b1, g0, r1]
        c110 = cube[b1, g1, r0]
        c111 = cube[b1, g1, r1]
        fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
        # Tetrahedral interpolation (Resolve / Nuke), smoother than trilinear on small cubes.
        mapped = np.where(
            fr > fg,
            np.where(
                fg > fb,
                c000 + fr * (c001 - c000) + fg * (c011 - c001) + fb * (c111 - c011),
                np.where(
                    fr > fb,
                    c000 + fr * (c001 - c000) + fb * (c101 - c001) + fg * (c111 - c101),
                    c000 + fb * (c100 - c000) + fr * (c101 - c100) + fg * (c111 - c101),
                ),
            ),
            np.where(
                fr > fb,
                c000 + fg * (c010 - c000) + fr * (c011 - c010) + fb * (c111 - c011),
                np.where(
                    fg > fb,
                    c000 + fg * (c010 - c000) + fb * (c110 - c010) + fr * (c111 - c110),
                    c000 + fb * (c100 - c000) + fg * (c110 - c100) + fr * (c111 - c110),
                ),
            ),
        )
        if 0.999 <= strength <= 1.001:
            out[y0:y1] = mapped
        else:
            out[y0:y1] = sl * (1.0 - strength) + mapped * strength
    return np.clip(out, 0.0, 1.0)


def list_images(folder: Path, recursive: bool = False) -> List[Path]:
    import os

    folder = Path(folder).expanduser()
    found: List[Path] = []
    if recursive:
        for root, dirs, names in os.walk(folder, followlinks=False):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in names:
                if is_image_name(name) and not name.startswith("."):
                    found.append(Path(root) / name)
    else:
        for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
            if path.is_file() and is_image_name(path.name) and not path.name.startswith("."):
                found.append(path)
    found.sort(key=lambda p: p.as_posix().lower())
    return found


def load_rgb(path: Path, max_side: Optional[int] = None, meta: Optional[dict] = None) -> np.ndarray:
    with Image.open(path) as im:
        orig_w, orig_h = im.size
        if im.getexif().get(0x0112) in (5, 6, 7, 8):
            orig_w, orig_h = orig_h, orig_w
        if max_side:
            im.draft(None, (max_side, max_side))
        im = ImageOps.exif_transpose(im) or im
        icc = im.info.get("icc_profile")
        dpi = im.info.get("dpi")
        src_format = (im.format or "").upper()
        if im.mode not in {"RGB", "RGBA", "I;16", "I;16B", "I"}:
            im = im.convert("RGB")
        elif im.mode == "RGBA":
            im = im.convert("RGB")
        if max_side:
            im.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
        arr = np.asarray(im)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    bits = 16 if arr.dtype in (np.uint16, np.int32) else 8
    if meta is not None:
        meta.update(
            {
                "suffix": path.suffix.lower(),
                "format": src_format,
                "orig_width": int(orig_w),
                "orig_height": int(orig_h),
                "width": int(arr.shape[1]),
                "height": int(arr.shape[0]),
                "bits": bits,
                "icc": icc,
                "dpi": dpi,
            }
        )
    if bits == 16:
        u16 = np.clip(arr, 0, 65535).astype(np.uint16)
        if meta is not None:
            meta["u16"] = u16
        return np.clip(u16.astype(np.float32) / 65535.0, 0.0, 1.0)
    u8 = arr.astype(np.uint8) if arr.dtype == np.uint8 else np.clip(arr, 0, 255).astype(np.uint8)
    if meta is not None:
        meta["u8"] = u8
    return u8.astype(np.float32) / 255.0


def _restore_locked(out: np.ndarray, src: np.ndarray, keep: Optional[np.ndarray], thresh: float = 0.08) -> np.ndarray:
    if keep is None or src is None or src.shape != out.shape:
        return out
    locked = keep < thresh
    if not np.any(locked):
        return out
    out = out.copy()
    out[locked] = src[locked]
    return out


def _quantize_u8(rgb: np.ndarray, meta: Optional[dict] = None) -> np.ndarray:
    x = np.clip(rgb, 0.0, 1.0) * 255.0
    u8 = np.clip(x + 0.5, 0, 255).astype(np.uint8)
    keep = None if not meta else meta.get("keep")
    src_u8 = None if not meta else meta.get("u8")
    return _restore_locked(u8, src_u8, keep)


def _quantize_u16(rgb: np.ndarray, meta: Optional[dict] = None) -> np.ndarray:
    u16 = np.clip(np.clip(rgb, 0.0, 1.0) * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    keep = None if not meta else meta.get("keep")
    src_u16 = None if not meta else meta.get("u16")
    return _restore_locked(u16, src_u16, keep)


def _save_kwargs(meta: Optional[dict]) -> dict:
    kwargs = {}
    if not meta:
        return kwargs
    if meta.get("icc"):
        kwargs["icc_profile"] = meta["icc"]
    if meta.get("dpi"):
        kwargs["dpi"] = meta["dpi"]
    return kwargs


def save_rgb(
    path: Path,
    rgb: np.ndarray,
    source_suffix: str = ".jpg",
    meta: Optional[dict] = None,
    preview: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = (path.suffix or (meta or {}).get("suffix") or source_suffix or ".jpg").lower()
    bits = int((meta or {}).get("bits") or 8)
    extra = _save_kwargs(meta)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if not preview and meta and meta.get("width") and (int(meta["width"]), int(meta["height"])) != (w, h):
        raise ValueError(f"导出尺寸被改变：{meta['width']}×{meta['height']} → {w}×{h}")

    if preview:
        pixels = _quantize_u8(rgb, meta)
        out = path if path.suffix.lower() in {".jpg", ".jpeg"} else path.with_suffix(".jpg")
        Image.fromarray(pixels, "RGB").save(out, "JPEG", quality=97, subsampling=0, **extra)
        return out

    if not preview and meta and meta.get("orig_width"):
        ow, oh = int(meta["orig_width"]), int(meta["orig_height"])
        if (w, h) != (ow, oh):
            raise ValueError(f"导出尺寸被改变：{ow}×{oh} → {w}×{h}")

    if suffix in {".png"}:
        out = path.with_suffix(".png") if path.suffix.lower() != ".png" else path
        if bits >= 16:
            rgb16 = _quantize_u16(rgb, meta)
            cv2.imwrite(str(out), rgb16[..., ::-1])
        else:
            Image.fromarray(_quantize_u8(rgb, meta), "RGB").save(out, "PNG", compress_level=1, **extra)
        return out

    if suffix in {".tif", ".tiff"}:
        out = path if path.suffix.lower() in {".tif", ".tiff"} else path.with_suffix(".tif")
        if bits >= 16:
            rgb16 = _quantize_u16(rgb, meta)
            cv2.imwrite(str(out), rgb16[..., ::-1])
        else:
            Image.fromarray(_quantize_u8(rgb, meta), "RGB").save(out, "TIFF", compression="tiff_adobe_deflate", **extra)
        return out

    if suffix == ".webp":
        out = path.with_suffix(".webp") if path.suffix.lower() != ".webp" else path
        Image.fromarray(_quantize_u8(rgb, meta), "RGB").save(out, "WEBP", lossless=True, quality=100, method=6, **extra)
        return out

    if suffix == ".bmp":
        out = path.with_suffix(".bmp") if path.suffix.lower() != ".bmp" else path
        Image.fromarray(_quantize_u8(rgb, meta), "RGB").save(out, "BMP")
        return out

    pixels = _quantize_u8(rgb, meta)
    out = path if path.suffix.lower() in {".jpg", ".jpeg"} else path.with_suffix(".jpg")
    Image.fromarray(pixels, "RGB").save(
        out, "JPEG", quality=98, subsampling=0, optimize=False, **extra
    )
    return out


def _median_hue(hue: np.ndarray) -> Optional[float]:
    if hue.size < 40:
        return None
    ang = hue * np.pi / 180.0
    deg = (np.degrees(np.arctan2(np.median(np.sin(ang)), np.median(np.cos(ang)))) % 360)
    return float(deg)


def analyze_refs(paths: Sequence[Path], max_side: int = 420, max_files: int = 80) -> dict:
    files = [Path(p) for p in paths][:max_files]
    if not files:
        raise ValueError("参考文件夹里没有图片")
    samples = []
    for path in files:
        try:
            samples.append(load_rgb(path, max_side=max_side).reshape(-1, 3))
        except Exception:
            continue
    if not samples:
        raise ValueError("参考图无法读取")
    pool = np.concatenate(samples, 0)
    # subsample for speed
    if pool.shape[0] > 400_000:
        rng = np.random.default_rng(0)
        pool = pool[rng.choice(pool.shape[0], 400_000, replace=False)]
    y = 0.2126 * pool[:, 0] + 0.7152 * pool[:, 1] + 0.0722 * pool[:, 2]
    h, s, v = rgb_to_hsv(pool.reshape(-1, 1, 3))
    h, s, v = h.reshape(-1), s.reshape(-1), v.reshape(-1)
    colorful = s > 0.16

    def band_rgb(lo: float, hi: float) -> List[float]:
        mask = (y >= lo) & (y < hi)
        if mask.sum() < 80:
            return [0.0, 0.0, 0.0]
        mean = pool[mask].mean(0)
        luma = float(0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2])
        chroma = (mean - luma).tolist()
        return [float(c) for c in chroma]

    def cluster(h0: float, h1: float) -> Tuple[Optional[float], float, float, List[int]]:
        mask = colorful & (h >= h0) & (h < h1)
        if mask.mean() < 0.02:
            return None, 0.0, 0.0, [128, 128, 128]
        hue = _median_hue(h[mask])
        sat = float(np.median(s[mask]))
        val = float(np.median(v[mask]))
        mean = pool[mask].mean(0)
        swatch = [int(np.clip(c * 255, 0, 255)) for c in mean]
        return hue, sat, val, swatch

    green_h, green_s, green_v, green_sw = cluster(60, 150)
    aqua_h, aqua_s, aqua_v, aqua_sw = cluster(160, 200)
    blue_h, blue_s, blue_v, blue_sw = cluster(190, 250)

    p10, p50, p90 = [float(x) for x in np.percentile(y, [10, 50, 90])]
    hi_mean = pool[y >= 0.82].mean(0) if (y >= 0.82).sum() > 40 else np.array([0.9, 0.9, 0.9])
    swatches = [green_sw, aqua_sw, blue_sw]
    if (y >= 0.82).sum() > 40:
        swatches.append([int(np.clip(c * 255, 0, 255)) for c in hi_mean])
    sh_mean = pool[y < 0.22].mean(0) if (y < 0.22).sum() > 40 else np.array([0.1, 0.15, 0.1])
    swatches.insert(0, [int(np.clip(c * 255, 0, 255)) for c in sh_mean])

    return {
        "count": len(files),
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "median_sat": float(np.median(s[colorful])) if colorful.any() else 0.4,
        "green_h": green_h,
        "green_s": green_s,
        "green_v": green_v,
        "aqua_h": aqua_h,
        "aqua_s": aqua_s,
        "blue_h": blue_h,
        "blue_s": blue_s,
        "blue_v": blue_v,
        "sh_chroma": band_rgb(0.0, 0.22),
        "mid_chroma": band_rgb(0.35, 0.65),
        "hi_chroma": band_rgb(0.75, 1.01),
        "swatches": swatches,
    }


def params_from_stats(stats: dict) -> dict:
    p = default_params()
    p10, p50, p90 = stats["p10"], stats["p50"], stats["p90"]
    med_sat = float(stats.get("median_sat") or 0.4)
    p["black"] = float(np.clip((p10 - 0.10) * 0.55, 0.0, 0.06))
    raw_gamma = float(np.clip(np.log(max(0.12, p50)) / np.log(max(0.12, 0.48)), 0.78, 1.18))
    spread = p90 - p10
    p["contrast"] = float(np.clip(0.92 + (spread - 0.48) * 0.45, 0.92, 1.18))
    p["sparkle"] = float(np.clip((p90 - 0.72) * 0.12, 0.0, 0.05))
    if med_sat < 0.32:
        # Low-sat refs are often brighter stills; don't lift the target photo to match.
        p["gamma"] = float(np.clip(max(raw_gamma, 1.0) + 0.08, 1.06, 1.16))
        p["black"] = float(min(p["black"], 0.015))
        p["sparkle"] = float(min(p["sparkle"], 0.004))
        p["contrast"] = float(np.clip(p["contrast"], 0.96, 1.05))
    else:
        p["gamma"] = raw_gamma
    p["sh_tint"] = [float(np.clip(c * 0.55, -0.08, 0.08)) for c in stats["sh_chroma"]]
    p["mid_tint"] = [float(np.clip(c * 0.38, -0.07, 0.07)) for c in stats["mid_chroma"]]
    p["hi_tint"] = [float(np.clip(c * 0.28, -0.05, 0.05)) for c in stats["hi_chroma"]]

    hue_moves = []
    sat_moves = []
    val_moves = []
    global_sat = float(np.clip(med_sat / 0.40 - 1.0, -0.36, 0.36))
    if abs(global_sat) >= 0.05:
        sat_moves.append([180.0, 180.0, global_sat])
        if global_sat < 0:
            val_moves.append([180.0, 180.0, float(np.clip(global_sat * 0.18, -0.08, -0.02))])
    if stats.get("green_h") is not None:
        shift = float(np.clip(stats["green_h"] - 118.0, -36, 18))
        hue_moves.append([118.0, 48.0, shift])
        residual = float(np.clip((float(stats["green_s"]) - med_sat) / 0.35, -0.14, 0.20))
        if global_sat < -0.08:
            residual = float(min(residual, 0.04))
        sat_moves.append([stats["green_h"], 46.0, residual])
        val_moves.append([stats["green_h"], 40.0, float(np.clip((stats["green_v"] - 0.48) * 0.22, -0.08, 0.14))])
    blue_h = stats.get("blue_h") or stats.get("aqua_h")
    if blue_h is not None:
        shift = float(np.clip(blue_h - 214.0, -36, 12))
        hue_moves.append([214.0, 52.0, shift])
        sat = stats.get("blue_s") or stats.get("aqua_s") or med_sat
        residual = float(np.clip((float(sat) - med_sat) / 0.35, -0.12, 0.20))
        if global_sat < -0.08:
            residual = float(min(residual, 0.04))
        sat_moves.append([blue_h, 50.0, residual])
        val_moves.append([blue_h, 46.0, float(np.clip(((stats.get("blue_v") or 0.7) - 0.62) * 0.2, -0.06, 0.12))])
    p["hue_moves"] = hue_moves
    p["sat_moves"] = sat_moves
    p["val_moves"] = val_moves
    p["gray_protect"] = 0.12
    return p


def _add_move(moves: list, center: float, width: float, amount: float) -> None:
    moves.append([center, width, amount])


def _tint_add(vec: list, rgb: Sequence[float], scale: float) -> None:
    for i in range(3):
        vec[i] = float(np.clip(vec[i] + float(rgb[i]) * scale, -0.14, 0.14))


def _desc_scale(text: str, start: int, end: int = None) -> float:
    prev = text[max(0, start - 12) : start]
    after = text[(end if end is not None else start) : (end if end is not None else start) + 8]
    if re.search(r"不要太|别太|不要|别|勿|少|降低|减弱|减少|压低|偏低", prev):
        return -1.0
    if re.search(r"压低|降低|偏低", after):
        return -1.0
    if re.search(r"稍微|略微|一点|一些|稍", prev):
        return 0.6
    if re.search(r"很|非常|明显|大幅|强烈", prev):
        return 1.55
    if "更" in prev:
        return 1.25
    return 1.0


def apply_description(params: dict, text: str) -> Tuple[dict, List[str]]:
    """Stack written look notes on top of reference-derived params."""
    p = clone_params(params)
    notes: List[str] = []
    raw = (text or "").strip()
    if not raw:
        return p, notes

    def apply_at(index: int, label: str, fn, end: int = None) -> None:
        scale = _desc_scale(raw, index, end)
        fn(p, scale)
        if scale < 0:
            flipped = label.replace("提高", "降低").replace("加强", "减弱").replace("提亮", "压暗").replace("更亮", "压低")
            notes.append(flipped if flipped != label else f"减弱{label}")
        else:
            notes.append(label)

    rules = [
        (r"黄绿|春绿|嫩绿|草绿|lime", "叶子偏黄绿", lambda q, s: (
            _add_move(q["hue_moves"], 120, 46, -20 * s),
            _add_move(q["sat_moves"], 96, 44, 0.22 * abs(s) * (1 if s > 0 else -1)),
            _add_move(q["val_moves"], 96, 40, 0.10 * s),
        )),
        (r"青绿|更青|青湖|tiffany|蒂芙尼|teal|cyan", "水色推向青", lambda q, s: (
            _add_move(q["hue_moves"], 214, 52, -20 * s),
            _add_move(q["sat_moves"], 194, 48, 0.22 * s),
            _tint_add(q["mid_tint"], (-0.035, 0.02, 0.04), s),
        )),
        (r"祖母绿|墨绿", "绿色加深", lambda q, s: (
            _add_move(q["hue_moves"], 130, 40, 8 * s),
            _add_move(q["sat_moves"], 130, 40, 0.16 * s),
            _add_move(q["val_moves"], 130, 36, -0.06 * s),
        )),
        (r"日落|橙|橘", "高光偏暖橙", lambda q, s: (
            _tint_add(q["hi_tint"], (0.05, 0.012, -0.03), s),
            _add_move(q["hue_moves"], 35, 28, -6 * s),
        )),
        (r"金色|暖黄", "高光偏金", lambda q, s: _tint_add(q["hi_tint"], (0.045, 0.02, -0.025), s)),
        (r"冷色|偏冷|cool", "整体偏冷", lambda q, s: (
            _tint_add(q["mid_tint"], (-0.03, 0.008, 0.04), s),
            _tint_add(q["hi_tint"], (-0.016, 0.004, 0.02), s),
        )),
        (r"暖色|偏暖|温暖|warm", "整体偏暖", lambda q, s: (
            _tint_add(q["mid_tint"], (0.04, 0.008, -0.03), s),
            _tint_add(q["hi_tint"], (0.03, 0.01, -0.02), s),
        )),
        (r"高对比|对比高|对比更强|硬朗", "提高对比", lambda q, s: q.__setitem__("contrast", float(np.clip(q["contrast"] + 0.14 * s, 0.8, 1.35)))),
        (r"低对比|柔和|软调|淡对比", "降低对比", lambda q, s: q.__setitem__("contrast", float(np.clip(q["contrast"] - 0.12 * abs(s), 0.8, 1.35)))),
        (r"通透|清新|提亮|更亮|明亮|高调|airy", "提亮通透", lambda q, s: (
            q.__setitem__("gamma", float(np.clip(q["gamma"] - 0.10 * s, 0.72, 1.28))),
            q.__setitem__("sparkle", float(np.clip(q["sparkle"] + 0.02 * max(s, 0), 0.0, 0.09))),
        )),
        (r"压暗|更暗|低调|沉一点|沉一点", "压暗中灰", lambda q, s: q.__setitem__("gamma", float(np.clip(q["gamma"] + 0.10 * s, 0.72, 1.28)))),
        (r"低饱和|饱和度压低|压低饱和|清淡|淡色|雾感|发灰|去饱和", "降低饱和", lambda q, s: (
            _add_move(q["sat_moves"], 180, 180, -0.22 * abs(s)),
            q.__setitem__("gamma", float(np.clip(max(q["gamma"], 1.02), 0.92, 1.22))),
            q.__setitem__("sparkle", float(np.clip(q["sparkle"] - 0.01 * abs(s), 0.0, 0.1))),
        )),
        (r"中性", "中性降彩", lambda q, s: (
            _add_move(q["sat_moves"], 180, 180, -0.06 * abs(s)),
            q.__setitem__("gamma", float(np.clip(max(q["gamma"], 1.0), 0.92, 1.2))),
            q.__setitem__("mid_tint", [float(c) * (1.0 - 0.35 * abs(s)) for c in q["mid_tint"]]),
            q.__setitem__("hi_tint", [float(c) * (1.0 - 0.25 * abs(s)) for c in q["hi_tint"]]),
        )),
        (r"(?<![低去])饱和|鲜艳|浓郁|更艳|浓一点", "提高饱和", lambda q, s: (
            _add_move(q["sat_moves"], 96, 80, 0.18 * s),
            _add_move(q["sat_moves"], 200, 80, 0.18 * s),
        )),
        (r"高光|闪光|sparkle|亮斑", "高光更亮", lambda q, s: q.__setitem__("sparkle", float(np.clip(q["sparkle"] + 0.045 * s, 0.0, 0.1)))),
        (r"阴影青|暗部青|阴影绿|暗部绿", "暗部偏青绿", lambda q, s: _tint_add(q["sh_tint"], (-0.02, 0.04, 0.03), s)),
        (r"胶片|复古|褪色|film|日系", "胶片褪色", lambda q, s: (
            q.__setitem__("black", float(np.clip(q["black"] + 0.03 * s, 0.0, 0.1))),
            q.__setitem__("contrast", float(np.clip(q["contrast"] - 0.05 * s, 0.82, 1.3))),
            _add_move(q["sat_moves"], 180, 160, -0.08 * s),
        )),
        (r"莫兰迪", "莫兰迪低饱和", lambda q, s: (
            _add_move(q["sat_moves"], 180, 170, -0.18 * abs(s)),
            q.__setitem__("black", float(np.clip(q["black"] + 0.02 * abs(s), 0.0, 0.1))),
        )),
        (r"紫|品红|洋红", "品色加强", lambda q, s: (
            _add_move(q["sat_moves"], 300, 36, 0.16 * s),
            _tint_add(q["mid_tint"], (0.03, -0.02, 0.03), s),
        )),
        (r"粉", "粉色倾向", lambda q, s: _tint_add(q["mid_tint"], (0.035, -0.01, 0.02), s)),
        (r"黄(?!绿)", "黄色倾向", lambda q, s: (
            _add_move(q["sat_moves"], 55, 28, 0.14 * s),
            _tint_add(q["mid_tint"], (0.03, 0.02, -0.02), s),
        )),
    ]
    used_spans = []
    for pattern, label, fn in rules:
        for match in re.finditer(pattern, raw, flags=re.IGNORECASE):
            span = match.span()
            if any(span[0] < b and span[1] > a for a, b in used_spans):
                continue
            used_spans.append(span)
            apply_at(match.start(), label, fn, match.end())

    # leftover single-color words not already consumed
    leftovers = [
        (r"青", "水色推向青", lambda q, s: (
            _add_move(q["hue_moves"], 214, 52, -16 * s),
            _tint_add(q["mid_tint"], (-0.03, 0.016, 0.032), s),
        )),
        (r"蓝", "偏向蓝色", lambda q, s: _tint_add(q["mid_tint"], (-0.02, 0.004, 0.03), s)),
        (r"绿", "偏向绿色", lambda q, s: _tint_add(q["mid_tint"], (-0.02, 0.02, -0.008), s)),
        (r"冷", "整体偏冷", lambda q, s: _tint_add(q["mid_tint"], (-0.028, 0.006, 0.036), s)),
        (r"暖", "整体偏暖", lambda q, s: _tint_add(q["mid_tint"], (0.036, 0.006, -0.028), s)),
    ]
    for pattern, label, fn in leftovers:
        for match in re.finditer(pattern, raw):
            span = match.span()
            if any(span[0] < b and span[1] > a for a, b in used_spans):
                continue
            used_spans.append(span)
            apply_at(match.start(), label, fn, match.end())

    # keep unique notes, preserve order
    seen = set()
    uniq = []
    for note in notes:
        if note in seen:
            continue
        seen.add(note)
        uniq.append(note)
    if raw and not uniq:
        uniq.append("描述里没识别到影调词，本次只按参考图生成")
    return p, uniq


def summarize_params(
    params: dict,
    stats: Optional[dict] = None,
    notes: Optional[Iterable[str]] = None,
    from_refs: bool = False,
    from_desc: bool = False,
) -> str:
    bits = []
    if from_refs and from_desc:
        bits.append("参考图+描述")
    elif from_refs:
        bits.append("只按参考图")
    elif from_desc:
        bits.append("只按描述")
    if stats:
        bits.append(f"参考 {stats.get('count') or 0} 张")
        med = stats.get("median_sat")
        if med is not None and med < 0.30:
            bits.append("参考图低饱和")
        elif med is not None and med > 0.50:
            bits.append("参考图偏饱和")
        if stats.get("green_h") is not None:
            bits.append(f"草绿约 {stats['green_h']:.0f}°")
        hue = stats.get("blue_h") or stats.get("aqua_h")
        if hue is not None:
            bits.append(f"水色约 {hue:.0f}°")
    p = params or {}
    if p.get("gamma", 1) < 0.95:
        bits.append("偏亮")
    elif 0.98 <= float(p.get("gamma", 1)) <= 1.06:
        bits.append("明度持平")
    if p.get("gamma", 1) > 1.06:
        bits.append("偏暗")
    if p.get("contrast", 1) > 1.06:
        bits.append("对比偏高")
    if notes:
        bits.extend([n for n in notes if n and "没识别到" not in n])
    return " · ".join(bits) or "中性影调微调"


def generate_look(
    ref_paths: Sequence[Path],
    description: str = "",
    base_params: Optional[dict] = None,
) -> dict:
    notes: List[str] = []
    stats = None
    if ref_paths:
        stats = analyze_refs(ref_paths)
        params = params_from_stats(stats)
        base = clone_params(params)
    elif base_params:
        params = clone_params(base_params)
        base = clone_params(base_params)
    else:
        params = default_params()
        base = default_params()
    params, desc_notes = apply_description(params, description)
    if stats and (stats.get("median_sat") or 0.4) < 0.30:
        notes.append("按参考图压低整体饱和")
    notes.extend(desc_notes)
    from_refs = bool(ref_paths) or bool(base_params)
    from_desc = bool((description or "").strip()) and not any("没识别到" in n for n in desc_notes)
    swatches = (stats or {}).get("swatches") or [
        [32, 48, 36],
        [80, 124, 54],
        [95, 176, 189],
        [93, 169, 213],
        [229, 237, 229],
    ]
    return {
        "params": params,
        "base_params": base,
        "stats": stats,
        "notes": notes,
        "swatches": swatches,
        "from_refs": from_refs,
        "from_desc": from_desc,
        "summary": summarize_params(params, stats, notes, from_refs, from_desc),
    }
