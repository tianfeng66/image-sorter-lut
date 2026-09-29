"""Measure exposure extremes, neon pure color, and main-hue harmony."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from image_io import load_preview_bgr


def _num(value: float, digits: int) -> float:
    return float(round(float(value), digits))

HARMONY_NAMES = {
    "mono": "同色系",
    "adjacent": "相邻色",
    "analogous": "邻近色",
    "contrast": "对比色",
    "complementary": "互补色",
    "none": "无标准搭配",
}


@dataclass
class ColorSwatch:
    h: float
    s: float
    v: float
    hex: str
    weight: float


@dataclass
class ColorResult:
    width: int
    height: int
    extreme_ratio: float
    bright_ratio: float
    dark_ratio: float
    white_ratio: float
    black_ratio: float
    over_ratio: float
    under_ratio: float
    neon_ratio: float
    harmony: str
    harmony_name: str
    harmony_detail: str
    colors: List[ColorSwatch] = field(default_factory=list)

    def as_public(self) -> dict:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "extreme_ratio": _num(self.extreme_ratio, 6),
            "bright_ratio": _num(self.bright_ratio, 6),
            "dark_ratio": _num(self.dark_ratio, 6),
            "white_ratio": _num(self.white_ratio, 6),
            "black_ratio": _num(self.black_ratio, 6),
            "over_ratio": _num(self.over_ratio, 6),
            "under_ratio": _num(self.under_ratio, 6),
            "neon_ratio": _num(self.neon_ratio, 6),
            "harmony": self.harmony,
            "harmony_name": self.harmony_name,
            "harmony_detail": self.harmony_detail,
            "colors": [
                {
                    "h": _num(c.h, 1),
                    "s": _num(c.s, 3),
                    "v": _num(c.v, 3),
                    "hex": c.hex,
                    "weight": _num(c.weight, 4),
                }
                for c in self.colors
            ],
        }


def analyze_color(path: str, max_side: int = 480) -> ColorResult:
    image, width, height = load_preview_bgr(path, max_side)
    if image is None:
        return ColorResult(
            width=width,
            height=height,
            extreme_ratio=0,
            bright_ratio=0,
            dark_ratio=0,
            white_ratio=0,
            black_ratio=0,
            over_ratio=0,
            under_ratio=0,
            neon_ratio=0,
            harmony="none",
            harmony_name=HARMONY_NAMES["none"],
            harmony_detail="无法读取图片",
        )
    work = image
    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32) * 2.0
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    maps = _exposure_maps(rgb, luma)
    white = maps["inherent_white"]
    black = maps["inherent_black"]
    over = maps["over"]
    under = maps["under"]
    bright = over
    dark = under
    extreme = bright | dark
    neon = (sat >= 0.82) & (val >= 0.82)

    colors, harmony, detail = _harmony(hue, sat, val)
    return ColorResult(
        width=width,
        height=height,
        extreme_ratio=float(extreme.mean()),
        bright_ratio=float(bright.mean()),
        dark_ratio=float(dark.mean()),
        white_ratio=float(white.mean()),
        black_ratio=float(black.mean()),
        over_ratio=float(over.mean()),
        under_ratio=float(under.mean()),
        neon_ratio=float(neon.mean()),
        harmony=harmony,
        harmony_name=HARMONY_NAMES[harmony],
        harmony_detail=detail,
        colors=colors,
    )


def _local_std(image: np.ndarray, ksize: int) -> np.ndarray:
    k = max(3, int(ksize) | 1)
    mean = cv2.GaussianBlur(image, (k, k), 0)
    mean2 = cv2.GaussianBlur(image * image, (k, k), 0)
    return np.sqrt(np.clip(mean2 - mean * mean, 0, None))


def _exposure_maps(rgb: np.ndarray, luma: np.ndarray) -> dict:
    """Count clipped/crushed pixels only. Textured white/black objects stay out."""
    micro = _local_std(luma, 5)
    meso = _local_std(luma, 21)
    has_detail = (micro > 0.012) | (meso > 0.018)
    channel_max = rgb.max(axis=2)
    channel_min = rgb.min(axis=2)
    over = (luma >= 0.975) & (channel_max >= 0.98) & ~has_detail
    under = (luma <= 0.025) & (channel_max <= 0.04) & ~has_detail
    inherent_white = (luma >= 0.82) & (channel_min >= 0.72) & has_detail & ~over
    inherent_black = (luma <= 0.18) & has_detail & ~under
    return {
        "over": over,
        "under": under,
        "inherent_white": inherent_white,
        "inherent_black": inherent_black,
    }


def _harmony(hue: np.ndarray, sat: np.ndarray, val: np.ndarray) -> Tuple[List[ColorSwatch], str, str]:
    chromatic = (sat >= 0.18) & (val >= 0.16)
    if float(chromatic.mean()) < 0.03:
        return [], "mono", "几乎没有彩色，视为同色系（中性灰/单色）"

    ys, xs = np.where(chromatic)
    if len(ys) > 2500:
        pick = np.linspace(0, len(ys) - 1, 2500).astype(int)
        ys, xs = ys[pick], xs[pick]
    h = hue[ys, xs]
    s = sat[ys, xs]
    v = val[ys, xs]
    feat = np.stack(
        [np.cos(np.deg2rad(h)), np.sin(np.deg2rad(h)), s, v],
        axis=1,
    ).astype(np.float32)

    k = 4 if len(feat) >= 40 else 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _ret, labels, centers = cv2.kmeans(feat, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    swatches = []
    for i, center in enumerate(centers):
        weight = float((labels == i).mean())
        if weight < 0.06:
            continue
        hh = (np.degrees(np.arctan2(center[1], center[0])) + 360.0) % 360.0
        ss = float(np.clip(center[2], 0, 1))
        vv = float(np.clip(center[3], 0, 1))
        swatches.append(ColorSwatch(h=hh, s=ss, v=vv, hex=_hsv_hex(hh, ss, vv), weight=weight))
    swatches.sort(key=lambda c: c.weight, reverse=True)
    swatches = _merge_close_hues(swatches)
    if not swatches:
        return [], "mono", "彩色占比过碎，视为同色系"
    if len(swatches) == 1:
        return swatches, "mono", f"主色 {swatches[0].hex}，同色系"

    kind, detail = _classify_pairs(swatches)
    return swatches[:5], kind, detail


def _merge_close_hues(swatches: List[ColorSwatch], limit: float = 12.0) -> List[ColorSwatch]:
    merged: List[ColorSwatch] = []
    for sw in swatches:
        hit: Optional[ColorSwatch] = None
        for other in merged:
            if _hue_dist(sw.h, other.h) <= limit:
                hit = other
                break
        if hit is None:
            merged.append(sw)
            continue
        total = hit.weight + sw.weight
        hit.h = _circular_mean([(hit.h, hit.weight), (sw.h, sw.weight)])
        hit.s = (hit.s * hit.weight + sw.s * sw.weight) / total
        hit.v = (hit.v * hit.weight + sw.v * sw.weight) / total
        hit.weight = total
        hit.hex = _hsv_hex(hit.h, hit.s, hit.v)
    merged.sort(key=lambda c: c.weight, reverse=True)
    return merged


def _classify_pairs(swatches: List[ColorSwatch]) -> Tuple[str, str]:
    # Prefer a named relation on the two strongest chromatic colors.
    ranked = [
        ("complementary", 150, 180, "互补色"),
        ("contrast", 90, 150, "对比色"),
        ("analogous", 40, 70, "邻近色"),
        ("adjacent", 15, 40, "相邻色"),
        ("mono", 0, 15, "同色系"),
    ]
    best = None
    for i in range(len(swatches)):
        for j in range(i + 1, len(swatches)):
            a, b = swatches[i], swatches[j]
            if a.weight < 0.08 or b.weight < 0.08:
                continue
            dist = _hue_dist(a.h, b.h)
            for kind, lo, hi, label in ranked:
                if lo <= dist <= hi:
                    pair = (a.weight + b.weight, kind, f"{label}：{a.hex} 与 {b.hex} 色相差 {dist:.0f}°")
                    if best is None or pair[0] > best[0]:
                        best = pair
                    break
    if best:
        return best[1], best[2]
    a, b = swatches[0], swatches[1]
    dist = _hue_dist(a.h, b.h)
    return "none", f"主色 {a.hex} 与 {b.hex} 色相差 {dist:.0f}°，不属于对比/互补/邻近/相邻/同色系"


def _hue_dist(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _circular_mean(pairs: List[Tuple[float, float]]) -> float:
    x = sum(w * np.cos(np.deg2rad(h)) for h, w in pairs)
    y = sum(w * np.sin(np.deg2rad(h)) for h, w in pairs)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def _hsv_hex(h: float, s: float, v: float) -> str:
    hsv = np.uint8([[[int(h / 2) % 180, int(s * 255), int(v * 255)]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
    return f"#{r:02x}{g:02x}{b:02x}"
