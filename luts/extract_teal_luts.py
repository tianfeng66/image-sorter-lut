"""Extract 2 teal/spring-green LUTs from the reference photo set."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.clip(rgb, 0.0, 1.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    df = mx - mn
    h = np.zeros_like(mx)
    eq = df > 1e-8
    r_is = eq & (np.abs(mx - r) <= 1e-8)
    g_is = eq & ~r_is & (np.abs(mx - g) <= 1e-8)
    b_is = eq & ~r_is & ~g_is
    h = np.where(r_is, np.mod((g - b) / np.where(eq, df, 1.0), 6.0), h)
    h = np.where(g_is, (b - r) / np.where(eq, df, 1.0) + 2.0, h)
    h = np.where(b_is, (r - g) / np.where(eq, df, 1.0) + 4.0, h)
    s = np.where(mx > 1e-8, df / np.maximum(mx, 1e-8), 0.0)
    return h * 60.0, s, mx


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = np.mod(h, 360.0)
    c = v * np.clip(s, 0.0, 1.5)
    x = c * (1.0 - np.abs(np.mod(h / 60.0, 2.0) - 1.0))
    m = v - c
    z = np.zeros_like(h)
    sector = (h / 60.0).astype(np.int32) % 6
    r = np.choose(sector, [c, x, z, z, x, c])
    g = np.choose(sector, [x, c, c, x, z, z])
    b = np.choose(sector, [z, z, x, c, c, x])
    rgb = np.stack([r, g, b], axis=-1) + m[..., None]
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def hue_weight(h: np.ndarray, center: float, width: float) -> np.ndarray:
    d = np.abs(np.mod(h - center + 180.0, 360.0) - 180.0)
    return np.clip(1.0 - d / width, 0.0, 1.0)


def tone_curve(y: np.ndarray, black: float, gamma: float, contrast: float) -> np.ndarray:
    x = (y - black) / max(1.0 - black, 1e-6)
    x = np.clip(x, 0.0, 1.0) ** gamma
    x = np.clip((x - 0.48) * contrast + 0.50, 0.0, 1.0)
    # soft highlight sparkle without clipping
    hi = np.clip((x - 0.78) / 0.22, 0.0, 1.0)
    x = x + hi * 0.035
    return np.clip(x, 0.0, 1.0)


def apply_look(rgb: np.ndarray, preset: str) -> np.ndarray:
    """Map neutral sRGB toward the measured blue-green look."""
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    if preset == "cyan_lake":
        black, gamma, contrast = 0.028, 0.90, 1.10
        # shadows teal-green, highlights cool mint-white
        sh_tint = np.array([-0.045, 0.035, 0.028], np.float32)
        hi_tint = np.array([-0.018, 0.012, 0.010], np.float32)
        hue_moves = [
            (215, 55, -22.0),  # sky blue -> cyan
            (190, 35, 4.0),  # aqua stay ~194
            (125, 45, -22.0),  # forest green -> lime
            (100, 30, -6.0),
        ]
        sat_moves = [
            (194, 50, 0.28),
            (96, 45, 0.26),
            (55, 25, 0.06),
            (0, 30, -0.12),
            (330, 30, -0.10),
        ]
        val_moves = [
            (194, 50, 0.08),
            (96, 40, 0.06),
        ]
        mid_teal = np.array([-0.035, 0.018, 0.032], np.float32)
    else:
        black, gamma, contrast = 0.022, 0.93, 1.08
        sh_tint = np.array([-0.030, 0.042, -0.008], np.float32)
        hi_tint = np.array([-0.010, 0.016, -0.004], np.float32)
        hue_moves = [
            (215, 50, -14.0),  # sky stays a bit more blue
            (125, 45, -28.0),  # green -> yellow-green ~96
            (90, 25, -4.0),
        ]
        sat_moves = [
            (96, 50, 0.34),
            (200, 45, 0.20),
            (55, 28, 0.10),
            (0, 30, -0.10),
        ]
        val_moves = [
            (96, 45, 0.10),
            (200, 40, 0.05),
        ]
        mid_teal = np.array([-0.028, 0.030, 0.006], np.float32)

    y2 = tone_curve(y, black, gamma, contrast)
    scale = np.where(y > 1e-5, y2 / np.maximum(y, 1e-5), 1.0)
    rgb = np.clip(rgb * scale[..., None], 0.0, 1.0)

    sh_w = np.clip(1.0 - y2 / 0.42, 0.0, 1.0)[..., None]
    hi_w = np.clip((y2 - 0.62) / 0.38, 0.0, 1.0)[..., None]
    mid_w = (1.0 - sh_w) * (1.0 - hi_w)
    rgb = rgb + sh_w * sh_tint + hi_w * hi_tint + mid_w * mid_teal
    rgb = np.clip(rgb, 0.0, 1.0)

    h, s, v = rgb_to_hsv(rgb)
    dh = np.zeros_like(h)
    ds = np.zeros_like(s)
    dv = np.zeros_like(v)
    for center, width, amount in hue_moves:
        dh += hue_weight(h, center, width) * amount
    for center, width, amount in sat_moves:
        ds += hue_weight(h, center, width) * amount
    for center, width, amount in val_moves:
        dv += hue_weight(h, center, width) * amount
    s = np.clip(s * (1.0 + ds), 0.0, 1.0)
    v = np.clip(v * (1.0 + dv), 0.0, 1.0)
    rgb = hsv_to_rgb(h + dh, s, v)

    # keep near-neutrals from going neon
    h2, s2, v2 = rgb_to_hsv(rgb)
    gray = np.clip(1.0 - s2 / 0.12, 0.0, 1.0)
    rgb = rgb * (1.0 - 0.15 * gray[..., None]) + v2[..., None] * 0.15 * gray[..., None]
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def identity_hald(cube: int = 64) -> np.ndarray:
    side = int(round((cube**3) ** 0.5))
    idx = np.arange(side * side, dtype=np.int32).reshape(side, side)
    r = idx % cube
    g = (idx // cube) % cube
    b = idx // (cube * cube)
    return np.stack([r, g, b], axis=-1).astype(np.float32) / float(cube - 1)


def identity_cube(size: int = 33) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, size, dtype=np.float32)
    b, g, r = np.meshgrid(xs, xs, xs, indexing="ij")
    # Resolve/cube order: R fastest, then G, then B
    return np.stack([r, g, b], axis=-1).reshape(-1, 3)


def save_hald(path: Path, rgb: np.ndarray) -> None:
    bgr = (np.clip(rgb[..., ::-1], 0, 1) * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(str(path), bgr)


def save_cube(path: Path, rgb: np.ndarray, size: int, title: str) -> None:
    lines = [
        f'TITLE "{title}"',
        f"LUT_3D_SIZE {size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
        "",
    ]
    for r, g, b in rgb:
        lines.append(f"{r:.6f} {g:.6f} {b:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def swatch_bar(colors: list[tuple[int, int, int]], w: int, h: int) -> np.ndarray:
    bar = np.zeros((h, w, 3), np.uint8)
    n = len(colors)
    for i, (r, g, b) in enumerate(colors):
        x0, x1 = int(i * w / n), int((i + 1) * w / n)
        bar[:, x0:x1] = (b, g, r)
    return bar


def hue_ramp(w: int, h: int, sat: float = 0.65, val: float = 0.78) -> np.ndarray:
    xs = np.linspace(0, 360, w, dtype=np.float32)
    hhue = np.tile(xs, (h, 1))
    s = np.full_like(hhue, sat)
    v = np.full_like(hhue, val)
    return hsv_to_rgb(hhue, s, v)


def make_test_scene(w: int = 720, h: int = 420) -> np.ndarray:
    img = np.zeros((h, w, 3), np.float32)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    yy, xx = np.broadcast_arrays(yy, xx)
    sky = np.stack(
        [
            0.45 + 0.35 * yy,
            0.58 + 0.22 * yy,
            0.78 + 0.12 * yy,
        ],
        axis=-1,
    )
    grass = np.stack(
        [
            0.22 + 0.10 * xx,
            0.38 + 0.08 * (1 - yy),
            0.16 + 0.04 * xx,
        ],
        axis=-1,
    )
    water = np.stack(
        [
            0.28 + 0.05 * xx,
            np.full_like(xx, 0.48),
            np.full_like(xx, 0.58),
        ],
        axis=-1,
    )
    img[:] = sky
    img[int(h * 0.42) : int(h * 0.62)] = water[int(h * 0.42) : int(h * 0.62)]
    img[int(h * 0.62) :] = grass[int(h * 0.62) :]
    # gray cards
    for i, g in enumerate((0.18, 0.42, 0.70)):
        x0 = 40 + i * 70
        img[20:70, x0 : x0 + 54] = g
    # skin-ish
    img[20:70, 260:320] = (0.76, 0.58, 0.48)
    # red flower
    img[20:70, 340:400] = (0.72, 0.18, 0.16)
    return np.clip(img, 0, 1)


def label(bgr: np.ndarray, text: str, y: int, x: int = 24) -> None:
    cv2.putText(bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (28, 28, 28), 2, cv2.LINE_AA)


def paste(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    h, w = src.shape[:2]
    dst[y : y + h, x : x + w] = src


def build_preview(halds: dict[str, np.ndarray], titles: dict[str, str]) -> np.ndarray:
    scene = make_test_scene(640, 360)
    ramp = hue_ramp(640, 36)
    canvas = np.full((980, 1360, 3), 245, np.uint8)
    x = 30
    for key, title in titles.items():
        hald_s = cv2.resize(
            (halds[key][..., ::-1] * 255).astype(np.uint8),
            (420, 420),
            interpolation=cv2.INTER_NEAREST,
        )
        before = (np.clip(scene[..., ::-1], 0, 1) * 255).astype(np.uint8)
        after = (np.clip(apply_look(scene, key)[..., ::-1], 0, 1) * 255).astype(np.uint8)
        ramp_b = (np.clip(ramp[..., ::-1], 0, 1) * 255).astype(np.uint8)
        ramp_a = (np.clip(apply_look(ramp, key)[..., ::-1], 0, 1) * 255).astype(np.uint8)
        if key == "cyan_lake":
            colors = [(54, 124, 80), (95, 176, 189), (93, 169, 213), (160, 204, 188), (229, 237, 229)]
        else:
            colors = [(80, 124, 54), (102, 163, 80), (93, 169, 213), (134, 188, 96), (236, 240, 228)]
        bar = swatch_bar(colors, 420, 40)
        paste(canvas, hald_s, x, 50)
        paste(canvas, bar, x, 478)
        paste(canvas, cv2.resize(before, (640, 170)), x, 532)
        paste(canvas, cv2.resize(after, (640, 170)), x, 710)
        paste(canvas, np.concatenate([ramp_b, ramp_a], 0), x, 890)
        label(canvas, title, 34, x)
        x += 680
    return canvas


def main() -> None:
    presets = {
        "cyan_lake": "Cyan Lake / Qing Hu",
        "spring_green": "Spring Green / Chun Lu",
    }
    halds = {}
    ident = identity_hald(64)
    cube_in = identity_cube(33)
    for key, title in presets.items():
        hald = apply_look(ident, key)
        cube = apply_look(cube_in.reshape(-1, 1, 3), key).reshape(-1, 3)
        save_hald(OUT / f"lut_{key}_hald.png", hald)
        save_cube(OUT / f"lut_{key}.cube", cube, 33, title)
        halds[key] = hald
        print("wrote", key)
    preview = build_preview(
        halds,
        {
            "cyan_lake": "LUT 1  Cyan Lake  teal water / cool sparkle",
            "spring_green": "LUT 2  Spring Green  lime foliage / sky blue",
        },
    )
    cv2.imwrite(str(OUT / "lut_preview.jpg"), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print("done", OUT)


if __name__ == "__main__":
    main()
