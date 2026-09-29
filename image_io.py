"""Load a downscaled preview without decoding the full-resolution photo."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_preview_bgr(path: str, max_side: int = 640) -> Tuple[Optional[np.ndarray], int, int]:
    try:
        with Image.open(path) as im:
            width, height = im.size
            try:
                im.draft("RGB", (max_side, max_side))
            except Exception:
                pass
            im = ImageOps.exif_transpose(im) or im
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
            rgb = np.array(im)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), width, height
    except Exception:
        return None, 0, 0
