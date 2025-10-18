"""Utility functions to segment plants and clean up the background."""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image
from rembg import remove

__all__ = [
    "segment_plant_rgba",
    "crop_to_alpha_bbox",
    "black_bg_composite",
]


def _ensure_rgba(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        return img.convert("RGBA")
    return img


def segment_plant_rgba(img_rgb: Image.Image) -> Image.Image:
    """Return an RGBA image where the background has been removed using rembg."""

    if img_rgb.mode != "RGB":
        img_rgb = img_rgb.convert("RGB")

    segmented = remove(img_rgb)
    if isinstance(segmented, Image.Image):
        img_rgba = segmented
    elif isinstance(segmented, bytes):
        img_rgba = Image.open(io.BytesIO(segmented))
    else:
        img_rgba = Image.fromarray(segmented)

    return _ensure_rgba(img_rgba)


def crop_to_alpha_bbox(img_rgba: Image.Image, pad: int | Tuple[int, int] = 0) -> Image.Image:
    """Crop an RGBA image around the non-zero alpha region."""

    img_rgba = _ensure_rgba(img_rgba)
    alpha = np.array(img_rgba.split()[-1])
    nonzero = np.argwhere(alpha > 0)
    if nonzero.size == 0:
        return img_rgba

    y_min, x_min = nonzero.min(axis=0)
    y_max, x_max = nonzero.max(axis=0)

    if isinstance(pad, tuple):
        pad_y, pad_x = pad
    else:
        pad_y = pad_x = pad

    y_min = max(int(y_min) - pad_y, 0)
    x_min = max(int(x_min) - pad_x, 0)
    y_max = min(int(y_max) + pad_y, alpha.shape[0] - 1)
    x_max = min(int(x_max) + pad_x, alpha.shape[1] - 1)

    return img_rgba.crop((x_min, y_min, x_max + 1, y_max + 1))


def black_bg_composite(img_rgba: Image.Image) -> Image.Image:
    """Composite an RGBA image on top of a black background."""

    img_rgba = _ensure_rgba(img_rgba)
    background = Image.new("RGB", img_rgba.size, (0, 0, 0))
    background.paste(img_rgba, mask=img_rgba.split()[-1])
    return background
