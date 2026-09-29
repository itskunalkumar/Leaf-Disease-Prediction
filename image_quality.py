from typing import Any, Dict
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter


def assess_image(image: Image.Image) -> Dict[str, Any]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    brightness = float(gray.mean())
    edges = np.asarray(rgb.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    detail = float(edges.var())

    warnings = []
    if min(width, height) < 160:
        warnings.append("Image resolution is low; use a closer leaf photo.")
    if brightness < 35:
        warnings.append("Image is very dark; use natural or even lighting.")
    if brightness > 235:
        warnings.append("Image is overexposed; avoid glare.")
    if detail < 8:
        warnings.append("Image may be blurry; retake a sharp photo.")

    quality = "good" if not warnings else ("acceptable" if len(warnings) == 1 else "poor")
    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 1),
        "detail_score": round(detail, 1),
        "quality": quality,
        "warnings": warnings,
    }


def assess_image_bytes(data: bytes) -> Dict[str, Any]:
    with Image.open(BytesIO(data)) as image:
        return assess_image(image)
