"""Draw deterministic group-colored boxes on retail shelf images."""

import colorsys
import math
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw


def annotate_groups(
    image: Image.Image,
    boxes_xyxy: np.ndarray,
    group_ids: np.ndarray,
) -> Image.Image:
    """Return an RGB copy with one colored box per detection.

    Group IDs and their colors are meaningful only within this image. Boxes
    use the original image's pixel coordinate system.
    """

    if not isinstance(image, Image.Image) or min(image.size) <= 0:
        raise ValueError("image must be a non-empty PIL image")

    try:
        boxes = np.asarray(boxes_xyxy, dtype=np.float64)
        groups = np.asarray(group_ids)
    except (TypeError, ValueError) as exc:
        raise ValueError("boxes_xyxy and group_ids must be arrays") from exc

    if boxes.ndim == 1 and boxes.size == 0:
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes_xyxy must have shape (N, 4)")
    if groups.ndim != 1 or len(groups) != len(boxes):
        raise ValueError("group_ids must align one-to-one with boxes_xyxy")
    if not np.isfinite(boxes).all():
        raise ValueError("boxes_xyxy must contain finite coordinates")
    if len(groups) and (
        not np.issubdtype(groups.dtype, np.integer) or np.any(groups < 0)
    ):
        raise ValueError("group_ids must be non-negative integers")

    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    line_width = max(2, annotated.width // 1000)

    for index, ((x1, y1, x2, y2), group_id) in enumerate(zip(boxes, groups)):
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"box {index} has non-positive area")
        if x2 <= 0 or y2 <= 0 or x1 >= annotated.width or y1 >= annotated.height:
            raise ValueError(f"box {index} does not intersect the image")

        left = max(0, min(annotated.width - 1, math.floor(x1)))
        top = max(0, min(annotated.height - 1, math.floor(y1)))
        right = max(0, min(annotated.width - 1, math.ceil(x2) - 1))
        bottom = max(0, min(annotated.height - 1, math.ceil(y2) - 1))
        draw.rectangle(
            (left, top, right, bottom),
            outline=_group_color(int(group_id)),
            width=line_width,
        )

    return annotated


def encode_jpeg(image: Image.Image, quality: int = 92) -> bytes:
    """Serialize an image for an HTTP response without writing a file."""

    if not isinstance(image, Image.Image) or min(image.size) <= 0:
        raise ValueError("image must be a non-empty PIL image")
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("quality must be an integer in [1, 100]")

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _group_color(group_id: int) -> tuple[int, int, int]:
    hue = (group_id * 0.61803398875) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return tuple(int(channel * 255) for channel in rgb)
