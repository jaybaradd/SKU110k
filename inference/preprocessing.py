"""CPU image cropping and aspect-ratio-preserving padding."""

import math

import numpy as np
from PIL import Image


def pad_to_square(
    crop: Image.Image,
    fill: tuple[int, int, int] = (240, 240, 240),
) -> Image.Image:
    """Return a centered RGB square without stretching the source crop."""

    if not isinstance(crop, Image.Image) or min(crop.size) <= 0:
        raise ValueError("crop must be a non-empty PIL image")
    if (
        not isinstance(fill, tuple)
        or len(fill) != 3
        or any(type(channel) is not int or not 0 <= channel <= 255 for channel in fill)
    ):
        raise ValueError("fill must contain three RGB integers in [0, 255]")

    rgb_crop = crop.convert("RGB")
    side = max(rgb_crop.size)
    square = Image.new("RGB", (side, side), fill)
    offset = ((side - rgb_crop.width) // 2, (side - rgb_crop.height) // 2)
    square.paste(rgb_crop, offset)
    return square


def extract_product_crops(
    image: Image.Image,
    boxes_xyxy: np.ndarray,
) -> list[Image.Image]:
    """Clip boxes to an RGB image and return one crop per box, in input order.

    Box coordinates must refer to the supplied image after any EXIF orientation
    has been applied. Invalid or non-intersecting boxes raise ValueError rather
    than silently changing the crop-to-detection alignment.
    """

    if not isinstance(image, Image.Image) or min(image.size) <= 0:
        raise ValueError("image must be a non-empty PIL image")

    try:
        boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("boxes_xyxy must contain numeric xyxy coordinates") from exc

    if boxes.ndim == 1 and boxes.size == 0:
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes_xyxy must have shape (N, 4)")
    if not np.isfinite(boxes).all():
        raise ValueError("boxes_xyxy must contain only finite coordinates")
    if len(boxes) == 0:
        return []

    rgb_image = image.convert("RGB")
    crops = []
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"box {index} has non-positive area")

        left = max(0, min(rgb_image.width, math.floor(x1)))
        top = max(0, min(rgb_image.height, math.floor(y1)))
        right = max(0, min(rgb_image.width, math.ceil(x2)))
        bottom = max(0, min(rgb_image.height, math.ceil(y2)))
        if right <= left or bottom <= top:
            raise ValueError(f"box {index} does not intersect the image")

        crops.append(rgb_image.crop((left, top, right, bottom)))

    return crops
