"""Run product detection, appearance grouping, and visualization."""

from typing import Any

import numpy as np
from PIL import Image, ImageOps

from inference.config import InferenceConfig
from inference.grouping import assign_appearance_groups
from inference.model_loader import LoadedModels
from inference.preprocessing import extract_product_crops, pad_to_square
from inference.visualization import annotate_groups


class ShelfInferencePipeline:
    """Use models initialized once to analyze individual shelf images."""

    def __init__(self, models: LoadedModels, config: InferenceConfig) -> None:
        self.models = models
        self.config = config

    def predict(self, image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
        """Return an annotated image and JSON-compatible internal results."""

        if not isinstance(image, Image.Image) or min(image.size) <= 0:
            raise ValueError("image must be a non-empty PIL image")
        rgb_image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = rgb_image.size

        results = self.models.detector.predict(
            source=rgb_image,
            imgsz=self.config.yolo_image_size,
            conf=self.config.detection_confidence,
            iou=self.config.detection_iou,
            max_det=self.config.max_detections,
            device=self.models.device,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError("detector must return one result for one image")

        detected_boxes = results[0].boxes
        if detected_boxes is None:
            boxes = np.empty((0, 4), dtype=np.float32)
            confidences = np.empty(0, dtype=np.float32)
        else:
            boxes = np.asarray(detected_boxes.xyxy.cpu().numpy(), dtype=np.float32).copy()
            confidences = np.asarray(
                detected_boxes.conf.cpu().numpy(), dtype=np.float32
            )
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise RuntimeError("detector boxes must have shape (N, 4)")
        if confidences.ndim != 1 or len(confidences) != len(boxes):
            raise RuntimeError("detector confidences must align with boxes")
        if not np.isfinite(boxes).all() or not np.isfinite(confidences).all():
            raise RuntimeError("detector returned non-finite values")
        if np.any((confidences < 0.0) | (confidences > 1.0)):
            raise RuntimeError("detector confidences must be in [0, 1]")

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        if np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise RuntimeError("detector returned a box without positive image area")

        if len(boxes):
            crops = extract_product_crops(rgb_image, boxes)
            square_crops = [
                pad_to_square(crop, self.config.crop_pad_rgb) for crop in crops
            ]
            embeddings = self.models.embed_crops(square_crops)
            group_ids = assign_appearance_groups(
                embeddings, boxes, self.config.grouping_similarity
            )
        else:
            group_ids = np.empty(0, dtype=np.int64)

        annotated = annotate_groups(rgb_image, boxes, group_ids)
        detections = [
            {
                "product_id": index,
                "bbox": [round(float(value), 2) for value in box],
                "confidence": round(float(confidence), 4),
                "group_id": int(group_id),
            }
            for index, (box, confidence, group_id) in enumerate(
                zip(boxes, confidences, group_ids)
            )
        ]
        output = {
            "image_width": width,
            "image_height": height,
            "total_products": len(detections),
            "total_groups": len(set(group_ids.tolist())),
            "grouping_method": "dinov2_vits14_cosine_average",
            "similarity_threshold": self.config.grouping_similarity,
            "detections": detections,
        }
        return annotated, output
