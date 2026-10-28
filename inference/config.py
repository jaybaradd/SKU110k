"""Validated settings for the retail shelf inference pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Inference settings; model files are resolved when models are loaded."""

    yolo_weights_path: Path = Path("weights/best.pt")
    yolo_image_size: int = 1280
    detection_confidence: float = 0.25
    detection_iou: float = 0.70
    max_detections: int = 750
    grouping_similarity: float = 0.72
    grouping_linkage: Literal["average"] = "average"
    dino_model_name: str = "vit_small_patch14_dinov2.lvd142m"
    dino_batch_size: int = 4
    crop_pad_rgb: tuple[int, int, int] = (240, 240, 240)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.yolo_weights_path, Path)
            or self.yolo_weights_path == Path(".")
        ):
            raise ValueError("yolo_weights_path must be a non-empty file path")
        if not _is_positive_int(self.yolo_image_size):
            raise ValueError("yolo_image_size must be positive")
        if not _is_unit_interval(self.detection_confidence):
            raise ValueError("detection_confidence must be in (0, 1]")
        if not _is_unit_interval(self.detection_iou):
            raise ValueError("detection_iou must be in (0, 1]")
        if not _is_positive_int(self.max_detections):
            raise ValueError("max_detections must be positive")
        if not _is_unit_interval(self.grouping_similarity):
            raise ValueError("grouping_similarity must be in (0, 1]")
        if self.grouping_linkage != "average":
            raise ValueError("grouping_linkage must be 'average'")
        if self.dino_model_name != "vit_small_patch14_dinov2.lvd142m":
            raise ValueError("dino_model_name must match the validated DINOv2 model")
        if not _is_positive_int(self.dino_batch_size):
            raise ValueError("dino_batch_size must be positive")
        if (
            not isinstance(self.crop_pad_rgb, tuple)
            or len(self.crop_pad_rgb) != 3
            or any(
                type(channel) is not int or not 0 <= channel <= 255
                for channel in self.crop_pad_rgb
            )
        ):
            raise ValueError("crop_pad_rgb must contain three integers in [0, 255]")

    @classmethod
    def from_env(cls) -> "InferenceConfig":
        """Read deployment-specific overrides without loading models or files."""

        weights_value = os.environ.get("YOLO_WEIGHTS_PATH", "weights/best.pt").strip()
        if not weights_value:
            raise ValueError("YOLO_WEIGHTS_PATH must be a non-empty file path")

        batch_value = os.environ.get("DINO_BATCH_SIZE", "4")
        try:
            batch_size = int(batch_value)
        except ValueError as exc:
            raise ValueError("DINO_BATCH_SIZE must be a positive integer") from exc

        return cls(yolo_weights_path=Path(weights_value), dino_batch_size=batch_size)


def _is_positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _is_unit_interval(value: object) -> bool:
    return type(value) in (int, float) and 0.0 < value <= 1.0
