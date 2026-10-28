"""Public JSON response models for shelf predictions."""

import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DetectedObject(BaseModel):
    """One detected product with an image-scoped appearance group ID."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    bbox: list[float] = Field(min_length=4, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)
    group_id: str = Field(pattern=r"^brand_(0|[1-9][0-9]*)$")

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, bbox: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in bbox):
            raise ValueError("bbox coordinates must be finite")
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must have positive area")
        return bbox


class PredictionMetadata(BaseModel):
    """Image dimensions, counts, and inference settings for one request."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    total_products: int = Field(ge=0)
    total_groups: int = Field(ge=0)
    grouping_method: Literal["dinov2_vits14_cosine_average"]
    similarity_threshold: float = Field(gt=0.0, le=1.0)
    inference_time_ms: float = Field(ge=0.0)


class PredictionResponse(BaseModel):
    """Single-endpoint response with directly displayable JPEG data."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    objects: list[DetectedObject]
    visualization_base64: str = Field(min_length=1)
    visualization_mime_type: Literal["image/jpeg"]
    metadata: PredictionMetadata

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if self.metadata.total_products != len(self.objects):
            raise ValueError("total_products must match the number of objects")
        if self.metadata.total_groups != len({obj.group_id for obj in self.objects}):
            raise ValueError("total_groups must match the number of group IDs")
        for obj in self.objects:
            x1, y1, x2, y2 = obj.bbox
            if (
                x1 < 0
                or y1 < 0
                or x2 > self.metadata.image_width
                or y2 > self.metadata.image_height
            ):
                raise ValueError("object bbox must fit within the image")
        return self
