"""Load the trained detector and DINOv2 encoder for GPU inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

from inference.config import InferenceConfig


@dataclass(frozen=True)
class LoadedModels:
    """Models and preprocessing kept alive for one inference process."""

    detector: Any
    visual_encoder: Any
    transform: Callable[[Image.Image], Any]
    device: str
    batch_size: int

    def embed_crops(self, crops: Sequence[Image.Image]) -> np.ndarray:
        """Return float32, L2-normalized DINOv2 features in crop order."""

        if not crops:
            return np.empty((0, 384), dtype=np.float32)
        if any(not isinstance(crop, Image.Image) or min(crop.size) <= 0 for crop in crops):
            raise ValueError("crops must contain non-empty PIL images")

        import torch
        import torch.nn.functional as functional

        embeddings = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                batch = torch.stack(
                    [self.transform(crop) for crop in crops[start : start + self.batch_size]]
                ).to(self.device)
                vectors = self.visual_encoder(batch)
                if vectors.ndim != 2 or vectors.shape != (len(batch), 384):
                    raise RuntimeError("DINOv2 must return one 384-d vector per crop")
                normalized = functional.normalize(vectors.float(), dim=-1)
                embeddings.append(normalized.cpu().numpy())

        return np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)


def load_models(config: InferenceConfig) -> LoadedModels:
    """Load both models once on CUDA; call during container startup."""

    if not isinstance(config, InferenceConfig):
        raise TypeError("config must be an InferenceConfig")
    weights_path = Path(config.yolo_weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {weights_path}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for production inference")

    import timm
    from ultralytics import YOLO

    device = "cuda:0"
    detector = YOLO(str(weights_path)).to(device)
    detector.model.eval()

    visual_encoder = timm.create_model(
        config.dino_model_name,
        pretrained=True,
        num_classes=0,
    ).to(device).eval()
    if visual_encoder.num_features != 384:
        raise RuntimeError("DINOv2 encoder must produce 384-d features")

    data_config = timm.data.resolve_model_data_config(visual_encoder)
    if tuple(data_config["input_size"]) != (3, 518, 518):
        raise RuntimeError("DINOv2 input size must be 3 x 518 x 518")
    transform = timm.data.create_transform(**data_config, is_training=False)

    return LoadedModels(
        detector=detector,
        visual_encoder=visual_encoder,
        transform=transform,
        device=device,
        batch_size=config.dino_batch_size,
    )
