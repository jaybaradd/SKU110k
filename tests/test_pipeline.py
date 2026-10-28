"""CPU-only checks for the complete shelf inference pipeline."""

import json
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from inference.config import InferenceConfig
from inference.pipeline import ShelfInferencePipeline


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeDetector:
    def __init__(self, boxes, confidences):
        self.boxes = boxes
        self.confidences = confidences
        self.options = None
        self.source = None

    def predict(self, source, **options):
        self.source = source
        self.options = options
        if self.boxes is None:
            detected = None
        else:
            detected = SimpleNamespace(
                xyxy=FakeTensor(self.boxes),
                conf=FakeTensor(self.confidences),
            )
        return [SimpleNamespace(boxes=detected)]


class FakeModels:
    def __init__(self, detector, embeddings):
        self.detector = detector
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.device = "cpu"
        self.crops = None

    def embed_crops(self, crops):
        self.crops = crops
        return self.embeddings


class ShelfPipelineTests(unittest.TestCase):
    def test_predict_returns_aligned_groups_and_original_size_image(self):
        detector = FakeDetector(
            boxes=[[5, 10, 15, 30], [30, 5, 50, 15], [60, 20, 82, 50]],
            confidences=[0.91, 0.8, 0.75],
        )
        models = FakeModels(detector, [[1, 0], [0, 1], [0.98, 0.2]])
        source = Image.new("RGB", (80, 60), "white")

        annotated, result = ShelfInferencePipeline(models, InferenceConfig()).predict(source)

        self.assertEqual(annotated.size, source.size)
        self.assertEqual(source.getpixel((5, 10)), (255, 255, 255))
        self.assertNotEqual(annotated.getpixel((5, 10)), (255, 255, 255))
        self.assertEqual(detector.source.size, source.size)
        self.assertEqual(
            detector.options,
            {"imgsz": 1280, "conf": 0.25, "iou": 0.70,
             "max_det": 750, "device": "cpu", "verbose": False},
        )
        self.assertEqual([crop.size for crop in models.crops], [(20, 20), (20, 20), (30, 30)])
        self.assertEqual(models.crops[0].getpixel((0, 0)), (240, 240, 240))
        self.assertEqual(result["image_width"], 80)
        self.assertEqual(result["image_height"], 60)
        self.assertEqual(result["total_products"], 3)
        self.assertEqual(result["total_groups"], 2)
        self.assertEqual(result["similarity_threshold"], 0.72)
        self.assertEqual(result["grouping_method"], "dinov2_vits14_cosine_average")
        self.assertEqual(
            [item["group_id"] for item in result["detections"]],
            [1, 0, 1],
        )
        self.assertEqual(
            [item["bbox"] for item in result["detections"]],
            [[5.0, 10.0, 15.0, 30.0], [30.0, 5.0, 50.0, 15.0],
             [60.0, 20.0, 80.0, 50.0]],
        )
        self.assertEqual([item["product_id"] for item in result["detections"]], [0, 1, 2])
        self.assertEqual([item["confidence"] for item in result["detections"]], [0.91, 0.8, 0.75])
        json.dumps(result)

    def test_empty_detection_skips_embedding_and_returns_empty_result(self):
        detector = FakeDetector(boxes=None, confidences=None)
        models = FakeModels(detector, [])
        source = Image.new("RGB", (32, 24), "white")

        annotated, result = ShelfInferencePipeline(models, InferenceConfig()).predict(source)

        self.assertEqual(annotated.size, (32, 24))
        self.assertEqual(annotated.getpixel((0, 0)), (255, 255, 255))
        self.assertIsNone(models.crops)
        self.assertEqual(result["total_products"], 0)
        self.assertEqual(result["total_groups"], 0)
        self.assertEqual(result["detections"], [])

    def test_rejects_invalid_detector_confidence(self):
        detector = FakeDetector(boxes=[[0, 0, 10, 10]], confidences=[1.2])
        models = FakeModels(detector, [[1, 0]])

        with self.assertRaisesRegex(RuntimeError, "confidences must be in"):
            ShelfInferencePipeline(models, InferenceConfig()).predict(
                Image.new("RGB", (20, 20), "white")
            )
        self.assertIsNone(models.crops)


if __name__ == "__main__":
    unittest.main()
