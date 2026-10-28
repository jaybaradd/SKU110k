"""HTTP contract checks that do not load the GPU models."""

import base64
import unittest
from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app


class FakePipeline:
    def __init__(self, detections=None, fail=False):
        self.detections = detections if detections is not None else []
        self.fail = fail

    def predict(self, image):
        if self.fail:
            raise RuntimeError("model failure details")
        width, height = image.size
        return image.copy(), {
            "image_width": width,
            "image_height": height,
            "total_products": len(self.detections),
            "total_groups": len({item["group_id"] for item in self.detections}),
            "grouping_method": "dinov2_vits14_cosine_average",
            "similarity_threshold": 0.72,
            "detections": self.detections,
        }


class PredictApiTests(unittest.TestCase):
    @staticmethod
    def image_bytes():
        output = BytesIO()
        Image.new("RGB", (32, 24), "white").save(output, format="JPEG")
        return output.getvalue()

    def post_image(self, client, payload=None):
        return client.post(
            "/predict",
            files={
                "file": (
                    "shelf.jpg",
                    self.image_bytes() if payload is None else payload,
                    "image/jpeg",
                )
            },
        )

    def test_prediction_returns_objects_metadata_and_decodable_image(self):
        detections = [
            {"bbox": [1, 2, 12, 16], "confidence": 0.91, "group_id": 0},
            {"bbox": [15, 3, 30, 20], "confidence": 0.84, "group_id": 1},
        ]
        with TestClient(create_app(FakePipeline(detections))) as client:
            response = self.post_image(client)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            [item["group_id"] for item in body["objects"]],
            ["brand_0", "brand_1"],
        )
        self.assertEqual(body["objects"][0]["bbox"], [1, 2, 12, 16])
        self.assertEqual(body["metadata"]["total_products"], 2)
        self.assertEqual(body["metadata"]["total_groups"], 2)
        self.assertEqual(body["visualization_mime_type"], "image/jpeg")
        with Image.open(BytesIO(base64.b64decode(body["visualization_base64"]))) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.size, (32, 24))

    def test_empty_shelf_returns_empty_objects_and_a_visualization(self):
        with TestClient(create_app(FakePipeline())) as client:
            response = self.post_image(client)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["objects"], [])
        self.assertEqual(body["metadata"]["total_products"], 0)
        self.assertEqual(body["metadata"]["total_groups"], 0)
        self.assertTrue(base64.b64decode(body["visualization_base64"]))

    def test_rejects_invalid_and_oversized_uploads(self):
        with TestClient(create_app(FakePipeline())) as client:
            invalid = self.post_image(client, b"not a JPEG")
            empty = self.post_image(client, b"")
            oversized = self.post_image(client, b"x" * (20 * 1024 * 1024 + 1))

        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(oversized.status_code, 413)

    def test_unavailable_and_failed_inference_are_not_exposed(self):
        with TestClient(create_app()) as client:
            unavailable = self.post_image(client)
        self.assertEqual(unavailable.status_code, 503)

        with patch("api.main.LOGGER.exception"):
            with TestClient(create_app(FakePipeline(fail=True))) as client:
                failed = self.post_image(client)

        self.assertEqual(failed.status_code, 500)
        self.assertEqual(failed.json(), {"detail": "Inference failed"})


if __name__ == "__main__":
    unittest.main()
