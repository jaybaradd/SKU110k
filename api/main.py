"""FastAPI endpoint for retail shelf image predictions."""

import base64
import json
import logging
import time
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from api.schemas import PredictionResponse
from inference.pipeline import ShelfInferencePipeline
from inference.visualization import encode_jpeg


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
IMAGE_TYPES = {
    ".jpg": ("image/jpeg", "JPEG"),
    ".jpeg": ("image/jpeg", "JPEG"),
    ".png": ("image/png", "PNG"),
}


def create_app(pipeline: ShelfInferencePipeline | None = None) -> FastAPI:
    """Create an API with an already initialized inference pipeline."""

    app = FastAPI(title="Retail Shelf Assessment")

    @app.middleware("http")
    async def log_prediction_request(request: Request, call_next):
        if request.url.path != "/predict":
            return await call_next(request)

        started = time.perf_counter()
        status_code = 500
        error_type = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            details = getattr(request.state, "prediction_log", {})
            LOGGER.info(
                json.dumps(
                    {
                        "event": "predict_request",
                        "status_code": status_code,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                        "inference_ms": details.get("inference_ms"),
                        "upload_bytes": details.get("upload_bytes"),
                        "image_width": details.get("image_width"),
                        "image_height": details.get("image_height"),
                        "products": details.get("products"),
                        "groups": details.get("groups"),
                        "error_type": details.get("error_type", error_type),
                    },
                    separators=(",", ":"),
                )
            )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
        details = {"upload_bytes": file.size}
        request.state.prediction_log = details
        try:
            if pipeline is None:
                raise HTTPException(status_code=503, detail="Inference service is not ready")

            image = _decode_upload(file)
            details["image_width"], details["image_height"] = image.size
            inference_start = time.perf_counter()
            annotated, result = pipeline.predict(image)
            visualization = base64.b64encode(encode_jpeg(annotated)).decode("ascii")
            inference_ms = round((time.perf_counter() - inference_start) * 1000, 2)
            response = PredictionResponse(
                objects=[
                    {
                        "bbox": detection["bbox"],
                        "confidence": detection["confidence"],
                        "group_id": f"brand_{detection['group_id']}",
                    }
                    for detection in result["detections"]
                ],
                visualization_base64=visualization,
                visualization_mime_type="image/jpeg",
                metadata={
                    "image_width": result["image_width"],
                    "image_height": result["image_height"],
                    "total_products": result["total_products"],
                    "total_groups": result["total_groups"],
                    "grouping_method": result["grouping_method"],
                    "similarity_threshold": result["similarity_threshold"],
                    "inference_time_ms": inference_ms,
                },
            )
            details.update(
                inference_ms=inference_ms,
                products=result["total_products"],
                groups=result["total_groups"],
            )
            return response
        except HTTPException as exc:
            details["error_type"] = type(exc).__name__
            raise
        except Exception as exc:
            details["error_type"] = type(exc).__name__
            LOGGER.exception("Shelf inference failed")
            raise HTTPException(status_code=500, detail="Inference failed") from exc

    return app


def _decode_upload(file: UploadFile) -> Image.Image:
    suffix = Path(file.filename or "").suffix.lower()
    expected_type = IMAGE_TYPES.get(suffix)
    if expected_type is None or file.content_type != expected_type[0]:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image")

    payload = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image upload is too large")
    if not payload:
        raise HTTPException(status_code=400, detail="Image upload is empty")

    try:
        with Image.open(BytesIO(payload)) as decoded:
            if decoded.format != expected_type[1]:
                raise HTTPException(status_code=415, detail="Image content does not match file type")
            if decoded.width * decoded.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions are too large")
            decoded.load()
            return decoded.copy()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc


app = create_app()
