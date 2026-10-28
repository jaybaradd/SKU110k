"""Streamlit client for the retail shelf prediction API."""

import base64
import binascii
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests
import streamlit as st
from dotenv import load_dotenv


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
DEFAULT_ANALYSIS_LOG_PATH = (
    Path(__file__).resolve().parents[1] / "outputs" / "tmp" / "analysis_events.jsonl"
)
IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


class APIResponseError(ValueError):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        super().__init__(f"API returned HTTP {status_code}: {detail}")


def _record_analysis(event: dict) -> None:
    """Append one local JSON line without storing the image or API response."""

    log_path = Path(os.getenv("ANALYSIS_LOG_PATH", str(DEFAULT_ANALYSIS_LOG_PATH)))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(descriptor, line) != len(line):
            raise OSError("Incomplete analysis log write")
    finally:
        os.close(descriptor)


def _api_url() -> str:
    """Read the endpoint from the process environment or local .env file."""

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    return os.getenv("PREDICT_API_URL", "").strip()


def _valid_api_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
        and parsed.path.endswith("/predict")
    )


def _predict(url: str, filename: str, image_bytes: bytes) -> tuple[dict, bytes]:
    """Send one multipart request and decode the annotated JPEG response."""

    mime_type = IMAGE_MIME_TYPES[Path(filename).suffix.lower()]
    response = requests.post(
        url,
        files={"file": (filename, image_bytes, mime_type)},
        timeout=(10, 180),
        allow_redirects=False,
    )
    if not 200 <= response.status_code < 300:
        try:
            error_body = response.json()
            detail = error_body.get("detail") if isinstance(error_body, dict) else None
        except ValueError:
            detail = None
        raise APIResponseError(response.status_code, detail or response.reason)

    try:
        result = response.json()
    except ValueError as exc:
        raise ValueError("API returned invalid JSON") from exc
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("objects"), list)
        or not isinstance(result.get("metadata"), dict)
        or result.get("visualization_mime_type") != "image/jpeg"
        or not isinstance(result.get("visualization_base64"), str)
    ):
        raise ValueError("API returned an unexpected prediction response")

    try:
        annotated_jpeg = base64.b64decode(result["visualization_base64"], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("API returned invalid visualization data") from exc
    if not annotated_jpeg.startswith(b"\xff\xd8") or not annotated_jpeg.endswith(b"\xff\xd9"):
        raise ValueError("API visualization is not a JPEG image")
    return result, annotated_jpeg


def main() -> None:
    st.set_page_config(page_title="Retail shelf assessment", layout="wide")
    st.title("Retail shelf assessment")
    st.write("Upload a shelf photo to detect products and group similar packaging.")

    api_url = _api_url()
    if not _valid_api_url(api_url):
        st.info("Set PREDICT_API_URL in .env to the full API URL ending in /predict.")

    uploaded = st.file_uploader("Shelf image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.session_state.pop("prediction", None)
        return

    image_bytes = uploaded.getvalue()
    st.image(image_bytes, caption="Original image", width="stretch")
    too_large = len(image_bytes) > MAX_UPLOAD_BYTES
    if too_large:
        st.error("Image exceeds the API's 20 MB upload limit.")

    source_key = (hashlib.sha256(image_bytes).hexdigest(), api_url)
    if st.session_state.get("source_key") != source_key:
        st.session_state.pop("prediction", None)
        st.session_state["source_key"] = source_key

    if st.button("Analyze shelf", disabled=too_large or not _valid_api_url(api_url)):
        st.session_state.pop("prediction", None)
        started = time.perf_counter()
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": "analyze_shelf",
            "upload_bytes": len(image_bytes),
            "outcome": "unexpected_error",
            "http_status": None,
            "client_duration_ms": None,
            "inference_ms": None,
            "products": None,
            "groups": None,
            "error_type": None,
        }
        with st.spinner("Analyzing shelf..."):
            try:
                prediction = _predict(api_url, uploaded.name, image_bytes)
                st.session_state["prediction"] = prediction
                metadata = prediction[0]["metadata"]
                event.update(
                    outcome="success",
                    http_status=200,
                    inference_ms=metadata.get("inference_time_ms"),
                    products=metadata.get("total_products"),
                    groups=metadata.get("total_groups"),
                )
            except requests.Timeout:
                event.update(outcome="timeout", error_type="Timeout")
                st.error("The API timed out. Please try again later.")
            except requests.RequestException as exc:
                event.update(outcome="network_error", error_type=type(exc).__name__)
                st.error(f"Could not reach the API: {exc}")
            except APIResponseError as exc:
                event.update(
                    outcome="http_error",
                    http_status=exc.status_code,
                    error_type=type(exc).__name__,
                )
                st.error(str(exc))
            except ValueError as exc:
                event.update(outcome="invalid_response", error_type=type(exc).__name__)
                st.error(str(exc))
            finally:
                event["client_duration_ms"] = round(
                    (time.perf_counter() - started) * 1000, 2
                )
                try:
                    _record_analysis(event)
                except (OSError, ValueError):
                    st.warning("Could not save the local analysis log.")

    prediction = st.session_state.get("prediction")
    if prediction is None:
        return
    result, annotated_jpeg = prediction
    metadata = result["metadata"]
    st.subheader("Results")
    product_column, group_column, time_column = st.columns(3)
    product_column.metric("Products", metadata.get("total_products", 0))
    group_column.metric("Appearance groups", metadata.get("total_groups", 0))
    time_column.metric("Inference time", f"{metadata.get('inference_time_ms', 0):.0f} ms")
    st.image(annotated_jpeg, caption="Grouped products", width="stretch")
    st.json({"objects": result["objects"], "metadata": metadata})

    stem = Path(uploaded.name).stem or "shelf"
    st.download_button(
        "Download annotated image",
        data=annotated_jpeg,
        file_name=f"{stem}_grouped.jpg",
        mime="image/jpeg",
    )
    st.download_button(
        "Download JSON",
        data=json.dumps(result, indent=2),
        file_name=f"{stem}_prediction.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
