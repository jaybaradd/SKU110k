# Retail Shelf Product Detection and Appearance Grouping

This project detects individual products in a retail shelf photo, groups products with similar packaging, and returns colored bounding boxes and structured results. It was built for the TNSQAI data science intern assignment.

**Status:** The Streamlit UI was tested against a Modal T4 development service. The GPU API was deployed on 29 October 2026 and passed a real shelf-image request. Private repository submission remains.

## Architecture

~~~mermaid
flowchart LR
    A["Shelf JPEG / PNG"] --> B["Streamlit UI"]
    B --> C["FastAPI POST /predict on Modal T4"]
    C --> D["Fine-tuned YOLO11s detector"]
    D --> E["Square-padded product crops"]
    E --> F["DINOv2 ViT-S/14 embeddings"]
    F --> G["Cosine distance and average-linkage grouping"]
    G --> H["Colored boxes and JSON"]
    H --> B
~~~

The detector predicts one class, `object`. Response values such as `brand_3` are **image-specific appearance group IDs**, not recognized brand names. They are not stable across photos.

## Dataset and preprocessing

Only [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) was used for detector training and validation. The [training notebook](notebook/sku-110k-training.ipynb) used a `SKU110K_fixed` copy mounted on Kaggle. Its annotation CSVs contain `image_name,x1,y1,x2,y2,class,image_width,image_height`. There are product boxes but no brand labels. No other dataset was used to fine-tune or evaluate the detector; the starting YOLO11s and DINOv2 weights are pretrained.

| Official split | Images | Product boxes | Use in this project |
|---|---:|---:|---|
| Train | 8,219 | 1,208,482 | Detector fine-tuning |
| Validation | 588 | 90,968 | Detector validation and sample inference |
| Test | 2,936 | 431,546 | One sample inference check; no aggregate test evaluation |

The notebook audits the annotations, clips boxes to image boundaries, converts `xyxy` coordinates to normalized YOLO labels, and symlinks the images into a training layout. It flagged 14,918 training boxes in the bounds audit; the small overflows were clipped, with no train or validation annotations dropped during conversion. Three truncated training JPEGs (`train_4222.jpg`, `train_882.jpg`, `train_924.jpg`) needed recoverable copies in Kaggle working storage. The source dataset was not changed.

Dense shelves contain many small products. In the notebook's simulated resizing analysis, about 30.37% of training boxes would be narrower than 16 pixels at 640-pixel input, versus 3.97% at 1280 pixels. That motivated 1280-pixel training and inference despite higher compute use. The full dataset is not stored in this repository.

## Detector training and validation

**YOLO11s** was initialized from the pretrained `yolo11s.pt` checkpoint and fine-tuned on the full official SKU-110K training split. The exported [`args.yaml`](reports/training_artifacts/args.yaml) records the full configuration.

| Setting | Value |
|---|---|
| Training environment | Kaggle NVIDIA Tesla T4, CUDA device 0 |
| Epochs / batch size / input size | 10 / 2 / 1280 |
| Automatic mixed precision / workers | Enabled / 2 |
| Mosaic / MixUp | 0.0 / 0.0 |
| Seed / deterministic mode | 42 / enabled |
| Maximum detections | 750 |
| Saved inference checkpoint | [`weights/best.pt`](weights/best.pt) |

The final row of [`results.csv`](reports/training_artifacts/results.csv) reports the following **validation** results on 588 images and 90,968 annotated products. These are not test-set scores or brand-grouping metrics.

| Precision | Recall | mAP@50 | mAP@50–95 |
|---:|---:|---:|---:|
| 0.907 | 0.870 | 0.934 | 0.592 |

Training and validation loss histories are in [`results.png`](reports/training_artifacts/results.png) and [`results.csv`](reports/training_artifacts/results.csv). The exported [F1 curve](reports/training_artifacts/BoxF1_curve.png), [precision–recall curve](reports/training_artifacts/BoxPR_curve.png), and [confusion matrix](reports/training_artifacts/confusion_matrix.png) provide further validation evidence.

The service uses `weights/best.pt` (SHA-256: `696797b741735462c297c0d8d1a68e2393d9fddf4cccf0b016863350dc2ad8a6`). `weights/last.pt` is retained as a training artifact. Retraining is not necessary for inference.

## Appearance grouping and inference

For each image, [the inference pipeline](inference/pipeline.py):

1. Applies EXIF orientation and converts the image to RGB.
2. Runs the fine-tuned YOLO11s detector with `imgsz=1280`, confidence `0.25`, IoU `0.70`, and `max_det=750`.
3. Clips boxes to the original image, crops every product, and centers each crop on an RGB `(240, 240, 240)` square without stretching it.
4. Applies the model-specific 518 × 518 evaluation transform for pretrained `timm` DINOv2 ViT-S/14 (`vit_small_patch14_dinov2.lvd142m`). It extracts 384-dimensional, L2-normalized embeddings in batches of four.
5. Computes pairwise cosine distances and average-linkage agglomerative clusters with similarity threshold `0.72`, equivalent to distance threshold `0.28`. Group IDs are assigned top-to-bottom, then left-to-right.
6. Draws a deterministic color per group on the original-size image. The API returns product records, metadata, and a base64-encoded annotated JPEG.

The threshold was selected using a small set of hand-picked visual comparisons, not a labeled brand benchmark. CLIP and OCR were also explored in the notebook; CLIP made questionable visual merges and OCR performed poorly on small crops. DINOv2 was **not** fine-tuned on SKU-110K. Because SKU-110K has no brand labels, this project does not claim measured brand recognition accuracy.

## FastAPI request and response

The service exposes `POST /predict`. Send a JPEG or PNG using multipart form field `file`. The API accepts at most **20 MiB** and **20 million decoded pixels**. It returns 4xx for invalid uploads and 5xx for inference failures without exposing model stack traces to clients.

~~~bash
curl -sS -X POST "https://jaydharmishtha--tnsqai-retail-shelf-serve.us-east.modal.run/predict" \
  -F "file=@path/to/shelf.jpg;type=image/jpeg" \
  -o prediction.json
~~~

| Response field | Contents |
|---|---|
| `objects[]` | Each product's original-image pixel `bbox: [x1,y1,x2,y2]`, `confidence`, and image-specific `group_id` |
| `visualization_base64` | Full annotated JPEG encoded inline for immediate use by the single endpoint |
| `visualization_mime_type` | `image/jpeg` |
| `metadata` | Image dimensions, product and group counts, grouping method, threshold, and server-side `inference_time_ms` |

The live response includes the JPEG as base64 rather than returning an inaccessible container-local file path. This makes the JSON larger. The saved example JSON files below omit that base64 field and use `visualization_file` to point to the adjacent JPEG; they are compact copies for inspection, not the exact live response schema.

## Modal GPU deployment

[`modal_app.py`](modal_app.py) defines one GPU-backed FastAPI service using `modal.App` and `@app.function`, the supported form used here in place of the PDF's older `@stub.function` wording. It requests a T4 and limits the service to one container and one input per container. The Modal image installs [pinned Linux inference dependencies](requirements-inference.txt), packages `weights/best.pt` at `/models/best.pt`, and caches DINOv2 weights during image build. Hugging Face offline mode is enabled at inference time; both models load once per container. Deployed inference has no Mac or Kaggle file path.

From the repository root, use Python 3.11 and your **own** configured Modal account:

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install modal==1.6.3
modal token new

# Temporary development URL, available while this command runs:
modal serve modal_app.py

# Deploy to your own workspace:
modal deploy modal_app.py
~~~

The deployed API is **https://jaydharmishtha--tnsqai-retail-shelf-serve.us-east.modal.run/predict**. Its [Modal deployment](https://modal.com/apps/jaydharmishtha/main/deployed/tnsqai-retail-shelf) is named `tnsqai-retail-shelf` and uses a T4. `modal serve` is for development; `modal deploy` creates a deployment whose URL remains addressable after the local terminal closes. Modal functions normally scale to zero when idle, but container startup, requests, and the warm period can incur usage. This endpoint has no authentication, so anyone with the URL can trigger usage. The author's workspace has usage and out-of-pocket limits configured; requests may stop working when a limit is reached. Reviewers can still deploy the checkpoint in their own workspace. See [Modal's deployment guide](https://modal.com/docs/guide/managing-deployments) and [budget guide](https://modal.com/docs/guide/budgets).

One `val_100.jpg` request to the deployed `/predict` URL returned HTTP 200, **144 products**, **82 groups**, and a decodable 2448 × 3264 annotated JPEG. The first post-deploy request took **23.39 seconds** at the client, including network and startup overhead; the response reported **6,748.52 ms** of server inference and encoding. These are measurements from one request, not a latency benchmark.

Reviewers can deploy the tracked checkpoint to their **own** Modal workspace with these instructions. They do not need the author's token or credits.

## Streamlit frontend and observability

The [Streamlit client](frontend/app.py) runs locally and sends images to the Modal API. It shows the original and annotated images, product and group counts, inference time, formatted JSON, and download buttons. It does not load YOLO, DINOv2, or PyTorch.

~~~bash
python -m pip install -r requirements-frontend.txt
cp .env.example .env
# Set PREDICT_API_URL in .env to the full Modal URL ending in /predict.
streamlit run frontend/app.py
~~~

Each **Analyze shelf** click appends one JSON line to `outputs/tmp/analysis_events.jsonl` on the machine running Streamlit. It records UTC time, upload size, outcome, HTTP status, elapsed client time, and available inference/product/group metrics. It stores no image, filename, API URL, or cost. Set `ANALYSIS_LOG_PATH` to change this path. The API also writes request summaries to the Modal container log stream. `.env` and `outputs/tmp/` are ignored by Git.

## Verified examples and local tests

The following SKU-110K images passed through the real Modal T4 development service. The reported times measure server inference and encoding, not Modal cold start or network time.

| Image | Products | Appearance groups | Reported inference time | Output |
|---|---:|---:|---:|---|
| `val_100.jpg` | 144 | 82 | 8,020 ms | [Annotated JPEG](outputs/examples/val_100_annotated.jpg) · [Compact JSON](outputs/examples/val_100_prediction.json) |
| `test_1904.jpg` | 136 | 48 | 5,336 ms | [Annotated JPEG](outputs/examples/test_1904_annotated.jpg) · [Compact JSON](outputs/examples/test_1904_prediction.json) |

These are individual examples, not an aggregate test-set evaluation. Similar packaging can be merged incorrectly; products from the same brand can be split.

The CPU-only test suite covers grouping, a pipeline with fake models, and the FastAPI contract with a fake pipeline. It does not load the trained models or incur GPU charges. The current Intel Mac environment passes 13 tests:

~~~bash
python -m unittest discover -s tests -v
~~~

For a fresh CPU-only Python 3.11 test environment, install the frontend requirements plus the tested CPU dependencies below. Do **not** install `requirements-inference.txt` on an Intel Mac; its CUDA PyTorch wheels are for Modal's Linux image.

~~~bash
python -m pip install -r requirements-frontend.txt
python -m pip install \
  numpy==1.26.4 scikit-learn==1.5.2 Pillow==10.4.0 \
  fastapi==0.162.0 pydantic==2.9.2 python-multipart==0.0.32 httpx==0.28.1
python -m unittest discover -s tests -v
~~~

## Reproducing training

The [notebook](notebook/sku-110k-training.ipynb) contains the dataset audit, YOLO label conversion, small pilot runs, JPEG recovery, full YOLO11s training run, validation results, and grouping experiments.

1. Obtain `SKU110K_fixed` from the [SKU-110K authors](https://github.com/eg4000/SKU110K_CVPR19) or an equivalent Kaggle copy. The dataset is not bundled here.
2. Run the notebook on a GPU system with the dataset mounted. Update its historical Kaggle `ROOT` path if your mount differs. Execute the audit, conversion, JPEG recovery, and full-training cells in order. The conversion creates `data.yaml` and normalized YOLO labels in working storage.
3. Compare the exported [training arguments](reports/training_artifacts/args.yaml), [per-epoch metrics](reports/training_artifacts/results.csv), and plots. Environment and GPU differences can affect exact results.
4. To reproduce inference without retraining, use the included `weights/best.pt` and deploy `modal_app.py` as above.

The notebook's Kaggle paths are historical training paths, not deployment requirements. The full official test split has not been evaluated as a held-out benchmark in this project.

## Limitations and remaining submission work

- Appearance groups are not verified brand identities. Group IDs are local to one image, and no brand-label accuracy is available.
- Average-linkage grouping uses an O(N²) distance matrix. Similar packages can cause false merges; different views or package designs can cause false splits.
- The detector may miss tiny or occluded products, and inference caps output at 750 detections.
- The live endpoint has no authentication. It is subject to the author's Modal workspace limits and may become unavailable when those limits are reached.
- Private repository submission is still pending. The development service was tested and stopped; the final deployed API passed one real request.
- Optional future work: labeled brand evaluation, better text recognition, embedding fine-tuning, and an RF-DETR comparison.

SKU-110K, pretrained YOLO11s, and pretrained DINOv2 each have upstream terms. Keep the submission repository private as the assignment requires and review those terms before wider redistribution.
