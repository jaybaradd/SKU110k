# Model Card: SKU-110K Product Detector

## Purpose and output

[`best.pt`](best.pt) is the one-class YOLO11s checkpoint used by this project's shelf inference service. Given a retail shelf image, it predicts product bounding boxes and confidence scores for class `object` (class ID `0`). It does not identify product names or brands. The separate [appearance-grouping stage](../README.md#appearance-grouping-and-inference) uses pretrained DINOv2 features to assign image-local visual groups; those groups are not part of this checkpoint.

## Provenance

| Item | Record |
|---|---|
| Starting checkpoint | Ultralytics pretrained `yolo11s.pt` |
| Fine-tuning data | Full official SKU-110K training split: 8,219 images, 1,208,482 product boxes |
| Validation data | Official SKU-110K validation split: 588 images, 90,968 product boxes |
| Held-out test data | 2,936 images; no aggregate test-set evaluation was run |
| Training environment | Kaggle NVIDIA Tesla T4, PyTorch 2.11.0+cu128, Ultralytics 8.4.190 |
| Main settings | 10 epochs, image size 1280, batch size 2, seed 42, automatic mixed precision, `max_det=750` |
| Training records | [Notebook](../notebook/sku-110k-training.ipynb), [exported arguments](../reports/training_artifacts/args.yaml), [per-epoch results](../reports/training_artifacts/results.csv) |

The notebook converted the source bounding boxes to normalized YOLO labels and clipped coordinates that crossed image boundaries. It also repaired three truncated training JPEGs in temporary Kaggle working storage. The source dataset was not modified or bundled with this repository. See the [training summary](../reports/training_summary.md) for the audit, preprocessing, and full experiment details.

## Checkpoints and integrity

| File | Role | SHA-256 |
|---|---|---|
| [`best.pt`](best.pt) | Selected checkpoint loaded by the API | `696797b741735462c297c0d8d1a68e2393d9fddf4cccf0b016863350dc2ad8a6` |
| [`last.pt`](last.pt) | Final training checkpoint retained for provenance | `d7459edf157d4ffadc74273e8302f4e65fb3aa3f628ebec78c07b1f9b2066364` |

These hashes were computed on the files in this repository. To check a downloaded copy, run `shasum -a 256 weights/best.pt weights/last.pt` from the repository root. The [Modal image](../modal_app.py) packages `best.pt` at `/models/best.pt`; the service does not train or alter the checkpoint.

## Validation evidence

The final recorded epoch (10) had the highest mAP@50–95 in the ten-epoch [results CSV](../reports/training_artifacts/results.csv). Metrics below describe **product detection on the validation split**, not brand grouping or performance on the full test split.

| Precision | Recall | mAP@50 | mAP@50–95 |
|---:|---:|---:|---:|
| 0.90694 | 0.87005 | 0.93357 | 0.59153 |

The [training curves](../reports/training_artifacts/results.png), [F1 curve](../reports/training_artifacts/BoxF1_curve.png), [precision–recall curve](../reports/training_artifacts/BoxPR_curve.png), and [confusion matrix](../reports/training_artifacts/confusion_matrix.png) are exported diagnostics from the same run. One official test image was used for an end-to-end inference smoke test; no test-set summary score is claimed.

## Intended use and limits

This checkpoint is intended to locate densely packed retail products in shelf photos similar to SKU-110K. It can miss tiny, blurred, occluded, or unusual products and can produce duplicate or background boxes. The deployed pipeline uses confidence `0.25`, IoU `0.70`, input size `1280`, and a maximum of `750` detections; changing these can change the counts and latency. It is not a product catalog, inventory truth source, or measured brand recognizer.

The [SKU-110K authors' repository](https://github.com/eg4000/SKU110K_CVPR19) provides the dataset source. No additional dataset was used to fine-tune or evaluate this detector. Because the dataset contains boxes but no brand IDs, brand recognition accuracy cannot be inferred from the detection metrics.

## Upstream terms

The detector starts from and runs with [Ultralytics YOLO](https://docs.ultralytics.com/help/contributing/), which Ultralytics describes as available under AGPL-3.0 or an Enterprise license. The grouping stage loads the [timm DINOv2 ViT-S/14 checkpoint](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m), whose model page lists Apache-2.0; DINOv2 is not included in `best.pt`. The SKU-110K dataset itself is not redistributed here. Check the relevant upstream terms before public redistribution or commercial use; this card does not grant rights to those assets.
