# Training Summary

## Experiment

A one-class YOLO11s product detector was fine-tuned on the full official training split of [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19). Its output class is `object`. The run started from the pretrained `yolo11s.pt` checkpoint and produced [`weights/best.pt`](../weights/best.pt), which the deployed inference pipeline loads. The [training notebook](../notebook/sku-110k-training.ipynb), exported [`args.yaml`](training_artifacts/args.yaml), and [`results.csv`](training_artifacts/results.csv) are the primary records for this report.

| Split | Images | Product boxes | Role |
|---|---:|---:|---|
| Train | 8,219 | 1,208,482 | Fine-tuning |
| Validation | 588 | 90,968 | Per-epoch detector evaluation |
| Test | 2,936 | 431,546 | No aggregate evaluation in this run |

The dataset contains product bounding boxes, not brand identities. No other dataset was used for detector fine-tuning or evaluation. The pretrained starting weights are distinct from the SKU-110K training data.

## Dataset preparation

The notebook read the fixed dataset's annotation CSVs, checked image availability and split overlap, and converted `image_name,x1,y1,x2,y2,class,image_width,image_height` records into one-class normalized YOLO labels. Image files were symlinked into the Kaggle working directory to avoid duplicating the dataset.

The training bounds audit flagged **14,918** boxes, mostly for small right or bottom edge overflows. Coordinates were clipped to the image bounds and positive-area boxes retained. The converted train and validation labels kept all **1,208,482** and **90,968** annotations, respectively. Three truncated training JPEGs (`train_4222.jpg`, `train_882.jpg`, `train_924.jpg`) were decoded and saved as recoverable copies in Kaggle working storage; their training symlinks were updated. Original dataset files were left untouched.

For small-product detection, the notebook compared simulated letterbox resizing: about **30.37%** of training boxes would be narrower than 16 pixels at 640-pixel input, compared with **3.97%** at 1280. This motivated the larger training resolution at the cost of longer GPU inference.

## Training configuration

| Parameter | Recorded value |
|---|---|
| Model | YOLO11s detect, initialized from `yolo11s.pt` |
| Training environment | Kaggle, NVIDIA Tesla T4, CUDA device `0` |
| Dataset | Full official SKU-110K train / validation partitions |
| Epochs | 10 |
| Input image size | 1280 |
| Batch size | 2 |
| Workers / cache | 2 / disabled |
| Automatic mixed precision | Enabled |
| Mosaic / MixUp | 0.0 / 0.0 |
| Seed / deterministic flag | 42 / enabled |
| Validation during training | Enabled |
| Maximum detections | 750 |
| Checkpoints | `best.pt`, `last.pt`, period saves requested |
| Optimizer | Ultralytics `auto` |

The [exported arguments](training_artifacts/args.yaml) contain the remaining defaults and historical Kaggle output paths. Those paths are training-session records; the Modal service uses a separately packaged checkpoint path. The final `time` field in [`results.csv`](training_artifacts/results.csv) is **6,528.48 seconds** (about 1 hour 49 minutes) for this run.

## Validation results

These are the **final recorded epoch-10 validation metrics** on 588 images and 90,968 annotated product boxes. They are not scores for the official test split and do not measure brand grouping.

| Metric | Epoch 1 | Epoch 10 |
|---|---:|---:|
| Precision | 0.86824 | **0.90694** |
| Recall | 0.78932 | **0.87005** |
| mAP@50 | 0.86716 | **0.93357** |
| mAP@50–95 | 0.51624 | **0.59153** |

Epoch 10 has the highest recorded mAP@50–95 in the exported ten-epoch CSV. Rounded values used in the README are precision **0.907**, recall **0.870**, mAP@50 **0.934**, and mAP@50–95 **0.592**.

| Loss | Train, epoch 1 → 10 | Validation, epoch 1 → 10 |
|---|---:|---:|
| Box | 1.48490 → 1.29825 | 1.38315 → 1.27791 |
| Class | 0.90456 → 0.59957 | 0.77209 → 0.58128 |
| DFL | 1.20372 → 1.08531 | 1.13454 → 1.07420 |

The [training curves](training_artifacts/results.png) plot these losses and validation metrics across all ten epochs. Additional exported diagnostics are the [F1 curve](training_artifacts/BoxF1_curve.png), [precision–recall curve](training_artifacts/BoxPR_curve.png), and [confusion matrix](training_artifacts/confusion_matrix.png). The values above come from the CSV, not from reading pixels off the plots.

## Checkpoint and scope

The inference service uses [`weights/best.pt`](../weights/best.pt), whose SHA-256 is `696797b741735462c297c0d8d1a68e2393d9fddf4cccf0b016863350dc2ad8a6`. [`weights/last.pt`](../weights/last.pt) is retained for provenance. Neither checkpoint requires retraining to reproduce inference.

A single official test image was used as an end-to-end deployment smoke test, but the **full test split was not evaluated** and no held-out aggregate test metric is claimed. The appearance-grouping method uses pretrained DINOv2 embeddings and a hand-selected similarity threshold; it was not trained or objectively scored against brand labels, which SKU-110K does not provide.
