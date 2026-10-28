# Appearance Grouping Experiments

## Goal and evidence

The grouping stage assigns the detector's product crops to **image-local appearance groups**. It does not output verified brands or product names. The comparisons below come from the [training notebook](../notebook/sku-110k-training.ipynb), primarily one SKU-110K validation shelf (`val_100.jpg`) with 144 detector predictions. The notebook's selected pairs were judged visually by a person; SKU-110K supplies product boxes but no brand labels. Pair counts here are diagnostic observations, **not** accuracy, precision, recall, or a representative evaluation of brand grouping.

## Crop features: CLIP and DINOv2

Both approaches used the same detected boxes. Each crop was clipped to the original image, centered on an RGB `(240, 240, 240)` square without stretching, transformed for its encoder, and L2-normalized before cosine comparison.

| Encoder tried | Feature size | Notebook observation |
|---|---:|---|
| OpenCLIP ViT-B/32 (`openai` weights) | 512 | Retrieved repeated products, but some high-similarity neighbors appeared to share only category or shape. |
| Pretrained `timm` DINOv2 ViT-S/14 (`vit_small_patch14_dinov2.lvd142m`) | 384 | Gave more useful packaging-detail comparisons on the inspected shelf, though it still made false merges and splits. |

For example, crop pair `17–75` had cosine similarity **0.803** with CLIP and **0.482** with DINOv2; pair `117–131` had **0.925** and **0.938**, respectively. These scores are on different feature spaces and their absolute values are not directly interchangeable. The notebook also compared both encoders at a matched **77 groups** using complete linkage: DINOv2 split members of several CLIP groups that had looked questionable on visual inspection. This motivated DINOv2 for the implemented grouping stage; it does not establish a general performance ranking between encoders.

Tesseract OCR was tried on eight small product crops after upscaling and contrast adjustment. Its output was mostly fragmented or incorrect packaging text, so OCR was not used to assign groups.

## Linkage comparison

The notebook compared complete and average linkage on DINOv2 cosine distances. To make this specific comparison, average linkage was forced to the same **84 groups** that complete linkage produced at threshold **0.70**. It was a matched-group-count comparison, **not** a comparison at the same threshold.

| Linkage in this comparison | Groups | Singletons | Largest group | Selected visually related pairs grouped | Selected visually different pairs merged |
|---|---:|---:|---:|---:|---:|
| Complete, threshold 0.70 | 84 | 45 | 5 | 0 of 7 | 0 of 3 |
| Average, fixed at 84 groups | 84 | 51 | 5 | 3 of 7 | 1 of 3 |

The seven related pairs were crop IDs `141–136`, `11–33`, `129–139`, `55–68`, `94–137`, `102–119`, and `82–98`; the three visually different pairs were `3–17`, `29–47`, and `14–56`. Average linkage recovered `129–139`, `55–68`, and `94–137`, while incorrectly merging `3–17`. Complete linkage kept all ten selected pairs separate. This tradeoff led to average linkage, with its threshold then examined separately.

## Average-linkage threshold sweep

For `val_100.jpg`, the notebook clustered the same 144 DINOv2 embeddings at each cosine-similarity cutoff below. The two pair-count columns refer only to the **seven selected related** and **three selected different** pairs above.

| Similarity cutoff | Groups | Singletons | Related pairs grouped, of 7 | Different pairs merged, of 3 | Largest group |
|---:|---:|---:|---:|---:|---:|
| 0.68 | 71 | 38 | 5 | 3 | 6 |
| 0.70 | 75 | 42 | 4 | 3 | 6 |
| **0.72** | **82** | **49** | **4** | **1** | **5** |
| 0.74 | 90 | 56 | 2 | 0 | 5 |
| 0.76 | 94 | 62 | 1 | 0 | 5 |
| 0.78 | 99 | 70 | 0 | 0 | 5 |
| 0.80 | 104 | 78 | 0 | 0 | 5 |
| 0.82 | 108 | 85 | 0 | 0 | 5 |

Higher cutoffs produced more, smaller groups and avoided more of the selected false merges, while also separating more visually related pairs. The chosen **0.72** is a qualitative compromise for this shelf and pair set, not an objectively optimal threshold. Do not confuse its **82 groups / 49 singletons** with the earlier complete-linkage **0.70** result of **84 groups / 45 singletons**.

## Implemented method and scope

[`inference/grouping.py`](../inference/grouping.py) L2-normalizes the 384-dimensional DINOv2 features, computes pairwise cosine distance as `max(0, 1 - similarity)`, and applies average-linkage agglomerative clustering with distance threshold `1 - 0.72 = 0.28`. It then renumbers clusters by the first product encountered in top-to-bottom, left-to-right order. IDs are stable for one result's spatial ordering; they are not global catalog IDs. Zero detections return no groups, and one detection returns group `0`.

The real Modal T4 development run returned **144 products and 82 groups** for [`val_100.jpg`](../outputs/examples/val_100_prediction.json), matching the notebook's final count. It also returned **136 products and 48 groups** for one held-out [`test_1904.jpg`](../outputs/examples/test_1904_prediction.json) smoke test. Neither result measures group correctness. A separate notebook check on `val_338.jpg` yielded 137 detections and 68 groups at threshold **0.70**. The notebook was executed out of order and redefined the grouping function before that cell, so this run is not evidence for the earlier complete-linkage configuration or the final **0.72** default.

The full official test split was not scored for grouping, and there is no labeled brand benchmark here. Similar packaging across different products can be merged; one product with different views, lighting, or package designs can be split. Pairwise distance storage and clustering also grow quadratically with the number of detections.
