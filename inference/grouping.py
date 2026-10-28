"""Group detected products by visual embedding similarity."""

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def assign_appearance_groups(
    embeddings: np.ndarray,
    boxes_xyxy: np.ndarray,
    similarity_threshold: float = 0.72,
) -> np.ndarray:
    """Return image-scoped group IDs aligned with the input detections.

    Embeddings are compared by cosine distance and clustered with average
    linkage. Group IDs follow the first appearance of each group in spatial
    top-to-bottom, then left-to-right order.
    """

    if isinstance(similarity_threshold, (bool, str)) or not isinstance(
        similarity_threshold, (int, float, np.integer, np.floating)
    ):
        raise ValueError("similarity_threshold must be a number in (0, 1]")
    threshold = float(similarity_threshold)
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("similarity_threshold must be in (0, 1]")

    try:
        features = np.asarray(embeddings, dtype=np.float32)
        boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("embeddings and boxes_xyxy must be numeric arrays") from exc

    if features.ndim == 1 and features.size == 0:
        features = features.reshape(0, 0)
    if boxes.ndim == 1 and boxes.size == 0:
        boxes = boxes.reshape(0, 4)
    if features.ndim != 2 or (len(features) > 0 and features.shape[1] == 0):
        raise ValueError("embeddings must have shape (N, D) with D > 0")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes_xyxy must have shape (N, 4)")
    if len(features) != len(boxes):
        raise ValueError("embedding and box counts must match")
    if not np.isfinite(features).all() or not np.isfinite(boxes).all():
        raise ValueError("embeddings and boxes_xyxy must contain finite values")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("boxes_xyxy must contain positive-area boxes")

    count = len(features)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == 1:
        return np.array([0], dtype=np.int64)

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.maximum(norms, 1e-12)
    similarity = normalized @ normalized.T
    distance = np.maximum(0.0, 1.0 - np.clip(similarity, -1.0, 1.0))
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)

    raw_labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - threshold,
        metric="precomputed",
        linkage="average",
    ).fit_predict(distance)

    center_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    spatial_order = np.lexsort((np.arange(count), center_x, center_y))
    id_by_cluster: dict[int, int] = {}
    for index in spatial_order:
        cluster = int(raw_labels[index])
        if cluster not in id_by_cluster:
            id_by_cluster[cluster] = len(id_by_cluster)

    return np.array(
        [id_by_cluster[int(cluster)] for cluster in raw_labels],
        dtype=np.int64,
    )
