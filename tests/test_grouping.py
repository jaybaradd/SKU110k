"""CPU-only behavior checks for appearance grouping."""

import math
import unittest

import numpy as np

from inference.grouping import assign_appearance_groups


class AppearanceGroupingTests(unittest.TestCase):
    def test_similar_products_share_groups_with_spatial_ids(self):
        # Detection order differs from shelf order. The upper group gets ID 0.
        embeddings = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
             [0.98, 0.20, 0.0], [0.0, 0.98, 0.20]],
            dtype=np.float32,
        )
        boxes = np.array(
            [[0, 20, 10, 30], [20, 0, 30, 10],
             [0, 40, 10, 50], [20, 30, 30, 40]],
            dtype=np.float32,
        )

        groups = assign_appearance_groups(embeddings, boxes)

        np.testing.assert_array_equal(groups, [1, 0, 1, 0])
        permutation = [3, 2, 0, 1]
        reordered = assign_appearance_groups(embeddings[permutation], boxes[permutation])
        np.testing.assert_array_equal(reordered, groups[permutation])

    def test_similarity_cutoff_changes_the_merge(self):
        embeddings = np.array([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)
        boxes = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=np.float32)

        np.testing.assert_array_equal(
            assign_appearance_groups(embeddings, boxes, similarity_threshold=0.72),
            [0, 0],
        )
        np.testing.assert_array_equal(
            assign_appearance_groups(embeddings, boxes, similarity_threshold=0.90),
            [0, 1],
        )

    def test_average_linkage_can_merge_a_three_product_chain(self):
        # Adjacent vectors are close, while the two ends are below the cutoff.
        # Average linkage still joins all three at 0.85 similarity.
        angles = [0, 20, 40]
        embeddings = np.array(
            [[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in angles],
            dtype=np.float32,
        )
        boxes = np.array(
            [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]],
            dtype=np.float32,
        )

        groups = assign_appearance_groups(embeddings, boxes, similarity_threshold=0.85)

        np.testing.assert_array_equal(groups, [0, 0, 0])

    def test_empty_singleton_and_zero_norm_feature(self):
        empty = assign_appearance_groups(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
        )
        self.assertEqual(empty.tolist(), [])

        one = assign_appearance_groups(
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[0, 0, 10, 10]], dtype=np.float32),
        )
        self.assertEqual(one.tolist(), [0])

        groups = assign_appearance_groups(
            np.array([[0, 0], [2, 0], [1, 0]], dtype=np.float32),
            np.array([[0, 0, 10, 10], [0, 20, 10, 30], [0, 40, 10, 50]]),
        )
        np.testing.assert_array_equal(groups, [0, 1, 1])

    def test_rejects_invalid_inputs(self):
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        boxes = np.array([[0, 0, 10, 10], [20, 0, 30, 10]])
        invalid_cases = [
            (embeddings[:1], boxes, 0.72),
            (np.array([[1.0, np.nan], [0.0, 1.0]]), boxes, 0.72),
            (embeddings, np.array([[0, 0, 10, 10], [20, 0, np.inf, 10]]), 0.72),
            (embeddings, boxes, 0.0),
            (embeddings, boxes, 1.1),
        ]

        for features, locations, threshold in invalid_cases:
            with self.subTest(features=features, boxes=locations, threshold=threshold):
                with self.assertRaises(ValueError):
                    assign_appearance_groups(features, locations, threshold)

    def test_does_not_modify_inputs(self):
        embeddings = np.array([[2.0, 0.0], [1.0, 0.1]], dtype=np.float32)
        boxes = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=np.float32)
        original_embeddings = embeddings.copy()
        original_boxes = boxes.copy()

        assign_appearance_groups(embeddings, boxes)

        np.testing.assert_array_equal(embeddings, original_embeddings)
        np.testing.assert_array_equal(boxes, original_boxes)


if __name__ == "__main__":
    unittest.main()
