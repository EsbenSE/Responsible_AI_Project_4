import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from derm7pt.cbm import (
    CONCEPTS,
    CONCEPT_COLUMNS,
    CONCEPT_LABELS,
    CONCEPT_IDS,
    DIAGNOSES,
    DIAGNOSIS_IDS,
    DIAGNOSIS_LABELS,
    _check_splits,
    _classification_metrics,
    _encode,
    build_cbm,
    build_concept_model,
)


class CbmTests(unittest.TestCase):
    def test_dataset_labels_map_to_grouped_class_ids(self):
        diagnosis = _encode(
            pd.Series(["basal cell carcinoma", "melanoma", "blue nevus"]),
            DIAGNOSES,
            DIAGNOSIS_IDS,
        )
        vascular = _encode(
            pd.Series(["absent", "dotted", "arborizing"]),
            CONCEPTS["vascular_structures"],
            CONCEPT_IDS["vascular_structures"],
        )

        np.testing.assert_array_equal(diagnosis, [0, 2, 1])
        np.testing.assert_array_equal(vascular, [0, 2, 1])

    def test_unknown_labels_fail(self):
        with self.assertRaisesRegex(ValueError, "Unknown labels"):
            _encode(pd.Series(["unknown"]), DIAGNOSES, DIAGNOSIS_IDS)

    def test_splits_must_be_disjoint_and_cover_all_rows(self):
        _check_splits(
            {"train": np.array([0, 1]), "valid": np.array([2]), "test": np.array([3])},
            4,
        )
        with self.assertRaisesRegex(ValueError, "disjoint and cover"):
            _check_splits(
                {
                    "train": np.array([0, 1]),
                    "valid": np.array([1]),
                    "test": np.array([2, 3]),
                },
                4,
            )

    def test_diagnosis_head_only_receives_concept_values(self):
        concepts = build_concept_model(32)
        cbm, diagnosis_head = build_cbm(concepts)
        outputs = cbm(tf.zeros((2, 32, 32, 3)), training=False)
        bottleneck_size = sum(len(CONCEPT_LABELS[name]) for name in CONCEPT_COLUMNS)

        self.assertEqual(diagnosis_head.input_shape[-1], bottleneck_size)
        self.assertEqual(tuple(outputs["diagnosis"].shape), (2, len(DIAGNOSIS_LABELS)))
        for name in CONCEPT_COLUMNS:
            self.assertEqual(
                tuple(outputs[f"concept_{name}"].shape),
                (2, len(CONCEPT_LABELS[name])),
            )

    def test_assessment_metrics(self):
        report = _classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ("a", "b"))

        self.assertEqual(report["accuracy"], 0.75)
        self.assertEqual(report["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(report["macro_f1"], (2 / 3 + 0.8) / 2)
        self.assertEqual(report["confusion_matrix"], [[1, 1], [0, 2]])


if __name__ == "__main__":
    unittest.main()
