
"""Smoke tests for the saved model and dataset downloader."""

import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd

from download_dataset import download


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "my_random_forest.joblib"
)

EXPECTED_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

LEAKAGE_COLUMNS = {"TWF", "HDF", "PWF", "OSF", "RNF"}


class TestProject(unittest.TestCase):

    def load_model_bundle(self):
        if not MODEL_PATH.exists():
            self.skipTest(
                "Saved model not found. Run python train.py first."
            )

        return joblib.load(MODEL_PATH)

    def make_sample_input(self):
        return pd.DataFrame([{
            "Type": "L",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 120,
        }])

    # TEST 1: Check saved model components
    def test_model_bundle(self):
        bundle = self.load_model_bundle()

        self.assertEqual(
            set(bundle.keys()),
            {"model", "threshold", "features", "feature_ranges"}
        )

        self.assertEqual(bundle["features"], EXPECTED_FEATURES)

        self.assertTrue(
            LEAKAGE_COLUMNS.isdisjoint(bundle["features"])
        )

        self.assertGreater(bundle["threshold"], 0)
        self.assertLess(bundle["threshold"], 1)

        self.assertEqual(
            set(bundle["feature_ranges"]),
            set(EXPECTED_FEATURES[1:])
        )

        # Check that every numerical range is valid
        for limits in bundle["feature_ranges"].values():
            self.assertLessEqual(
                limits["min"],
                limits["max"]
            )

    # TEST 2: Check prediction probabilities
    def test_prediction_probabilities(self):
        bundle = self.load_model_bundle()
        model = bundle["model"]

        sample = self.make_sample_input()

        probabilities = model.predict_proba(sample)

        self.assertEqual(probabilities.shape, (1, 2))

        self.assertTrue(np.all(np.isfinite(probabilities)))

        self.assertTrue(
            np.all((probabilities >= 0) & (probabilities <= 1))
        )

        self.assertAlmostEqual(
            probabilities[0].sum(),
            1.0,
            places=6
        )

        failure_probability = probabilities[0, 1]

        prediction = int(
            failure_probability >= bundle["threshold"]
        )

        self.assertIn(prediction, [0, 1])

    # TEST 3: Check that missing model inputs are rejected
    def test_missing_feature_rejected(self):
        bundle = self.load_model_bundle()
        model = bundle["model"]

        sample = self.make_sample_input()

        sample = sample.drop(columns=["Tool wear [min]"])

        with self.assertRaises((ValueError, KeyError)):
            model.predict_proba(sample)

    # TEST 4: Check dataset ZIP extraction
    def test_dataset_zip_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:

            archive_bytes = io.BytesIO()

            with zipfile.ZipFile(archive_bytes, "w") as archive:
                archive.writestr(
                    "example/ai4i2020.csv",
                    "Type,Machine failure\nL,0\nH,1\n"
                )

            fake_response = Mock()
            fake_response.content = archive_bytes.getvalue()
            fake_response.raise_for_status.return_value = None

            destination = (
                Path(tmp) / "data" / "ai4i2020.csv"
            )

            with patch(
                "download_dataset.requests.get",
                return_value=fake_response
            ):
                result = download(destination)

            self.assertEqual(result, destination)

            self.assertTrue(destination.exists())

            self.assertIn(
                "Machine failure",
                destination.read_text()
            )


if __name__ == "__main__":
    unittest.main()
