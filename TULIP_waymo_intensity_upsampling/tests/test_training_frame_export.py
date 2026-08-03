import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from waymo_preprocess.training_frame_export import (
    TrainingSegment, completed_training_output, load_training_segments,
    pending_frame_indices, selected_frame_indices, training_output_paths,
)


class TrainingFrameExportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.records = self.root / "records"
        self.records.mkdir()
        self.rows = []
        for role, count in (("train", 3), ("validation", 2), ("test", 2)):
            for index in range(count):
                segment = f"segment-{role}-{index}"
                path = self.records / f"{segment}.tfrecord"
                path.touch()
                self.rows.append({
                    "dataset_role": role,
                    "source_split": "training",
                    "segment_id": segment,
                    "source_tfrecord": str(path),
                })
        self.manifest = self.root / "segments.csv"
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)

    def tearDown(self):
        self.temp.cleanup()

    def test_role_limits_are_applied_independently(self):
        selected = load_training_segments(
            self.manifest, {"train": 2, "validation": 1, "test": 1}, set())
        self.assertEqual(
            [value.dataset_role for value in selected],
            ["train", "train", "validation", "test"])

    def test_rejects_evaluation_leakage_again(self):
        with self.assertRaisesRegex(ValueError, "evaluation leakage"):
            load_training_segments(
                self.manifest, {"train": 3, "validation": 2, "test": 2},
                {"segment-train-1"})

    def test_selects_fixed_stride_indices(self):
        self.assertEqual(
            selected_frame_indices(4, 10, 3), (3, 13, 23, 33))
        with self.assertRaisesRegex(ValueError, "positive"):
            selected_frame_indices(0, 10)

    def test_resume_requires_matching_role_and_identity(self):
        segment = TrainingSegment(
            "train", "training", "segment-a", self.records / "a.tfrecord")
        segment.tfrecord.touch()
        npz, metadata = training_output_paths(self.root / "out", segment, 10)
        npz.parent.mkdir(parents=True)
        npz.touch()
        metadata.write_text(json.dumps({
            "segment_id": "segment-a",
            "dataset_role": "train",
            "source_subset": "training",
            "source_frame_index": 10,
        }), encoding="utf-8")
        self.assertTrue(completed_training_output(
            self.root / "out", segment, 10))
        pending, skipped = pending_frame_indices(
            self.root / "out", segment, (0, 10), True)
        self.assertEqual(pending, [0])
        self.assertEqual(skipped, 1)

    def test_rejects_segment_in_multiple_roles(self):
        duplicate = dict(self.rows[0])
        duplicate["dataset_role"] = "validation"
        with self.manifest.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.rows[0].keys())
            writer.writerow(duplicate)
        with self.assertRaisesRegex(ValueError, "multiple roles"):
            load_training_segments(
                self.manifest, {"train": 3, "validation": 3, "test": 2}, set())


if __name__ == "__main__":
    unittest.main()
