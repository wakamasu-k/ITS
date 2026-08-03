import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from waymo_preprocess.training_pair_build import (
    TrainingFrame, completed_pair, index_row, load_training_frames, pair_paths,
)


class TrainingPairBuildTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tfrecord = self.root / "segment-a.tfrecord"
        self.tfrecord.touch()
        self.frame_npz = self.root / "frame_64_32.npz"
        self.frame_npz.touch()
        self.frame = TrainingFrame(
            "train", "training", "segment-a", 10, 123,
            self.tfrecord, self.frame_npz)
        self.index = self.root / "frames.csv"

    def tearDown(self):
        self.temp.cleanup()

    def write_index(self, rows):
        fields = (
            "dataset_role", "q_subset", "q_seg", "q_frame_index",
            "frame_timestamp_micros", "source_tfrecord", "frame_npz",
        )
        with self.index.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def row(self, role="train"):
        return {
            "dataset_role": role,
            "q_subset": "training",
            "q_seg": "segment-a",
            "q_frame_index": "10",
            "frame_timestamp_micros": "123",
            "source_tfrecord": str(self.tfrecord),
            "frame_npz": str(self.frame_npz),
        }

    def test_loads_role_without_relabeling(self):
        self.write_index([self.row("validation")])
        frames = load_training_frames(self.index)
        self.assertEqual(frames[0].dataset_role, "validation")
        self.assertEqual(frames[0].timestamp_micros, 123)

    def test_rejects_unknown_role(self):
        self.write_index([self.row("localization_evaluation")])
        with self.assertRaisesRegex(ValueError, "invalid dataset_role"):
            load_training_frames(self.index)

    def test_rejects_duplicate_role_segment_frame(self):
        self.write_index([self.row(), self.row()])
        with self.assertRaisesRegex(ValueError, "duplicate frame key"):
            load_training_frames(self.index)

    def test_resume_requires_role_and_full_identity(self):
        pair_npz, pair_json = pair_paths(self.frame)
        pair_npz.touch()
        pair_json.write_text(json.dumps({
            "dataset_role": "train",
            "subset": "training",
            "segment_id": "segment-a",
            "frame_index": 10,
            "timestamp_micros": 123,
            "task": "Waymo TOP LiDAR 32-to-64 range and intensity upsampling",
            "exact_subset_verified": True,
        }), encoding="utf-8")
        self.assertTrue(completed_pair(self.frame))
        wrong_role = TrainingFrame(
            "validation", "training", "segment-a", 10, 123,
            self.tfrecord, self.frame_npz)
        self.assertFalse(completed_pair(wrong_role))

    def test_index_row_is_dataset_compatible(self):
        row = index_row(self.frame, {
            "input_shape": [32, 2650, 2],
            "target_shape": [64, 2650, 2],
        })
        self.assertEqual(row["dataset_role"], "train")
        self.assertEqual(row["timestamp_micros"], "123")
        self.assertEqual(row["channels"], "2")
        self.assertEqual(row["generated_target_rows"], "1:64:2")


if __name__ == "__main__":
    unittest.main()
