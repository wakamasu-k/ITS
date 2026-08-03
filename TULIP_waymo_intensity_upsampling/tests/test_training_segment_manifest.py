import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from waymo_preprocess.training_segment_manifest import (
    discover_tfrecords, read_excluded_segments, split_segments,
    validate_no_overlap,
)


class TrainingSegmentManifestTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "training").mkdir()
        for name in ("segment-a", "segment-b", "segment-c", "segment-d",
                     "segment-e"):
            (self.root / "training" / f"{name}.tfrecord").touch()

    def tearDown(self):
        self.temp.cleanup()

    def test_excludes_evaluation_and_splits_by_segment(self):
        records = discover_tfrecords(self.root)
        rows, summary = split_segments(
            records, excluded_segments={"segment-a"}, seed=7,
            validation_fraction=0.25, test_fraction=0.25)
        self.assertNotIn("segment-a", {row["segment_id"] for row in rows})
        self.assertEqual(summary["excluded_evaluation_segments"], 1)
        self.assertEqual(summary["train_segments"], 2)
        self.assertEqual(summary["validation_segments"], 1)
        self.assertEqual(summary["test_segments"], 1)

    def test_seed_is_deterministic(self):
        records = discover_tfrecords(self.root)
        first, _ = split_segments(records, excluded_segments=set(), seed=19)
        second, _ = split_segments(reversed(records), excluded_segments=set(), seed=19)
        self.assertEqual(first, second)

    def test_overlap_validation_rejects_leakage(self):
        with self.assertRaisesRegex(ValueError, "evaluation leakage"):
            validate_no_overlap([{
                "segment_id": "segment-a", "dataset_role": "train"
            }], {"segment-a"})

    def test_reads_unique_excluded_segments(self):
        manifest = self.root / "evaluation.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("q_seg",))
            writer.writeheader()
            writer.writerows([{"q_seg": "segment-a"}, {"q_seg": "segment-a"}])
        self.assertEqual(read_excluded_segments(manifest), {"segment-a"})

    def test_max_segments_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            split_segments([], excluded_segments=set(), max_segments=0)


if __name__ == "__main__":
    unittest.main()
