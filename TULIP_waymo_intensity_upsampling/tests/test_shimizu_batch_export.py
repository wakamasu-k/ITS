import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from waymo_preprocess.shimizu_batch_export import (
    ExportItem, completed_output, context_matches, group_by_tfrecord,
    load_export_items, select_pending, validate_frame_identity,
)

class ShimizuBatchExportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tfrecord = self.root / "training/segment-a.tfrecord"
        self.tfrecord.parent.mkdir()
        self.tfrecord.touch()

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self, rows):
        path = self.root / "frames.csv"
        fields = ("q_subset", "q_seg", "q_frame_index",
                  "frame_timestamp_micros", "source_tfrecord", "tfrecord_exists")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def row(self, index=0, timestamp=10):
        return {"q_subset": "training", "q_seg": "segment-a",
                "q_frame_index": str(index),
                "frame_timestamp_micros": str(timestamp),
                "source_tfrecord": str(self.tfrecord),
                "tfrecord_exists": "true"}

    def test_loads_and_limits_items(self):
        rows = load_export_items(
            self.manifest([self.row(0), self.row(1)]), max_frames=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].frame_index, 0)

    def test_rejects_duplicate_key(self):
        with self.assertRaisesRegex(ValueError, "duplicate frame key"):
            load_export_items(self.manifest([self.row(), self.row()]))

    def test_rejects_missing_tfrecord(self):
        row = self.row()
        row["tfrecord_exists"] = "false"
        with self.assertRaises(FileNotFoundError):
            load_export_items(self.manifest([row]))

    def test_normalizes_waymo_context_name(self):
        self.assertTrue(context_matches(
            "segment-167_1_2_with_camera_labels", "167_1_2"))
        self.assertFalse(context_matches(
            "segment-167_1_2_with_camera_labels", "999_1_2"))

    def test_validates_timestamp(self):
        item = ExportItem("training", "segment-a", 0, 10, self.tfrecord)
        with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
            validate_frame_identity(item, "segment-a", 11)

    def test_groups_frames_by_tfrecord(self):
        items = [
            ExportItem("training", "segment-a", 2, 12, self.tfrecord),
            ExportItem("training", "segment-a", 0, 10, self.tfrecord),
        ]
        grouped = group_by_tfrecord(items)
        self.assertEqual(
            [item.frame_index for item in grouped[self.tfrecord]], [0, 2])

    def test_resume_requires_matching_complete_output(self):
        item = ExportItem("training", "segment-a", 0, 10, self.tfrecord)
        output = self.root / "output"
        directory = output / "training/segment-a/000000"
        directory.mkdir(parents=True)
        (directory / "frame_64_32.npz").touch()
        metadata = directory / "metadata.json"
        metadata.write_text(json.dumps({
            "segment_id": "segment-a",
            "source_frame_index": 0,
            "timestamp_micros": 10,
        }), encoding="utf-8")
        self.assertTrue(completed_output(output, item))
        pending, skipped = select_pending([item], output, resume=True)
        self.assertEqual(pending, [])
        self.assertEqual(skipped, 1)

if __name__ == "__main__":
    unittest.main()
