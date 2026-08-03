import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_shimizu_32_to_64_pairs import pair_complete, index_row
from waymo_preprocess.shimizu_batch_export import ExportItem

class BuildShimizuPairsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.item = ExportItem(
            "training", "segment-a", 3, 123, self.root / "a.tfrecord")

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_pair_requires_identity_and_task(self):
        npz = self.root / "pair.npz"
        metadata = self.root / "pair.json"
        npz.touch()
        metadata.write_text(json.dumps({
            "segment_id": "segment-a", "frame_index": 3,
            "timestamp_micros": 123,
            "task": "Waymo TOP LiDAR 32-to-64 range and intensity upsampling",
            "exact_subset_verified": True,
        }), encoding="utf-8")
        self.assertTrue(pair_complete(npz, metadata, self.item))

    def test_complete_pair_rejects_wrong_timestamp(self):
        npz = self.root / "pair.npz"
        metadata = self.root / "pair.json"
        npz.touch()
        metadata.write_text(json.dumps({
            "segment_id": "segment-a", "frame_index": 3,
            "timestamp_micros": 999,
            "task": "Waymo TOP LiDAR 32-to-64 range and intensity upsampling",
            "exact_subset_verified": True,
        }), encoding="utf-8")
        self.assertFalse(pair_complete(npz, metadata, self.item))

    def test_index_marks_localization_evaluation_role(self):
        row = index_row(
            self.item, self.root / "pair.npz", self.root / "pair.json", {
                "input_shape": [32, 2650, 2],
                "target_shape": [64, 2650, 2],
            }, "localization_evaluation")
        self.assertEqual(row["dataset_role"], "localization_evaluation")
        self.assertEqual(row["input_height"], "32")
        self.assertEqual(row["target_height"], "64")
        self.assertEqual(row["generated_target_rows"], "1:64:2")

if __name__ == "__main__":
    unittest.main()
