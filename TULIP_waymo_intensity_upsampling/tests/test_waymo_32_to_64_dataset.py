import csv
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tulip_adapter.waymo_32_to_64_dataset import (
    Waymo32To64Dataset, load_pair,
)

class Waymo32To64DatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pair = self.root / "pair.npz"
        width = 5
        input_32 = np.zeros((32, width, 2), dtype=np.float32)
        target_64 = np.zeros((64, width, 2), dtype=np.float32)
        input_32[..., 0] = 2
        input_32[..., 1] = 3
        target_64[..., 0] = 4
        target_64[..., 1] = 7
        np.savez_compressed(
            self.pair, input_32=input_32,
            input_valid_mask_32=np.ones((32, width), dtype=bool),
            target_64=target_64,
            target_valid_mask_64=np.ones((64, width), dtype=bool),
            observed_target_rows=np.arange(0, 64, 2),
            generated_target_rows=np.arange(1, 64, 2))
        self.index = self.root / "index.csv"
        fields = ["dataset_role", "q_subset", "q_seg", "q_frame_index",
                  "timestamp_micros", "pair_npz", "input_height",
                  "target_height", "width", "channels"]
        with self.index.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                "dataset_role": "localization_evaluation",
                "q_subset": "training", "q_seg": "segment-a",
                "q_frame_index": "3", "timestamp_micros": "123",
                "pair_npz": str(self.pair), "input_height": "32",
                "target_height": "64", "width": "5", "channels": "2"})

    def tearDown(self):
        self.temp.cleanup()

    def test_loads_channel_first_tensors_and_generated_mask(self):
        dataset = Waymo32To64Dataset(
            self.index, dataset_role="localization_evaluation")
        sample = dataset[0]
        self.assertEqual(tuple(sample["input"].shape), (2, 32, 5))
        self.assertEqual(tuple(sample["target"].shape), (2, 64, 5))
        self.assertEqual(int(sample["generated_mask"].sum()), 32 * 5)
        self.assertFalse(bool(sample["generated_mask"][0].any()))
        self.assertTrue(bool(sample["generated_mask"][1].all()))
        self.assertEqual(sample["segment_id"], "segment-a")

    def test_refuses_evaluation_index_as_train(self):
        with self.assertRaisesRegex(ValueError, "no rows"):
            Waymo32To64Dataset(self.index, dataset_role="train")

    def test_log1p_transforms_only_intensity_channel(self):
        pair = load_pair(self.pair, "log1p")
        np.testing.assert_allclose(pair["input"][0], 2)
        np.testing.assert_allclose(pair["input"][1], np.log1p(3))
        np.testing.assert_allclose(pair["target"][0], 4)
        np.testing.assert_allclose(pair["target"][1], np.log1p(7))

if __name__ == "__main__":
    unittest.main()
