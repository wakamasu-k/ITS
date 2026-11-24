import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset, load_pair


class Waymo16To32DatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pair = self.root / "pair.npz"
        width = 5
        input_16 = np.zeros((16, width, 2), dtype=np.float32)
        target_32 = np.zeros((32, width, 2), dtype=np.float32)
        input_16[..., 0] = 2.0
        input_16[..., 1] = 3.0
        target_32[..., 0] = 4.0
        target_32[..., 1] = 7.0
        np.savez_compressed(
            self.pair,
            input_16=input_16,
            input_valid_mask_16=np.ones((16, width), dtype=bool),
            target_32=target_32,
            target_valid_mask_32=np.ones((32, width), dtype=bool),
            ring_ids_16=np.arange(0, 64, 4, dtype=np.int32),
            ring_ids_32=np.arange(0, 64, 2, dtype=np.int32),
            observed_target_rows=np.arange(0, 32, 2),
            generated_target_rows=np.arange(1, 32, 2),
        )
        self.index = self.root / "index.csv"
        fields = ["dataset_role", "q_subset", "q_seg", "q_frame_index",
                  "timestamp_micros", "pair_npz", "input_height",
                  "target_height", "width", "channels"]
        with self.index.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                "dataset_role": "train", "q_subset": "training",
                "q_seg": "segment-a", "q_frame_index": "3",
                "timestamp_micros": "123", "pair_npz": str(self.pair),
                "input_height": "16", "target_height": "32",
                "width": "5", "channels": "2",
            })

    def tearDown(self):
        self.temp.cleanup()

    def test_padding_and_channel_first_shapes(self):
        sample = Waymo16To32Dataset(self.index, dataset_role="train")[0]
        self.assertEqual(tuple(sample["input"].shape), (2, 16, 2688))
        self.assertEqual(tuple(sample["target"].shape), (2, 32, 2688))
        self.assertEqual(sample["input"].dtype, __import__("torch").float32)
        self.assertEqual(sample["target"].dtype, __import__("torch").float32)
        self.assertEqual(tuple(sample["input_mask"].shape), (16, 2688))
        self.assertEqual(tuple(sample["generated_mask"].shape), (32, 2688))
        np.testing.assert_allclose(sample["input"][0, :, 2650:], 2.0)

    def test_log1p_only_changes_intensity(self):
        pair = load_pair(self.pair, intensity_transform="log1p")
        np.testing.assert_allclose(pair["input"][0], 2.0)
        np.testing.assert_allclose(pair["input"][1], np.log1p(3.0))
        np.testing.assert_allclose(pair["target"][0], 4.0)
        np.testing.assert_allclose(pair["target"][1], np.log1p(7.0))

    def test_range_scale_is_explicit(self):
        pair = load_pair(self.pair, range_scale_m=2.0)
        np.testing.assert_allclose(pair["input"][0], 1.0)
        np.testing.assert_allclose(pair["target"][0], 2.0)

    def test_invalid_rows_are_rejected(self):
        with np.load(self.pair, allow_pickle=False) as loaded:
            values = {name: loaded[name] for name in loaded.files}
        values["observed_target_rows"] = np.arange(0, 32, 2) + 1
        np.savez_compressed(self.pair, **values)
        with self.assertRaisesRegex(ValueError, "observed target rows"):
            load_pair(self.pair)


if __name__ == "__main__":
    unittest.main()
