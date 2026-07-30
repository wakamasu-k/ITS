import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from waymo_preprocess.handover_frame_manifest import build_rows, verify_rows

class HandoverFrameManifestTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.line32 = self.root / "32line"
        self.waymo = self.root / "waymo"
        (self.waymo / "training").mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def metadata(self, segment, index, timestamp):
        path = self.line32 / "datasets/pairs" / segment / str(index) / "cam.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"segment_id": segment, "frame_index": index,
                                    "timestamp_us": timestamp}), encoding="utf-8")
        return path

    def manifest(self, rows):
        path = self.root / "pairs.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["segment_id", "k", "cam_json"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_deduplicates_pair_rows(self):
        segment = "segment-example_with_camera_labels"
        metadata = self.metadata(segment, 7, 123456)
        stale = "/old/root/datasets/" + metadata.as_posix().split("/datasets/", 1)[1]
        manifest = self.manifest([
            {"segment_id": segment, "k": "0", "cam_json": stale},
            {"segment_id": segment, "k": "1", "cam_json": stale}])
        tfrecord = self.waymo / "training" / f"{segment}.tfrecord"
        tfrecord.touch()
        rows = build_rows(manifest, self.waymo, line32_root=self.line32)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["frame_index"], "7")
        self.assertEqual(rows[0]["timestamp_us"], "123456")
        self.assertEqual(rows[0]["source_manifest_rows"], "2")
        self.assertEqual(rows[0]["source_tfrecord"], str(tfrecord))

    def test_marks_missing_tfrecord_with_empty_path(self):
        segment = "segment-missing"
        metadata = self.metadata(segment, 0, 10)
        manifest = self.manifest([
            {"segment_id": segment, "k": "0", "cam_json": str(metadata)}])
        row = build_rows(manifest, self.waymo)[0]
        self.assertEqual(row["source_tfrecord"], "")
        self.assertEqual(row["status"], "missing_tfrecord")

    def test_rejects_segment_mismatch(self):
        metadata = self.metadata("segment-a", 0, 10)
        manifest = self.manifest([
            {"segment_id": "segment-b", "k": "0", "cam_json": str(metadata)}])
        with self.assertRaisesRegex(ValueError, "segment mismatch"):
            build_rows(manifest, self.waymo)

    def test_verifies_identity(self):
        row = {"segment_id": "segment-a", "frame_index": "1",
               "timestamp_us": "20", "source_tfrecord": str(self.root / "a.tfrecord"),
               "status": "metadata_matched"}
        verify_rows([row], lambda _: [(0, "segment-a", 10), (1, "segment-a", 20)])
        self.assertEqual(row["status"], "tfrecord_verified")

    def test_rejects_timestamp_mismatch(self):
        row = {"segment_id": "segment-a", "frame_index": "0",
               "timestamp_us": "20", "source_tfrecord": str(self.root / "a.tfrecord"),
               "status": "metadata_matched"}
        with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
            verify_rows([row], lambda _: [(0, "segment-a", 21)])

if __name__ == "__main__":
    unittest.main()
