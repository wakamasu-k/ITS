import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from waymo_preprocess.shimizu_frame_manifest import build_frame_rows

class ShimizuFrameManifestTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.line32 = self.root / "32line"
        self.waymo = self.root / "waymo"

    def tearDown(self):
        self.temp.cleanup()

    def camera_json(self, subset, segment, index, timestamp):
        path = (self.line32 / "datasets/cam_gray/cross/q" / subset /
                segment / "FRONT" / f"{index:05d}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "frame_index": index,
            "frame_timestamp_micros": timestamp,
        }), encoding="utf-8")
        return path

    def manifest(self, rows):
        path = self.root / "gt.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "q_subset", "q_seg", "q_frame_index", "q_meta_json", "anchor_id"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_deduplicates_by_query_identity_and_reads_timestamp(self):
        subset, segment = "training", "segment-a"
        metadata = self.camera_json(subset, segment, 32, 123456)
        stale = "/old/data/cam_gray/" + metadata.as_posix().split("/cam_gray/", 1)[1]
        manifest = self.manifest([
            {"q_subset": subset, "q_seg": segment, "q_frame_index": "32",
             "q_meta_json": stale, "anchor_id": "0"},
            {"q_subset": subset, "q_seg": segment, "q_frame_index": "32",
             "q_meta_json": stale, "anchor_id": "1"}])
        tfrecord = self.waymo / subset / f"{segment}.tfrecord"
        tfrecord.parent.mkdir(parents=True)
        tfrecord.touch()
        rows = build_frame_rows(manifest, self.line32, self.waymo)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["frame_timestamp_micros"], "123456")
        self.assertEqual(rows[0]["source_manifest_rows"], "2")
        self.assertEqual(rows[0]["tfrecord_exists"], "true")

    def test_reports_progress_and_caches_repeated_metadata(self):
        metadata = self.camera_json("training", "segment-a", 0, 10)
        manifest = self.manifest([
            {"q_subset": "training", "q_seg": "segment-a", "q_frame_index": "0",
             "q_meta_json": str(metadata), "anchor_id": str(anchor)}
            for anchor in range(3)])
        reports = []
        rows = build_frame_rows(
            manifest, self.line32, self.waymo, progress_every=2,
            progress=lambda source_rows, unique_frames:
                reports.append((source_rows, unique_frames)))
        self.assertEqual(reports, [(2, 1)])
        self.assertEqual(rows[0]["source_manifest_rows"], "3")

    def test_subset_is_part_of_deduplication_key(self):
        rows = []
        for subset, timestamp in (("training", 10), ("validation", 20)):
            metadata = self.camera_json(subset, "segment-a", 0, timestamp)
            rows.append({"q_subset": subset, "q_seg": "segment-a",
                         "q_frame_index": "0", "q_meta_json": str(metadata),
                         "anchor_id": "0"})
        result = build_frame_rows(self.manifest(rows), self.line32, self.waymo)
        self.assertEqual(len(result), 2)

    def test_rejects_json_frame_index_mismatch(self):
        metadata = self.camera_json("training", "segment-a", 4, 10)
        manifest = self.manifest([{
            "q_subset": "training", "q_seg": "segment-a", "q_frame_index": "3",
            "q_meta_json": str(metadata), "anchor_id": "0"}])
        with self.assertRaisesRegex(ValueError, "frame index mismatch"):
            build_frame_rows(manifest, self.line32, self.waymo)

    def test_reports_missing_tfrecord(self):
        metadata = self.camera_json("testing", "segment-a", 0, 10)
        manifest = self.manifest([{
            "q_subset": "testing", "q_seg": "segment-a", "q_frame_index": "0",
            "q_meta_json": str(metadata), "anchor_id": "0"}])
        row = build_frame_rows(manifest, self.line32, self.waymo)[0]
        self.assertEqual(row["tfrecord_exists"], "false")

if __name__ == "__main__":
    unittest.main()
