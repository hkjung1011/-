"""API verification. Real inference runs only with WAFER_REAL_INFERENCE=1."""
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as platform


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(platform.app)

    def test_health_and_local_ui(self):
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["hardware_status"], "NOT VERIFIED")
        self.assertEqual(response.json()["inference_device"], "cpu")
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)

    def test_reject_ambiguous_or_missing_input(self):
        self.assertEqual(self.client.post("/api/infer").status_code, 422)
        samples = self.client.get("/api/samples").json()["samples"]
        if samples:
            response = self.client.post("/api/infer", data={"sample_id": samples[0]["id"]},
                                        files={"file": ("sample.jpg", b"invalid", "image/jpeg")})
            self.assertEqual(response.status_code, 422)

    def test_invalid_image_and_empty_file(self):
        for payload in (b"", b"this is not a raster image"):
            response = self.client.post("/api/infer", files={"file": ("input.jpg", payload, "image/jpeg")})
            self.assertEqual(response.status_code, 422)

    def test_oversized_file_rejected_before_decode(self):
        response = self.client.post("/api/infer", files={"file": ("large.jpg", b"x" * (platform.MAX_UPLOAD + 1), "image/jpeg")})
        self.assertEqual(response.status_code, 413)

    def test_no_arbitrary_paths(self):
        self.assertEqual(self.client.get("/api/results/not-a-uuid").status_code, 404)
        self.assertEqual(self.client.get("/api/sample-image/not-a-uuid").status_code, 404)
        self.assertEqual(self.client.get("/api/training/log/not-a-uuid").status_code, 404)
        self.assertEqual(self.client.get("/api/history?limit=1000").status_code, 422)

    def test_training_limit_and_concurrency(self):
        self.assertEqual(self.client.post("/api/training/start", json={"epochs": 100}).status_code, 422)
        baseline = Mock(spec=Path)
        baseline.is_file.return_value = True
        with patch.dict(platform.MODEL_FILES, {"baseline": baseline}):
            with patch.object(platform, "_training_state_locked", return_value={"active": True, "pid": 123}) as state:
                with patch.object(platform.subprocess, "Popen") as launch:
                    self.assertEqual(self.client.post("/api/training/start", json={"epochs": 3}).status_code, 409)
                    state.assert_called_once()
                    launch.assert_not_called()

    def test_camera_index_validation_without_access(self):
        self.assertEqual(self.client.post("/api/camera/capture", json={"camera_index": 999}).status_code, 422)

    def test_tile_bound_and_classwise_merge(self):
        self.assertEqual(platform.tile_positions(1024), [0])
        self.assertEqual(platform.tile_positions(2048), [0, 819, 1024])
        candidates = [
            {"class_id": 0, "confidence": .9, "xyxy": [10, 10, 100, 100]},
            {"class_id": 0, "confidence": .8, "xyxy": [11, 11, 100, 100]},
            {"class_id": 1, "confidence": .7, "xyxy": [10, 10, 100, 100]},
        ]
        merged = platform.merge_boxes(candidates)
        self.assertEqual(len(merged), 2)
        self.assertEqual({box["class_id"] for box in merged}, {0, 1})
        buffer = io.BytesIO()
        Image.new("RGB", (4096, 4096)).save(buffer, format="PNG")
        with self.assertRaises(platform.HTTPException) as context:
            platform.run_inference(buffer.getvalue(), "too-many-tiles.png", "baseline", .25, True)
        self.assertEqual(context.exception.status_code, 413)

    @unittest.skipUnless(os.environ.get("WAFER_REAL_INFERENCE") == "1", "Set WAFER_REAL_INFERENCE=1 for actual model inference")
    def test_actual_cpu_inference_history_and_images(self):
        samples = self.client.get("/api/samples").json()["samples"]
        self.assertTrue(samples, "Prepared dataset samples required")
        sample = samples[0]
        response = self.client.post("/api/infer", data={"sample_id": sample["id"], "model": "baseline", "confidence": .25})
        self.assertEqual(response.status_code, 200, response.text)
        record = response.json()
        self.assertEqual(record["device"], "cpu")
        self.assertEqual(record["mode"], "full_frame")
        self.assertGreater(record["latency_ms"], 0)
        self.assertEqual(record["box_count"], len(record["boxes"]))
        self.assertEqual(self.client.get(record["annotated_url"]).status_code, 200)
        self.assertEqual(self.client.get(record["original_url"]).status_code, 200)
        self.assertIn(record["id"], [row["id"] for row in self.client.get("/api/history").json()["history"]])
        payload = self.client.get(sample["url"]).content
        uploaded = self.client.post("/api/infer", data={"model": "baseline", "tiled": "true"},
                                    files={"file": ("../../outside.jpg", payload, "image/jpeg")})
        self.assertEqual(uploaded.status_code, 200, uploaded.text)
        self.assertEqual(uploaded.json()["source"], "outside.jpg")
        self.assertEqual(uploaded.json()["mode"], "tile_prototype")
        self.assertLessEqual(uploaded.json()["tile_count"], 16)
        for box in uploaded.json()["boxes"]:
            x1, y1, x2, y2 = box["xyxy"]
            self.assertTrue(0 <= x1 <= x2 <= uploaded.json()["width"])
            self.assertTrue(0 <= y1 <= y2 <= uploaded.json()["height"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
