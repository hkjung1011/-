"""Local wafer inspection UI. Run: python -m uvicorn app:app --host 127.0.0.1 --port 8893."""
from __future__ import annotations

import ctypes
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Inference must not install packages or contact the network automatically.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
for directory in (RUNTIME, RUNTIME / "inputs", RUNTIME / "results", RUNTIME / "training"):
    directory.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000
CLASS_KO = {
    "edge_bite": "가장자리 파손", "gray_line": "회색 선", "open": "단선",
    "scratch": "스크래치", "short_circuit": "단락", "stains_enbedded": "오염·매립 결함",
}
MODEL_FILES = {"baseline": ROOT / "models" / "baseline.pt", "finetuned": ROOT / "models" / "finetuned.pt",
               "field_finetuned": ROOT / "models" / "field_finetuned.pt"}
HISTORY_FILE = RUNTIME / "history.jsonl"
JOB_FILE = RUNTIME / "training" / "active.json"
_models: dict[str, tuple[tuple[int, int], Any]] = {}
_inference_lock = threading.Lock()
_history_lock = threading.Lock()
_training_lock = threading.Lock()
_training_process: subprocess.Popen | None = None
_training_stream: Any = None
_training_job: dict[str, Any] | None = None

app = FastAPI(title="Wafer Inspection Platform", version="R01")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
app.mount("/hardware", StaticFiles(directory=ROOT / "hardware", check_dir=False), name="hardware")
app.mount("/reports", StaticFiles(directory=ROOT / "reports", check_dir=False), name="reports")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"status": "UNREADABLE", "detail": f"{path.name}을 읽을 수 없습니다."}


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def pid_alive(pid: Any) -> bool:
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        if os.name == "nt":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.restype = ctypes.c_void_p
            handle = kernel.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            code = ctypes.c_ulong()
            try:
                return bool(kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))) and code.value == 259
            finally:
                kernel.CloseHandle(ctypes.c_void_p(handle))
        os.kill(pid, 0)
        return True
    except (ValueError, TypeError, OSError):
        return False


def _training_state_locked() -> dict[str, Any]:
    global _training_process, _training_stream, _training_job
    if _training_process is not None:
        result = _training_process.poll()
        if result is None:
            return {**(_training_job or {}), "active": True}
        finished = {**(_training_job or {}), "active": False, "finished_at": now(), "exit_code": result,
                    "state": "completed" if result == 0 else "failed"}
        write_json_atomic(RUNTIME / "training" / "last.json", finished)
        if JOB_FILE.exists() and read_json(JOB_FILE).get("job_id") == finished.get("job_id"):
            JOB_FILE.unlink()
        if _training_stream:
            _training_stream.close()
        _training_process = None
        _training_stream = None
        _training_job = None
        return finished
    report = read_json(ROOT / "reports" / "training_status.json")
    if isinstance(report, dict) and report.get("status") not in {"COMPLETE", "FAILED", "INTERRUPTED"} and pid_alive(report.get("pid")):
        return {"active": True, "state": "running", "pid": report["pid"], "external": True,
                "epochs": report.get("epochs_requested"), "report_status": report.get("status")}
    existing = read_json(JOB_FILE)
    if isinstance(existing, dict):
        pid = existing.get("pid") or existing.get("owner_pid")
        if pid_alive(pid):
            return {**existing, "active": True}
        finished = {**existing, "active": False, "state": "interrupted", "finished_at": now()}
        write_json_atomic(RUNTIME / "training" / "last.json", finished)
        JOB_FILE.unlink(missing_ok=True)
        return finished
    return read_json(RUNTIME / "training" / "last.json") or {"active": False, "state": "idle"}


def training_state() -> dict[str, Any]:
    with _training_lock:
        return _training_state_locked()


def sample_files() -> dict[str, Path]:
    samples: dict[str, Path] = {}
    for split in ("test", "val", "train"):
        folder = ROOT / "data" / "wafer" / "images" / split
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                token = uuid.uuid5(uuid.NAMESPACE_URL, path.relative_to(ROOT).as_posix()).hex
                samples[token] = path
                if len(samples) >= 72:
                    return samples
    return samples


def sample_path(sample_id: str) -> Path:
    path = sample_files().get(sample_id)
    if path is None:
        raise HTTPException(404, "샘플 이미지를 찾을 수 없습니다.")
    return path


def decode_image(payload: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(payload)) as candidate:
            width, height = candidate.size
            if width * height > MAX_PIXELS or max(width, height) > 12000:
                raise HTTPException(413, "이미지는 최대 4,000만 화소, 한 변 12,000픽셀까지 지원합니다.")
            if width < 8 or height < 8:
                raise HTTPException(422, "이미지 크기가 너무 작습니다.")
            candidate.load()
            return ImageOps.exif_transpose(candidate).convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "이미지 파일을 해석할 수 없습니다. JPG 또는 PNG를 확인해 주세요.") from exc


def load_model(model_id: str) -> Any:
    path = MODEL_FILES.get(model_id)
    if path is None:
        raise HTTPException(422, "지원하지 않는 모델입니다.")
    if not path.is_file():
        raise HTTPException(409, "선택한 모델 파일이 아직 준비되지 않았습니다.")
    stat = path.stat()
    fingerprint = (stat.st_mtime_ns, stat.st_size)
    cached = _models.get(model_id)
    if cached and cached[0] == fingerprint:
        return cached[1]
    try:
        from ultralytics import YOLO
        model = YOLO(str(path))
    except Exception as exc:
        raise HTTPException(503, "모델 로드에 실패했습니다. 설치 환경과 모델 파일을 확인해 주세요.") from exc
    _models[model_id] = (fingerprint, model)
    return model


def tile_positions(length: int, tile_size: int = 1024) -> list[int]:
    if length <= tile_size:
        return [0]
    return sorted(set([*range(0, length - tile_size + 1, round(tile_size * .8)), length - tile_size]))


def merge_boxes(boxes: list[dict], iou_threshold: float = .45) -> list[dict]:
    """Classwise NMS on global coordinates; retain the highest-confidence duplicate."""
    remaining = sorted(boxes, key=lambda box: box["confidence"], reverse=True)
    kept = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        retained = []
        ax1, ay1, ax2, ay2 = best["xyxy"]
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        for candidate in remaining:
            if candidate["class_id"] != best["class_id"]:
                retained.append(candidate)
                continue
            bx1, by1, bx2, by2 = candidate["xyxy"]
            intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
            union = area_a + max(0, bx2 - bx1) * max(0, by2 - by1) - intersection
            if not union or intersection / union <= iou_threshold:
                retained.append(candidate)
        remaining = retained
    return kept


def annotate_boxes(image: Image.Image, boxes: list[dict]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    size = max(12, round(max(image.size) / 90))
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except OSError:
        font = ImageFont.load_default(size=size)
    colors = ["#4baaf0", "#a382ef", "#f0af52", "#30bda4", "#ec729b", "#718fea"]
    for box in boxes:
        coordinates = box["xyxy"]
        color = colors[box["class_id"] % len(colors)]
        draw.rectangle(coordinates, outline=color, width=max(2, round(size / 5)))
        label = f"{box['class_name']} {box['confidence']:.2f}"
        x, y = coordinates[0], max(0, coordinates[1] - size - 8)
        bounds = draw.textbbox((x, y), label, font=font)
        draw.rectangle((bounds[0] - 2, bounds[1] - 2, bounds[2] + 3, bounds[3] + 3), fill=color)
        draw.text((x, y), label, font=font, fill="#ffffff")
    return annotated


def run_inference(payload: bytes, source: str, model_id: str, confidence: float, tiled: bool = False) -> dict[str, Any]:
    image = decode_image(payload)
    identifier = uuid.uuid4().hex
    positions = [(x, y) for y in tile_positions(image.height) for x in tile_positions(image.width)] if tiled else [(0, 0)]
    if len(positions) > 16:
        raise HTTPException(413, "타일 검사는 최대 16개 패치로 제한합니다. 관심 영역을 잘라서 다시 검사해 주세요.")
    # Ultralytics models are shared and their predictor is mutable. Serialize calls.
    with _inference_lock:
        model = load_model(model_id)
        started = time.perf_counter()
        try:
            boxes = []
            for offset_x, offset_y in positions:
                patch = image.crop((offset_x, offset_y, min(image.width, offset_x + 1024), min(image.height, offset_y + 1024))) if tiled else image
                result = model.predict(source=patch, conf=confidence, imgsz=1024, device="cpu", verbose=False)[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        class_name = str(result.names[class_id])
                        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                        boxes.append({"class_id": class_id, "class_name": class_name,
                                      "class_name_ko": CLASS_KO.get(class_name, class_name),
                                      "confidence": round(float(box.conf.item()), 4),
                                      "xyxy": [round(max(0, min(value, maximum)), 1) for value, maximum in
                                               ((x1 + offset_x, image.width), (y1 + offset_y, image.height),
                                                (x2 + offset_x, image.width), (y2 + offset_y, image.height))]})
            if tiled:
                boxes = merge_boxes(boxes)
                annotated_image = annotate_boxes(image, boxes)
            else:
                annotated_image = Image.fromarray(result.plot()[:, :, ::-1])
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            annotated_image.save(RUNTIME / "results" / f"{identifier}.jpg", quality=92)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(503, "추론에 실패했습니다. 모델 환경과 서버 로그를 확인해 주세요.") from exc
    image.save(RUNTIME / "inputs" / f"{identifier}.jpg", quality=92)
    record = {"id": identifier, "created_at": now(), "source": source, "model": model_id,
              "device": "cpu", "mode": "tile_prototype" if tiled else "full_frame", "tile_count": len(positions),
              "confidence_threshold": confidence, "width": image.width, "height": image.height,
              "latency_ms": latency_ms, "box_count": len(boxes), "boxes": boxes,
              "annotated_url": f"/api/results/{identifier}", "original_url": f"/api/inputs/{identifier}",
              "interpretation": "검출 결과는 학습 모델의 예측이며 실제 장치의 검사 성능 검증 결과가 아닙니다."}
    with _history_lock:
        with HISTORY_FILE.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
def status():
    reports = {name: read_json(ROOT / "reports" / f"{name}.json")
               for name in ("data_summary", "comparison", "training_status")}
    return {"name": "웨이퍼 결함 검사 플랫폼", "version": "R01", "inference_device": "cpu",
            "models": [{"id": key, "available": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0}
                       for key, path in MODEL_FILES.items()],
            "reports": reports, "training": training_state(), "hardware_status": "NOT VERIFIED",
            "class_names": CLASS_KO, "max_upload_mb": 20,
            "default_model": (read_json(ROOT / "models" / "registry.json") or {}).get("default_model", "baseline"),
            "evaluation_report_available": (ROOT / "reports" / "summary.html").is_file()}


@app.get("/api/samples")
def samples():
    return {"samples": [{"id": token, "name": path.name, "split": path.parent.name,
                         "url": f"/api/sample-image/{token}"} for token, path in sample_files().items()]}


@app.get("/api/sample-image/{sample_id}")
def sample_image(sample_id: str):
    return FileResponse(sample_path(sample_id))


@app.post("/api/infer")
async def infer(file: UploadFile | None = File(default=None), sample_id: str | None = Form(default=None),
                model: str = Form(default="baseline"), confidence: float = Form(default=0.25), tiled: bool = Form(default=False)):
    if not 0.01 <= confidence <= 0.95:
        raise HTTPException(422, "신뢰도 기준은 0.01~0.95 사이여야 합니다.")
    if bool(file) == bool(sample_id):
        raise HTTPException(422, "업로드 파일 또는 샘플 중 하나를 선택해 주세요.")
    if file:
        payload = await file.read(MAX_UPLOAD + 1)
        await file.close()
        source = re.split(r"[/\\]", file.filename or "upload")[-1][:160]
    else:
        path = sample_path(sample_id or "")
        if path.stat().st_size > MAX_UPLOAD:
            raise HTTPException(413, "파일은 최대 20MB까지 지원합니다.")
        payload = path.read_bytes()
        source = path.name
    if len(payload) > MAX_UPLOAD:
        raise HTTPException(413, "파일은 최대 20MB까지 지원합니다.")
    if not payload:
        raise HTTPException(422, "이미지 파일이 비어 있습니다.")
    # CPU inference must not block the ASGI event loop.
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(run_inference, payload, source, model, confidence, tiled)


def stored_image(identifier: str, category: str):
    if not re.fullmatch(r"[0-9a-f]{32}", identifier):
        raise HTTPException(404, "이미지를 찾을 수 없습니다.")
    path = RUNTIME / category / f"{identifier}.jpg"
    if not path.is_file():
        raise HTTPException(404, "이미지를 찾을 수 없습니다.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/results/{identifier}")
def result_image(identifier: str):
    return stored_image(identifier, "results")


@app.get("/api/inputs/{identifier}")
def input_image(identifier: str):
    return stored_image(identifier, "inputs")


@app.get("/api/history")
def history(limit: int = 30):
    if not 1 <= limit <= 200:
        raise HTTPException(422, "이력 조회 개수는 1~200 사이여야 합니다.")
    records = []
    if HISTORY_FILE.is_file():
        from collections import deque
        with _history_lock, HISTORY_FILE.open(encoding="utf-8") as stream:
            for line in deque(stream, maxlen=limit):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return {"history": list(reversed(records))}


class TrainRequest(BaseModel):
    epochs: int = Field(default=3)


@app.post("/api/training/start", status_code=202)
def start_training(request: TrainRequest):
    global _training_process, _training_stream, _training_job
    if request.epochs not in (3, 5, 10):
        raise HTTPException(422, "실험 횟수는 3, 5, 10 epoch 중 선택해 주세요.")
    script = ROOT / "scripts" / "finetune.py"
    if not script.is_file():
        raise HTTPException(409, "파인튜닝 스크립트가 아직 준비되지 않았습니다.")
    if not MODEL_FILES["baseline"].is_file():
        raise HTTPException(409, "기준 모델이 아직 준비되지 않았습니다.")
    with _training_lock:
        current = _training_state_locked()
        if current.get("active"):
            raise HTTPException(409, "이미 파인튜닝이 진행 중입니다.")
        job = {"job_id": uuid.uuid4().hex, "epochs": request.epochs, "started_at": now(),
               "owner_pid": os.getpid(), "state": "running", "active": True}
        # Exclusive creation also blocks a second app process.
        try:
            with JOB_FILE.open("x", encoding="utf-8") as stream:
                json.dump(job, stream, ensure_ascii=False)
        except FileExistsError as exc:
            raise HTTPException(409, "다른 프로세스에서 파인튜닝을 시작하고 있습니다.") from exc
        logfile = RUNTIME / "training" / f"{job['job_id']}.log"
        try:
            _training_stream = logfile.open("w", encoding="utf-8")
            environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            _training_process = subprocess.Popen([sys.executable, str(script), "--epochs", str(request.epochs)],
                cwd=str(ROOT), stdout=_training_stream, stderr=subprocess.STDOUT, shell=False,
                env=environment, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            job.update(pid=_training_process.pid, log_url=f"/api/training/log/{job['job_id']}")
            _training_job = job
            write_json_atomic(JOB_FILE, job)
        except OSError as exc:
            if _training_stream:
                _training_stream.close()
            _training_stream = None
            JOB_FILE.unlink(missing_ok=True)
            raise HTTPException(503, "학습 프로세스를 시작할 수 없습니다.") from exc
        return job


@app.get("/api/training/status")
def get_training_status():
    return {"job": training_state(), "report": read_json(ROOT / "reports" / "training_status.json")}


@app.get("/api/training/log/{job_id}")
def training_log(job_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(404, "학습 로그를 찾을 수 없습니다.")
    logfile = RUNTIME / "training" / f"{job_id}.log"
    if not logfile.is_file():
        raise HTTPException(404, "학습 로그를 찾을 수 없습니다.")
    with logfile.open("rb") as stream:
        stream.seek(max(0, logfile.stat().st_size - 12000))
        value = stream.read().decode("utf-8", errors="replace")
    return {"log": value}


class CameraRequest(BaseModel):
    camera_index: int = Field(default=0, ge=0, le=5)
    model: str = "baseline"
    confidence: float = Field(default=0.25, ge=0.01, le=0.95)
    tiled: bool = False


@app.post("/api/camera/capture")
def capture_camera(request: CameraRequest):
    import cv2
    # Explicit single-frame acquisition only; no motor, illumination or stage commands.
    capture = cv2.VideoCapture(request.camera_index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(request.camera_index)
    try:
        if not capture.isOpened():
            raise HTTPException(409, "카메라를 열 수 없습니다. 연결과 다른 앱의 점유 상태를 확인해 주세요.")
        frame = None
        for _ in range(5):
            received, candidate = capture.read()
            if received and candidate is not None:
                frame = candidate
        if frame is None:
            raise HTTPException(409, "카메라에서 프레임을 받지 못했습니다.")
        encoded, data = cv2.imencode(".jpg", frame)
        if not encoded:
            raise HTTPException(503, "카메라 프레임을 변환할 수 없습니다.")
    finally:
        capture.release()
    return run_inference(data.tobytes(), f"camera:{request.camera_index}", request.model, request.confidence, request.tiled)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
