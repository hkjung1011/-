"""Reproducible small, bounded wafer fine-tuning with validation-only deployment selection."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'runtime' / 'yolo_config'))
os.environ.setdefault('TQDM_DISABLE', '1')
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)

def validate_data():
    """Verify actual split files, label semantics and decoded duplicate leakage before training."""
    from PIL import Image
    import numpy as np
    import yaml
    classes = ['edge_bite', 'gray_line', 'open', 'scratch', 'short_circuit', 'stains_enbedded']
    dataset_root = (ROOT / 'data' / 'wafer').resolve()
    cfg = yaml.safe_load((dataset_root / 'data.yaml').read_text(encoding='utf-8-sig'))
    if Path(cfg['path']).resolve() != dataset_root or cfg.get('names') != classes:
        raise RuntimeError('Frozen data.yaml path or class mapping changed; use the separate field-data workflow')
    for split in ('train','val','test'):
        if cfg.get(split) != f'images/{split}':
            raise RuntimeError('Frozen split reference changed')
    frozen = json.loads((ROOT / 'audit' / 'copied_wafer_file_hashes.json').read_text(encoding='utf-8-sig'))
    frozen_hashes = {str((ROOT / row['path']).resolve()):row['sha256'] for row in frozen['files']}
    expected = {'train': (1472, 2418), 'val': (275, 480), 'test': (287, 496)}
    decoded_splits = {}
    files = []
    counts = {}
    for split in expected:
        images = sorted((ROOT / 'data' / 'wafer' / 'images' / split).glob('*.jpg'))
        labels = sorted((ROOT / 'data' / 'wafer' / 'labels' / split).glob('*.txt'))
        actual_paths = {str(p.resolve()) for kind in ('images','labels') for p in (dataset_root / kind / split).rglob('*') if p.is_file()}
        expected_paths = {p for p in frozen_hashes if Path(p).parent in ((dataset_root / 'images' / split),(dataset_root / 'labels' / split))}
        if actual_paths != expected_paths:
            raise RuntimeError(f'Unreviewed extra or missing split files: {split}')
        if {p.stem for p in images} != {p.stem for p in labels}:
            raise RuntimeError(f'Image/label pairing failure: {split}')
        total_boxes = 0
        class_counts = [0] * len(classes)
        for image in images:
            label = ROOT / 'data' / 'wafer' / 'labels' / split / (image.stem + '.txt')
            with Image.open(image) as im:
                arr = np.asarray(im.convert('RGB'))
                digest = hashlib.sha256(str(arr.shape).encode() + arr.tobytes()).hexdigest()
            prior = decoded_splits.get(digest)
            if prior and prior != split:
                raise RuntimeError(f'Cross-split decoded duplicate: {image.name}')
            decoded_splits[digest] = split
            for line in label.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():
                    continue
                parts = [float(x) for x in line.split()]
                if len(parts) != 5 or not all(math.isfinite(x) for x in parts):
                    raise RuntimeError(f'Malformed/nonfinite label: {label.name}')
                cid, x, y, w, h = parts
                if int(cid) != cid or not 0 <= cid < 6 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise RuntimeError(f'Invalid class/box: {label.name}')
                if x - w/2 < -1e-5 or y - h/2 < -1e-5 or x+w/2 > 1+1e-5 or y+h/2 > 1+1e-5:
                    raise RuntimeError(f'Box outside image: {label.name}')
                total_boxes += 1
                class_counts[int(cid)] += 1
            for path in (image,label):
                digest = sha(path)
                if digest != frozen_hashes[str(path.resolve())]:
                    raise RuntimeError(f'Frozen source file changed: {path.name}; use the field-data workflow')
                files.append({'path':str(path.relative_to(ROOT)), 'sha256':digest})
        if (len(images), total_boxes) != expected[split]:
            raise RuntimeError(f'Frozen split counts changed: {split}: {len(images)}, {total_boxes}')
        counts[split] = {'images': len(images), 'boxes': total_boxes, 'class_boxes': dict(zip(classes, class_counts))}
    return {'status': 'PASS_FILE_AND_LABEL_CHECK', 'utc': now(), 'counts': counts, 'files': files,
            'decoded_cross_split_duplicates': 0, 'physical_wafer_lot_independence': 'NOT_VERIFIED'}

def metric_json(metrics, names):
    return {'bbox_map50': float(metrics.box.map50), 'bbox_map50_95': float(metrics.box.map),
            'per_class': [{'class_id': int(cid), 'name': names[int(cid)], 'ap50': float(metrics.box.ap50[i]), 'ap50_95': float(metrics.box.ap[i])}
                          for i, cid in enumerate(metrics.box.ap_class_index)],
            'speed_ms': {k: float(v) for k, v in metrics.speed.items()}}

def run(epochs):
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for this bounded training profile')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    run_id = f'finetune_{stamp}_{epochs}ep'
    run_dir = ROOT / 'runs' / run_id
    evidence = run_dir / 'evidence'
    baseline = ROOT / 'models' / 'baseline.pt'
    if not baseline.is_file():
        raise RuntimeError('Missing models/baseline.pt')
    baseline_digest = sha(baseline)
    started = time.monotonic()
    history = []

    def status(state, **extra):
        save(ROOT / 'reports' / 'training_status.json', {'status': state, 'pid': os.getpid(), 'run_id': run_id,
             'updated_utc': now(), 'epochs_requested': epochs, 'elapsed_seconds': round(time.monotonic()-started, 2), **extra})
    status('PREFLIGHT')
    audit = validate_data()
    save(evidence / 'preflight.json', audit)
    data = ROOT / 'data' / 'wafer' / 'data.yaml'
    config = dict(data=str(data), epochs=epochs, batch=4, imgsz=1024, nbs=8, device=0, workers=0,
        optimizer='AdamW', lr0=0.0002, lrf=0.1, warmup_epochs=0.5, warmup_bias_lr=0.0,
        amp=False, seed=42, deterministic=True, patience=epochs+1, cache=False,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0, translate=0.0, scale=0.0,
        shear=0.0, perspective=0.0, fliplr=0.5, flipud=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.1,
        box=7.5, cls=0.5, dfl=1.5, cls_pw=0.0, conf=0.001, iou=0.7, max_det=300,
        val=True, plots=True, save=True, save_period=1, project=str(run_dir / 'training'), name='model', exist_ok=False)
    save(evidence / 'protocol.json', {'created_utc': now(), 'initial_weights_sha256': baseline_digest,
        'configuration': config, 'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
        'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda, 'script_sha256': sha(__file__),
        'data_yaml_sha256': sha(data), 'preflight_sha256': sha(evidence / 'preflight.json'),
        'checkpoint_selection': 'validation mAP50-95 only', 'deployment_selection': 'candidate validation mAP50-95 >= baseline validation mAP50-95',
        'holdout_policy': 'Previously examined R01 test split; development holdout, not a newly untouched physical wafer test',
        'new_human_labels': 0, 'carinthia_used_for_this_training': False,
        'physical_device_validated': False, 'normal_only_false_alarm_validated': False})

    def evaluate(weight, split, name):
        model = YOLO(str(weight))
        metrics = model.val(data=str(data), split=split, imgsz=1024, batch=4, device=0, workers=0,
            half=False, conf=0.001, iou=0.7, max_det=300, plots=True, save_json=True,
            project=str(run_dir / 'evaluation'), name=name, exist_ok=False)
        return metric_json(metrics, model.names)

    status('BASELINE_VALIDATION')
    baseline_val = evaluate(baseline, 'val', 'baseline_val')
    save(evidence / 'baseline_val.json', baseline_val)

    def parameter_hash(model):
        h = hashlib.sha256()
        for name, param in unwrap_model(model).named_parameters():
            h.update(name.encode())
            h.update(param.detach().cpu().float().contiguous().numpy().tobytes())
        return h.hexdigest()

    class AuditedTrainer(DetectionTrainer):
        in_final_eval = False
        def optimizer_step(self):
            if any(not torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None):
                raise RuntimeError('Nonfinite gradient; stopped')
            super().optimizer_step()
        def _handle_nan_recovery(self, epoch):
            if not math.isfinite(float(self.fitness)):
                raise RuntimeError('Nonfinite validation fitness; stopped')
            return False
        def final_eval(self):
            self.in_final_eval = True
            try:
                super().final_eval()
            finally:
                self.in_final_eval = False

    def on_start(trainer):
        trainer.direct_steps = 0
        trainer.start_parameter_sha = parameter_hash(trainer.model)
        def count(*args):
            trainer.direct_steps += 1
        trainer.step_hook = trainer.optimizer.register_step_post_hook(count)
        torch.cuda.reset_peak_memory_stats()
        status('TRAINING', epoch=0, optimizer_steps=0)
    def on_epoch(trainer):
        if trainer.in_final_eval:
            return
        vals = {k: float(v) for k, v in trainer.metrics.items()}
        if not all(math.isfinite(v) for v in vals.values()):
            raise RuntimeError('Nonfinite epoch metric')
        row = {'epoch': trainer.epoch+1, 'utc': now(), 'optimizer_steps': trainer.direct_steps, 'validation': vals}
        history.append(row)
        save(evidence / 'epochs.json', history)
        status('TRAINING', epoch=trainer.epoch+1, optimizer_steps=trainer.direct_steps, validation=vals)
        print(f'PLATFORM_EPOCH {trainer.epoch+1}/{epochs} steps={trainer.direct_steps} mAP50_95={vals.get("metrics/mAP50-95(B)")}', flush=True)
    def on_end(trainer):
        final_digest = parameter_hash(trainer.model)
        if trainer.direct_steps <= 0 or final_digest == trainer.start_parameter_sha:
            raise RuntimeError('No actual optimizer updates')
        save(evidence / 'training_summary.json', {'status': 'ACTUAL_FINETUNING_COMPLETE', 'epochs_completed': len(history),
            'optimizer_steps': trainer.direct_steps, 'initial_parameter_sha256': trainer.start_parameter_sha,
            'final_parameter_sha256': final_digest, 'parameters_changed': True,
            'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'best_checkpoint_sha256': sha(trainer.best), 'utc': now()})

    model = YOLO(str(baseline))
    model.add_callback('on_pretrain_routine_end', on_start)
    model.add_callback('on_fit_epoch_end', on_epoch)
    model.add_callback('on_train_end', on_end)
    model.train(trainer=AuditedTrainer, **config)
    candidate = Path(model.trainer.best).resolve()
    candidate_digest = sha(candidate)
    save(evidence / 'selection_frozen.json', {'utc': now(), 'checkpoint_sha256': candidate_digest, 'selected_on': 'validation only'})
    status('CANDIDATE_VALIDATION', epoch=len(history))
    candidate_val = evaluate(candidate, 'val', 'candidate_val')
    deploy = 'finetuned' if candidate_val['bbox_map50_95'] >= baseline_val['bbox_map50_95'] else 'baseline'
    save(evidence / 'deployment_frozen.json', {'utc': now(), 'selected_model': deploy, 'selected_on': 'validation only'})
    status('DEVELOPMENT_HOLDOUT_EVALUATION', epoch=len(history))
    baseline_test = evaluate(baseline, 'test', 'baseline_test')
    candidate_test = evaluate(candidate, 'test', 'candidate_test')
    shutil.copy2(candidate, ROOT / 'models' / 'finetuned.pt')
    if sha(baseline) != baseline_digest:
        raise RuntimeError('Baseline unexpectedly changed')
    comparison = {'status': 'PUBLIC_IMAGE_DEVELOPMENT_EVALUATED', 'utc': now(), 'run_id': run_id,
        'epochs': len(history), 'default_model': deploy, 'selection_basis': 'validation mAP50-95 only',
        'training_output_dir': str(candidate.parent.parent),
        'baseline': {'sha256': baseline_digest, 'validation': baseline_val, 'test': baseline_test},
        'finetuned': {'sha256': candidate_digest, 'validation': candidate_val, 'test': candidate_test},
        'test_delta_map50_points': (candidate_test['bbox_map50']-baseline_test['bbox_map50'])*100,
        'test_delta_map50_95_points': (candidate_test['bbox_map50_95']-baseline_test['bbox_map50_95'])*100,
        'holdout_note': 'R01 already examined this test split; development evaluation, not an untouched external test',
        'physical_device_validated': False, 'normal_only_false_alarm_validated': False,
        'semantic_scope': 'Six-class Wafer-Datas microscopy bounding boxes; Carinthia-S remains a separate SEM segmentation dataset'}
    save(evidence / 'comparison.json', comparison)
    save(ROOT / 'reports' / 'comparison.json', comparison)
    save(ROOT / 'models' / 'registry.json', {'default_model': deploy, 'baseline_sha256': baseline_digest,
         'finetuned_sha256': candidate_digest, 'run_id': run_id, 'selected_on': 'validation only'})
    status('COMPLETE', epoch=len(history), default_model=deploy,
        test_delta_map50_points=comparison['test_delta_map50_points'],
        test_delta_map50_95_points=comparison['test_delta_map50_95_points'])
    print('PLATFORM_FINETUNE_COMPLETE ' + json.dumps({k: comparison[k] for k in ('default_model', 'test_delta_map50_points', 'test_delta_map50_95_points')}), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, choices=(3,5,10), default=5)
    args = parser.parse_args()
    lock_path = ROOT / 'runtime' / 'training.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, 'a+b') as lock:
        lock.seek(0)
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
        import msvcrt
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError('Another training job holds the GPU lock')
        try:
            run(args.epochs)
        except Exception as exc:
            save(ROOT / 'reports' / 'training_status.json', {'status': 'FAILED', 'pid': os.getpid(), 'utc': now(), 'error': str(exc)})
            traceback.print_exc()
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)

if __name__ == '__main__':
    main()
