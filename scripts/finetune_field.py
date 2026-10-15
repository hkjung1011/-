"""Additional field-camera fine-tuning after prepare_field_data.py has verified reviewed labels."""
from pathlib import Path
import argparse
import json
import hashlib
import math
import os
import shutil
import traceback
from finetune import ROOT, now, sha, save, metric_json

NAMES = ['edge_bite','gray_line','open','scratch','short_circuit','stains_enbedded']


def validate_field_dataset(dataset):
    """Check the complete prepared snapshot before importing GPU libraries.

    This validates files and human-supplied metadata; it does not establish a
    physical wafer's identity or confirm that a human actually reviewed labels.
    """
    from PIL import Image
    import numpy as np
    import yaml
    dataset = Path(dataset).resolve()
    audit_path, data = dataset/'audit.json', dataset/'data.yaml'
    audit = json.loads(audit_path.read_text(encoding='utf-8-sig'))
    if audit.get('status') != 'PASS_REVIEWED_FIELD_DATA' or sha(data) != audit.get('data_yaml_sha256'):
        raise RuntimeError('Reviewed field dataset audit required')
    configuration = yaml.safe_load(data.read_text(encoding='utf-8-sig'))
    if not isinstance(configuration, dict) or Path(configuration.get('path', '')).resolve() != dataset:
        raise RuntimeError('Field YAML must point to this audited dataset')
    names = configuration.get('names')
    if isinstance(names, dict):
        names = [names.get(i) for i in range(len(NAMES))]
    if names != NAMES or audit.get('names') != NAMES:
        raise RuntimeError('Field six-class names/order changed')
    for split in ('train','val','test'):
        if configuration.get(split) != 'images/' + split:
            raise RuntimeError('Field YAML split paths changed')
    audited = {}
    for item in audit.get('files', []):
        source = (dataset/item['path']).resolve()
        if not source.is_relative_to(dataset) or not source.is_file() or sha(source) != item['sha256']:
            raise RuntimeError('Field dataset changed after audit')
        relative = source.relative_to(dataset).as_posix()
        if relative in audited:
            raise RuntimeError('Duplicate audited file entry')
        audited[relative] = item['sha256']
    actual = {path.resolve().relative_to(dataset).as_posix()
              for kind in ('images','labels') for path in (dataset/kind).rglob('*') if path.is_file()}
    if actual != set(audited):
        raise RuntimeError('Unreviewed or missing image/label files after field audit')
    records = audit.get('records', [])
    if not records:
        raise RuntimeError('Reviewed field records required')
    expected, groups, decoded = set(), {}, set()
    counts = {split:{'images':0,'boxes':0,'normal_images':0,'classes':{name:0 for name in NAMES}}
              for split in ('train','val','test')}
    for record in records:
        if record.get('reviewed') is not True:
            raise RuntimeError('Every field record must be human-reviewed')
        split = record.get('split')
        wafer = str(record.get('wafer_id', '')).strip()
        if split not in counts or not wafer:
            raise RuntimeError('Explicit field split and physical wafer_id required')
        if wafer in groups and groups[wafer] != split:
            raise RuntimeError('Physical wafer crosses field dataset splits')
        groups[wafer] = split
        image = (dataset/record['image']).resolve()
        label = (dataset/record['label']).resolve()
        for path, kind in ((image,'images'),(label,'labels')):
            if not path.is_relative_to(dataset/kind/split):
                raise RuntimeError('Field record points outside its declared split')
            relative = path.relative_to(dataset).as_posix()
            if relative not in audited or relative in expected:
                raise RuntimeError('Duplicate or unaudited field record')
            expected.add(relative)
        if image.stem != label.stem or label.suffix != '.txt':
            raise RuntimeError('Field image/label pairing changed')
        with Image.open(image) as candidate:
            array = np.asarray(candidate.convert('RGB'))
        digest = hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()
        if digest in decoded or digest != record.get('decoded_sha256'):
            raise RuntimeError('Duplicate or changed decoded field image')
        decoded.add(digest)
        lines = [line for line in label.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if not lines and record.get('normal_verified') is not True:
            raise RuntimeError('Empty field label requires a human-confirmed normal image')
        for line in lines:
            try:
                values = [float(value) for value in line.split()]
            except ValueError as exc:
                raise RuntimeError('Malformed field label') from exc
            if len(values) != 5 or not all(math.isfinite(value) for value in values):
                raise RuntimeError('Malformed or nonfinite field label')
            cid,x,y,width,height = values
            if int(cid) != cid or not 0 <= cid < len(NAMES) or not (0 < width <= 1 and 0 < height <= 1
                and x-width/2 >= -1e-5 and y-height/2 >= -1e-5 and x+width/2 <= 1+1e-5 and y+height/2 <= 1+1e-5):
                raise RuntimeError('Out-of-range field class or bounding box')
            counts[split]['classes'][NAMES[int(cid)]] += 1
        counts[split]['images'] += 1
        counts[split]['boxes'] += len(lines)
        counts[split]['normal_images'] += int(not lines)
    if expected != set(audited):
        raise RuntimeError('Audited file is absent from reviewed field records')
    if any(count['images'] == 0 for count in counts.values()) or counts['train']['boxes'] == 0:
        raise RuntimeError('All field splits and training defect boxes are required')
    if counts != audit.get('counts') or groups != audit.get('wafer_split'):
        raise RuntimeError('Field split/count/wafer metadata changed')
    return audit


def record_epoch(trainer, history):
    """Do not count Ultralytics' final best-checkpoint evaluation as training."""
    if getattr(trainer, 'in_final_eval', False):
        return None
    values = {key:float(value) for key,value in trainer.metrics.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError('Nonfinite field epoch metric')
    row = {'epoch':trainer.epoch + 1,'optimizer_steps':trainer.field_steps,'validation':values}
    if history and row['epoch'] <= history[-1]['epoch']:
        raise RuntimeError('Duplicated or reversed field training epoch')
    history.append(row)
    return row

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--epochs',type=int,choices=(3,5,10),default=5)
    args=p.parse_args(); dataset=Path(args.dataset).resolve(); audit_path=dataset/'audit.json'; data=dataset/'data.yaml'
    audit=validate_field_dataset(dataset)
    initial=ROOT/'models/baseline.pt'
    registry=ROOT/'models/registry.json'
    if registry.exists():
        key=json.loads(registry.read_text(encoding='utf-8'))['default_model']
        if key not in ('baseline','finetuned','field_finetuned'):
            raise RuntimeError('Unknown field starting model in registry')
        initial=ROOT/'models'/(key+'.pt')
    protected={path:sha(path) for path in (ROOT/'models/baseline.pt',ROOT/'models/finetuned.pt') if path.is_file()}
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    import msvcrt
    lock_path=ROOT/'runtime/training.lock'; lock_path.parent.mkdir(parents=True,exist_ok=True)
    with open(lock_path,'a+b') as lock:
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            from datetime import datetime,timezone
            run_id='field_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
            evidence=ROOT/'runs'/run_id/'evidence'
            cfg=dict(data=str(data),epochs=args.epochs,imgsz=1024,batch=4,nbs=8,device=0,workers=0,
                optimizer='AdamW',lr0=.0002,lrf=.1,warmup_epochs=.5,warmup_bias_lr=0.,amp=False,
                seed=42,deterministic=True,mosaic=0.,mixup=0.,degrees=0.,translate=0.,scale=0.,
                fliplr=.5,flipud=.5,hsv_h=0.,hsv_s=0.,hsv_v=.1,patience=args.epochs+1,
                val=True,plots=True,project=str(ROOT/'runs'/run_id/'training'),name='model',exist_ok=False)
            save(evidence/'protocol.json',{'utc':now(),'configuration':cfg,'initial_sha256':sha(initial),
                'audit_sha256':sha(audit_path),'script_sha256':sha(__file__),'selection':'validation mAP50-95',
                'identity_basis':'Human-supplied physical wafer IDs','physical_device_validated':False})
            history=[]
            model=YOLO(str(initial))
            class FieldTrainer(DetectionTrainer):
                in_final_eval=False
                def optimizer_step(self):
                    if any(not torch.isfinite(param.grad).all() for param in self.model.parameters() if param.grad is not None):
                        raise RuntimeError('Nonfinite field gradient; stopped')
                    super().optimizer_step()
                def _handle_nan_recovery(self,epoch):
                    if not math.isfinite(float(self.fitness)):
                        raise RuntimeError('Nonfinite field validation fitness; stopped')
                    return False
                def final_eval(self):
                    self.in_final_eval=True
                    try:
                        super().final_eval()
                    finally:
                        self.in_final_eval=False
            def report(status,**extra):
                save(ROOT/'reports/training_status.json',{'status':status,'pid':os.getpid(),'run_id':run_id,
                    'updated_utc':now(),'epochs_requested':args.epochs,'dataset_kind':'field',
                    'physical_device_validated':False,**extra})
            def on_start(t):
                t.field_steps=0
                t.field_hook=t.optimizer.register_step_post_hook(lambda *x: setattr(t,'field_steps',t.field_steps+1))
            def on_epoch(t):
                row=record_epoch(t,history)
                if row is not None:
                    save(evidence/'epochs.json',history); report('TRAINING',**row)
            model.add_callback('on_pretrain_routine_end',on_start); model.add_callback('on_fit_epoch_end',on_epoch)
            report('TRAINING',epoch=0); model.train(trainer=FieldTrainer,**cfg)
            if len(history)!=args.epochs or getattr(model.trainer,'field_steps',0)<=0:
                raise RuntimeError('Field training epochs or actual optimizer steps are incomplete')
            best=Path(model.trainer.best).resolve()
            if any(sha(path)!=digest for path,digest in protected.items()):
                raise RuntimeError('Public baseline or fine-tuned model unexpectedly changed')
            save(evidence/'training_summary.json',{'status':'ACTUAL_FIELD_FINETUNING_COMPLETE',
                'epochs_completed':len(history),'optimizer_steps':model.trainer.field_steps,
                'best_checkpoint_sha256':sha(best),'utc':now(),'physical_device_validated':False})
            save(evidence/'selection_frozen.json',{'sha256':sha(best),'utc':now(),'selected_on':'validation only'})
            report('DEVELOPMENT_HOLDOUT_EVALUATION')
            evaluator=YOLO(str(best)); metrics=evaluator.val(data=str(data),split='test',imgsz=1024,batch=4,
                device=0,workers=0,half=False,conf=.001,iou=.7,max_det=300,plots=True,
                project=str(ROOT/'runs'/run_id/'evaluation'),name='test')
            save(evidence/'field_test_metrics.json',{'metrics':metric_json(metrics,evaluator.names),'audit_sha256':sha(audit_path),
                'checkpoint_sha256':sha(best),'physical_identity_independently_verified':False})
            if any(sha(path)!=digest for path,digest in protected.items()):
                raise RuntimeError('Public baseline or fine-tuned model unexpectedly changed')
            shutil.copy2(best,ROOT/'models/field_finetuned.pt')
            report('COMPLETE',field_model='field_finetuned.pt',epoch=args.epochs)
        except Exception as exc:
            save(ROOT/'reports/training_status.json',{'status':'FAILED','pid':os.getpid(),'utc':now(),'error':str(exc),'dataset_kind':'field'})
            traceback.print_exc(); raise
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)

if __name__=='__main__': main()
