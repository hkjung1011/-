"""Prepare reviewed six-class camera YOLO data with explicit physical-wafer splits."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from PIL import Image
import numpy as np

NAMES = ['edge_bite','gray_line','open','scratch','short_circuit','stains_enbedded']
ROOT = Path(__file__).resolve().parents[1]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(inventory, output=None):
    inventory = Path(inventory).resolve()
    with inventory.open(encoding='utf-8-sig',newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Inventory has no reviewed images')
    wafer_split, decoded_split, checked = {}, {}, []
    counts = {s:{'images':0,'boxes':0,'normal_images':0,'classes':{n:0 for n in NAMES}} for s in ('train','val','test')}
    for i,row in enumerate(rows):
        if row.get('reviewed','').lower() != 'yes':
            raise ValueError(f'Row {i+2}: reviewed must be yes after human label verification')
        split = row.get('split','')
        if split not in counts:
            raise ValueError(f'Row {i+2}: explicit train/val/test split required')
        wafer = row.get('wafer_id','').strip()
        if not wafer:
            raise ValueError(f'Row {i+2}: physical wafer_id required')
        if wafer in wafer_split and wafer_split[wafer] != split:
            raise ValueError(f'Same physical wafer appears in two splits: {wafer}')
        wafer_split[wafer] = split
        image = (inventory.parent / row['image']).resolve()
        label = (inventory.parent / row['label']).resolve()
        with Image.open(image) as im:
            arr = np.asarray(im.convert('RGB'))
        digest = hashlib.sha256(str(arr.shape).encode()+arr.tobytes()).hexdigest()
        if digest in decoded_split:
            raise ValueError(f'Duplicate decoded image: {image.name}')
        decoded_split[digest] = split
        lines = [line for line in label.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if not lines and row.get('normal_verified','').lower() != 'yes':
            raise ValueError(f'{image.name}: empty label requires a human-confirmed normal image')
        for line in lines:
            vals = [float(v) for v in line.split()]
            if len(vals)!=5 or not all(math.isfinite(v) for v in vals):
                raise ValueError(f'Invalid label row: {label.name}')
            cid,x,y,w,h=vals
            if int(cid)!=cid or not 0<=cid<6 or not(0<w<=1 and 0<h<=1 and x-w/2>=-1e-5 and y-h/2>=-1e-5 and x+w/2<=1+1e-5 and y+h/2<=1+1e-5):
                raise ValueError(f'Out-of-range class or bounding box: {label.name}')
            counts[split]['classes'][NAMES[int(cid)]] += 1
        counts[split]['images'] += 1
        counts[split]['boxes'] += len(lines)
        counts[split]['normal_images'] += int(not lines)
        checked.append((row,image,label,arr,digest))
    if any(c['images']==0 for c in counts.values()):
        raise ValueError('Each split needs at least one separate physical wafer; do not split crops from one wafer')
    if counts['train']['boxes']==0:
        raise ValueError('Training split has no defect boxes')
    if output is None:
        output = ROOT/'data/field_sessions'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Output already exists; preserve previous dataset')
    output.mkdir(parents=True)
    files,records=[],[]
    for row,image,label,arr,digest in checked:
        split=row['split']; stem=digest[:20]
        target_image=output/'images'/split/(stem+'.png')
        target_label=output/'labels'/split/(stem+'.txt')
        target_image.parent.mkdir(parents=True,exist_ok=True); target_label.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(arr).save(target_image)
        target_label.write_text(label.read_text(encoding='utf-8-sig'),encoding='utf-8')
        for p in (target_image,target_label):
            files.append({'path':str(p.relative_to(output)),'sha256':sha(p)})
        records.append({'image':str(target_image.relative_to(output)),'label':str(target_label.relative_to(output)),
            'source_image':str(image),'source_image_sha256':sha(image),'decoded_sha256':digest,
            'wafer_id':row['wafer_id'],'lot_id':row.get('lot_id',''),'split':split,'reviewed':True,
            'normal_verified':row.get('normal_verified','').lower()=='yes'})
    data_yaml=output/'data.yaml'
    data_yaml.write_text('path: '+json.dumps(str(output))+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join('- '+n+'\n' for n in NAMES),encoding='utf-8')
    audit={'status':'PASS_REVIEWED_FIELD_DATA','utc':datetime.now(timezone.utc).isoformat(),
        'source_inventory_sha256':sha(inventory),'data_yaml_sha256':sha(data_yaml),'files':files,
        'counts':counts,'records':records,'names':NAMES,'wafer_split':wafer_split,
        'physical_identity_basis':'User-supplied wafer_id, not independently measured',
        'lot_independence':'NOT_VERIFIED' if any(not r['lot_id'] for r in records) else 'RECORDED_NOT_INDEPENDENTLY_VERIFIED'}
    (output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'dataset':str(output),'counts':counts,'physical_wafer_groups':len(wafer_split)}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--inventory',required=True); p.add_argument('--output')
    args=p.parse_args(); print(json.dumps(prepare(args.inventory,args.output),ensure_ascii=False,indent=2))
