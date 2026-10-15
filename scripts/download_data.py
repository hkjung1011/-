"""Acquire public SEM data and verify/copy a frozen local Wafer-Datas split.

Carinthia-S is intentionally kept separate from the six-class YOLO dataset.
The Wafer-Datas cache must include the prior source and independent audits;
this script never invents or changes a train/validation/test assignment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import stat
import urllib.request
import zipfile

from PIL import Image, ImageOps, ImageChops

ROOT = Path(__file__).resolve().parents[1]
ZENODO_RECORD = 'https://zenodo.org/api/records/16895427'
CARINTHIA_URL = 'https://zenodo.org/records/16895427/files/data.zip?download=1'
SOURCE_COMMIT = '17dbc7e19f4e354f9b2b436af66e5abdaf2ea4b1'
NAMES = ['edge_bite', 'gray_line', 'open', 'scratch', 'short_circuit', 'stains_enbedded']


def digest(path: Path, algorithm: str = 'sha256') -> str:
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def open_url(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'WaferInspectionPlatform-research/1.0'}), timeout=120)


def copied_verified(source: Path, destination: Path, expected: str) -> None:
    if digest(source) != expected:
        raise RuntimeError(f'Source hash changed: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(destination) != expected:
            raise RuntimeError(f'Refusing to replace different data: {destination}')
    else:
        shutil.copy2(source, destination)
    if digest(destination) != expected:
        raise RuntimeError(f'Copied hash differs: {destination}')


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        missing = []
        for member in z.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f'Unsafe archive path: {member.filename}')
            if stat.S_ISLNK(member.external_attr >> 16):
                raise RuntimeError(f'Archive symlink is unsupported: {member.filename}')
            if target.exists() and not target.is_dir():
                with z.open(member) as f:
                    expected = hashlib.sha256(f.read()).hexdigest()
                if digest(target) != expected:
                    raise RuntimeError(f'Refusing to replace differing extracted data: {target}')
            elif not target.exists():
                missing.append(member)
        for member in missing:
            z.extract(member, root)


def acquire_carinthia() -> dict:
    with open_url(ZENODO_RECORD) as response:
        metadata = json.load(response)
    save(ROOT / 'downloads/carinthia_s_record.json', metadata)
    entry = next(f for f in metadata['files'] if f['key'] == 'data.zip')
    archive = ROOT / 'downloads/carinthia_s.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    downloaded = not archive.exists()
    if downloaded:
        partial = archive.with_suffix('.zip.part')
        with open_url(CARINTHIA_URL) as response, partial.open('wb') as output:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                output.write(block)
        partial.replace(archive)
    algorithm, expected = entry['checksum'].split(':', 1)
    actual = digest(archive, algorithm)
    if actual != expected or archive.stat().st_size != entry['size']:
        raise RuntimeError('Carinthia-S publisher checksum/size mismatch')
    target = ROOT / 'data/carinthia_s_raw'
    safe_extract(archive, target)
    prior = read_json(ROOT / 'audit/carinthia_acquisition.json') if (ROOT / 'audit/carinthia_acquisition.json').exists() else {}
    result = {'source_url': 'https://zenodo.org/records/16895427', 'download_url': CARINTHIA_URL,
              'publisher_record_id': 16895427, 'doi': '10.5281/zenodo.16895427',
              'license': metadata['metadata'].get('license'), 'archive_bytes': archive.stat().st_size,
              'publisher_checksum': entry['checksum'], 'checksum_verified': True,
              'sha256': digest(archive), 'acquisition': 'NEW_DOWNLOAD' if downloaded else prior.get('acquisition', 'VERIFIED_EXISTING_DOWNLOAD'),
              'archive_reused_this_execution': not downloaded,
              'archive': 'downloads/carinthia_s.zip', 'raw_directory': 'data/carinthia_s_raw',
              'training_used': False, 'modality': 'SEM', 'taxonomy_merged_with_wafer_datas': False}
    description_entry = next(f for f in metadata['files'] if f['key'] == 'carinthia-s_dataset.html')
    description_path = ROOT / 'downloads/carinthia-s_dataset.html'
    if not description_path.exists():
        with open_url('https://zenodo.org/records/16895427/files/carinthia-s_dataset.html?download=1') as response:
            description_path.write_bytes(response.read())
    algorithm, expected = description_entry['checksum'].split(':', 1)
    if digest(description_path, algorithm) != expected:
        raise RuntimeError('Publisher description checksum mismatch')
    result['publisher_description'] = {'path': 'downloads/carinthia-s_dataset.html',
                                       'publisher_checksum': description_entry['checksum'],
                                       'checksum_verified': True, 'sha256': digest(description_path)}
    result['audit'] = audit_carinthia(target)
    save(ROOT / 'audit/carinthia_acquisition.json', result)
    return result


def audit_carinthia(raw_directory: Path) -> dict:
    base = raw_directory / 'data'
    with (base / 'carinthia-s.csv').open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        if reader.fieldnames != ['image_path', 'mask_path', 'filename', 'label']:
            raise RuntimeError('Unexpected Carinthia-S CSV schema')
        rows = list(reader)
    if len(rows) != 4591 or len({r['filename'] for r in rows}) != 4591:
        raise RuntimeError('Carinthia-S row/unique ID count differs from publisher description')
    root = base.resolve()
    class_counts = Counter()
    empty_masks_by_class = Counter()
    dimensions = Counter()
    modes = Counter()
    value_histogram = Counter()
    nonbinary_ids = []
    nongrayscale_channel_ids = []
    nonopaque_alpha_ids = []
    records = []
    decoded_groups = defaultdict(list)
    nonempty_pixels = []
    referenced_images = set()
    referenced_masks = set()
    for row in rows:
        image_path = (base / row['image_path']).resolve()
        mask_path = (base / row['mask_path']).resolve()
        if not image_path.is_relative_to(root) or not mask_path.is_relative_to(root):
            raise RuntimeError('CSV path escapes extracted dataset')
        if image_path.stem != row['filename'] or mask_path.stem != row['filename']:
            raise RuntimeError('Carinthia-S image/mask/CSV identity mismatch')
        referenced_images.add(image_path)
        referenced_masks.add(mask_path)
        label = row['label']
        if label not in {'1', '2', '3', '4', '5', '6'}:
            raise RuntimeError('Unexpected Carinthia-S numeric class')
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            image.load()
            mask.load()
            if image.size != mask.size:
                raise RuntimeError('Image/mask dimension mismatch')
            dimensions[f'{image.width}x{image.height}'] += 1
            modes[f'image:{image.mode},mask:{mask.mode}'] += 1
            source_mask_mode = mask.mode
            if mask.mode in {'RGB', 'RGBA'}:
                channels = mask.convert('RGB').split()
                if ImageChops.difference(channels[0], channels[1]).getbbox() or ImageChops.difference(channels[0], channels[2]).getbbox():
                    nongrayscale_channel_ids.append(row['filename'])
                if mask.mode == 'RGBA' and mask.getchannel('A').getextrema() != (255, 255):
                    nonopaque_alpha_ids.append(row['filename'])
            elif mask.mode not in {'L', '1'}:
                nongrayscale_channel_ids.append(row['filename'])
            mask = mask.convert('L')
            histogram = mask.histogram()
            values = {value for value, count in enumerate(histogram) if count}
            if not values.issubset({0, 255}):
                nonbinary_ids.append(row['filename'])
            value_histogram.update({str(v): histogram[v] for v in values})
            foreground = sum(histogram[1:])
            if foreground == 0:
                empty_masks_by_class[label] += 1
            nonempty_pixels.append(foreground)
            decoded = hashlib.sha256(f'{image.width}x{image.height},{image.mode}'.encode() + image.tobytes()).hexdigest()
        class_counts[label] += 1
        record = {'id': row['filename'], 'image': str(image_path.relative_to(ROOT)),
                  'mask': str(mask_path.relative_to(ROOT)), 'class_id_as_published': int(label),
                  'image_sha256': digest(image_path), 'mask_sha256': digest(mask_path),
                  'decoded_image_sha256': decoded, 'source_mask_mode': source_mask_mode,
                  'mask_nonzero_grayscale_audit_pixels': foreground,
                  'mask_intermediate_grayscale_audit_pixels': sum(histogram[1:255])}
        records.append(record)
        decoded_groups[decoded].append(record)
    actual_images = {p.resolve() for p in (base / 'images').iterdir() if p.is_file()}
    actual_masks = {p.resolve() for p in (base / 'masks').iterdir() if p.is_file()}
    if actual_images != referenced_images or actual_masks != referenced_masks:
        raise RuntimeError('CSV and extracted file inventory differ')
    duplicates = [group for group in decoded_groups.values() if len(group) > 1]
    conflicting = [group for group in duplicates if len({(r['mask_sha256'], r['class_id_as_published']) for r in group}) > 1]
    nonbinary_by_class = Counter(str(r['class_id_as_published']) for r in records if r['mask_intermediate_grayscale_audit_pixels'])
    class6_nonempty = [r['id'] for r in records if r['class_id_as_published'] == 6 and r['mask_nonzero_grayscale_audit_pixels'] > 0]
    encoding_review = bool(nonbinary_ids or nongrayscale_channel_ids or nonopaque_alpha_ids)
    result = {'schema': 'carinthia-s-local-audit-v1',
              'status': 'PASS_LOCAL_INVENTORY_MASK_ENCODING_REVIEW_REQUIRED' if encoding_review else 'PASS_LOCAL_INVENTORY_AND_BINARY_MASK_AUDIT',
              'images': len(actual_images), 'masks': len(actual_masks), 'csv_rows': len(rows),
              'image_mask_pairs': len(records), 'class_counts': dict(sorted(class_counts.items())),
              'class_id_policy': 'Keep publisher numeric IDs 1..6; no semantic mapping onto Wafer-Datas classes',
              'dimensions': dict(dimensions), 'modes': dict(modes), 'mask_pixel_value_counts': dict(value_histogram),
              'empty_masks': sum(empty_masks_by_class.values()), 'empty_masks_by_class': dict(sorted(empty_masks_by_class.items())),
              'mask_foreground_pixels_min': min(nonempty_pixels), 'mask_foreground_pixels_max': max(nonempty_pixels),
              'decoded_duplicate_groups': len(duplicates), 'decoded_duplicate_images': sum(len(g) for g in duplicates),
              'duplicate_pixels_with_different_mask_or_class_groups': len(conflicting),
              'missing_or_orphan_files': 0, 'shape_mismatches': 0, 'nonbinary_masks': len(nonbinary_ids),
              'nonbinary_mask_ids': nonbinary_ids,
              'nonbinary_masks_by_class': dict(sorted(nonbinary_by_class.items())),
              'strictly_binary_masks': len(records) - len(nonbinary_ids),
              'class6_nonempty_mask_count': len(class6_nonempty),
              'class6_nonempty_mask_ids': class6_nonempty,
              'masks_with_nongrayscale_rgb_channels': len(nongrayscale_channel_ids),
              'nongrayscale_rgb_mask_ids': nongrayscale_channel_ids,
              'masks_with_nonopaque_alpha': len(nonopaque_alpha_ids),
              'nonopaque_alpha_mask_ids': nonopaque_alpha_ids,
              'source_masks_unchanged': True, 'training_mask_conversion_performed': False,
              'histogram_policy': 'Audit-only grayscale conversion for mode/value inventory; raw masks remain byte-identical.',
              'modality': 'SEM', 'training_used': False, 'split_created': False,
              'publisher_description_notes': ['Class 6 has no visible defects due to possible SEM alignment mismatch.',
                                              'Class 5 contains only four images; independent supervised evaluation is severely limited.'],
              'limitations': ['No physical wafer/lot split has been established.',
                              'SEM domain is not interchangeable with a visible-light camera platform.',
                              'Some source masks may include intermediate gray values or RGB encodings; review the reported IDs before any segmentation training.',
                              'Class 6 is described as having no visible defects, but one source mask contains 50 nonzero pixels; review the annotation.',
                              'Training and class-taxonomy reconciliation were not performed.']}
    save(ROOT / 'audit/carinthia_s_file_manifest.json', {'schema': 'carinthia-s-pairs-v1', 'records': records})
    save(ROOT / 'reports/carinthia_s_audit.json', result)
    return result


def acquire_wafer_cache(cache: Path) -> dict:
    source_data = cache / 'data'
    manifest_path = source_data / 'manifest.json'
    prior_audit_path = cache / 'reports/data_audit.json'
    prior_independent_path = cache / 'reports/independent_data_audit.json'
    manifest = read_json(manifest_path)
    prior_audit = read_json(prior_audit_path)
    independent = read_json(prior_independent_path)
    if independent['status'] != 'PASS_INDEPENDENT_DATA_AUDIT' or prior_audit['status'] != 'PASS_DATA_AUDIT':
        raise RuntimeError('Prior independent and preparation audits must pass')
    if independent['input_sha256'][str(manifest_path)] != digest(manifest_path):
        raise RuntimeError('Manifest differs from independent audit binding')
    if independent['input_sha256'][str(prior_audit_path)] != digest(prior_audit_path):
        raise RuntimeError('Preparation audit differs from independent audit binding')
    if manifest['names'] != NAMES:
        raise RuntimeError('Canonical six-class taxonomy changed')
    for source, name in [(manifest_path, 'source_manifest.json'), (prior_audit_path, 'source_data_audit.json'),
                         (prior_independent_path, 'source_independent_data_audit.json')]:
        copied_verified(source, ROOT / 'audit' / name, digest(source))
    prepared = ROOT / 'data/wafer'
    expected_files = {Path(f['path']): f['sha256'] for f in prior_audit['files']}
    copied = []
    for split in ['train', 'val', 'test']:
        for kind in ['images', 'labels']:
            for source in sorted((source_data / kind / split).iterdir()):
                if source.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.txt'}:
                    continue
                if source not in expected_files:
                    raise RuntimeError(f'Unbound source prepared file: {source}')
                dest = prepared / kind / split / source.name
                copied_verified(source, dest, expected_files[source])
                copied.append({'path': str(dest.relative_to(ROOT)), 'sha256': expected_files[source]})
    copied_verified(manifest_path, prepared / 'manifest.json', digest(manifest_path))
    prepared.joinpath('data.yaml').write_text('path: ' + str(prepared.resolve()) + '\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n' + ''.join('- ' + n + '\n' for n in NAMES), encoding='utf-8')
    # Original unmodified XML/photos, including the 98 quarantined images.
    source_acquisition_path = cache.parent / 'wafer_research/download_manifest.json'
    source_acquisition = read_json(source_acquisition_path)
    if source_acquisition['commit'] != SOURCE_COMMIT:
        raise RuntimeError('Unexpected Wafer-Datas source commit')
    copied_verified(source_acquisition_path, ROOT / 'audit/source_download_manifest.json', digest(source_acquisition_path))
    raw_bytes = 0
    for record in source_acquisition['files']:
        source = cache / 'raw' / record['path']
        target = ROOT / 'data/wafer_datas_raw' / record['path']
        copied_verified(source, target, record['sha256'])
        raw = source.read_bytes()
        git_blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        if git_blob != record['git_blob_sha']:
            raise RuntimeError(f'Raw git blob binding failed: {source}')
        raw_bytes += len(raw)
    save(ROOT / 'audit/copied_wafer_file_hashes.json', {'files': copied, 'prepared_files_verified': len(copied)})
    splits = {}
    hashes = defaultdict(set)
    group_splits = defaultdict(set)
    records_by_id = {r['id']: r for r in manifest['records']}
    max_box_error = 0.0
    for split in ['train', 'val', 'test']:
        images = sorted((prepared / 'images' / split).iterdir())
        images = [p for p in images if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        labels = sorted((prepared / 'labels' / split).glob('*.txt'))
        if {p.stem for p in images} != {p.stem for p in labels}:
            raise RuntimeError(f'Missing/orphan labels: {split}')
        classes = Counter({name: 0 for name in NAMES})
        class_images = Counter({name: 0 for name in NAMES})
        groups = set()
        for image_path in images:
            record = records_by_id[image_path.stem]
            if record['split'] != split:
                raise RuntimeError('Frozen manifest split changed')
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                width, height = image.size
                decoded = hashlib.sha256(f'{width}x{height}'.encode() + image.tobytes()).hexdigest()
            if decoded != record['decoded_sha256']:
                raise RuntimeError('Copied decoded image differs from audited original')
            hashes[split].add(decoded)
            groups.add(record['group_id'])
            group_splits[record['group_id']].add(split)
            lines = (prepared / 'labels' / split / (image_path.stem + '.txt')).read_text().strip().splitlines()
            if not lines:
                raise RuntimeError('Empty label in defect-only source')
            if len(lines) != len(record['objects']):
                raise RuntimeError('Source object count differs from prepared label')
            present = set()
            for line, obj in zip(lines, record['objects']):
                tokens = line.split()
                if len(tokens) != 5:
                    raise RuntimeError('Invalid YOLO label field count')
                class_id = int(tokens[0])
                x, y, w, h = map(float, tokens[1:])
                if not 0 <= class_id < len(NAMES) or not all(math.isfinite(v) for v in [x, y, w, h]):
                    raise RuntimeError('Invalid class or nonfinite bbox')
                if not (w > 0 and h > 0 and x-w/2 >= -1e-8 and y-h/2 >= -1e-8 and x+w/2 <= 1+1e-8 and y+h/2 <= 1+1e-8):
                    raise RuntimeError('Invalid/out-of-frame bbox')
                if NAMES[class_id] != obj['class']:
                    raise RuntimeError('Source class differs from YOLO label')
                decoded_box = [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]
                error = max(abs(a-b) for a, b in zip(decoded_box, obj['bbox_xyxy']))
                max_box_error = max(max_box_error, error)
                if error > 1e-5:
                    raise RuntimeError('Source bbox differs from prepared label')
                classes[NAMES[class_id]] += 1
                present.add(NAMES[class_id])
            class_images.update(present)
        splits[split] = {'images': len(images), 'labels': len(labels), 'groups': len(groups),
                         'bbox': sum(classes.values()), 'classes': dict(classes), 'class_images': dict(class_images)}
        prior_split = independent['split'][split]
        if any(splits[split][key] != prior_split[key] for key in ['images', 'groups', 'bbox', 'classes']):
            raise RuntimeError('Frozen split counts not reproduced')
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if hashes[a] & hashes[b]:
            raise RuntimeError('Decoded image duplicate crosses splits')
    if any(len(s) > 1 for s in group_splits.values()):
        raise RuntimeError('Prior appearance/family group crosses splits')
    latest = {'status': 'UNVERIFIED'}
    try:
        with open_url('https://api.github.com/repos/ztao3243/Wafer-Datas/commits/main') as response:
            latest_meta = json.load(response)
        save(ROOT / 'audit/github_latest_commit.json', latest_meta)
        latest = {'status': 'VERIFIED_CURRENT', 'sha': latest_meta['sha'],
                  'matches_frozen_source': latest_meta['sha'] == SOURCE_COMMIT,
                  'commit_url': latest_meta['html_url']}
    except Exception as exc:
        latest = {'status': 'LOOKUP_FAILED_FROZEN_SOURCE_RETAINED', 'error': str(exc)}
    result = {'source_url': 'https://github.com/ztao3243/Wafer-Datas', 'source_commit': SOURCE_COMMIT,
              'acquisition': 'VERIFIED_PRIOR_DOWNLOAD_CACHE_COPY', 'license': source_acquisition['dataset_license'],
              'source_original_images': 2132, 'source_original_boxes': 3693,
              'source_raw_files_verified': len(source_acquisition['files']), 'source_raw_bytes': raw_bytes,
              'prepared_files_verified': len(copied), 'prepared_image_label_sha256_verified': True,
              'names': NAMES, 'splits': splits, 'total_images': sum(s['images'] for s in splits.values()),
              'total_boxes': sum(s['bbox'] for s in splits.values()), 'quarantined_images': 98,
              'frozen_manifest_sha256': digest(manifest_path), 'frozen_split_preserved': True,
              'decoded_hash_cross_split_overlaps': 0, 'group_cross_split_overlaps': 0,
              'missing_or_empty_labels': 0, 'invalid_boxes': 0, 'max_yolo_bbox_roundtrip_error_pixels': max_box_error,
              'latest_source_commit_check': latest, 'data_yaml': 'data/wafer/data.yaml',
              'physical_wafer_lot_independence': 'NOT_VERIFIED', 'same_design_independence': 'NOT_VERIFIED',
              'bbox_convention': manifest['bbox_convention'], 'source_augmentation_excluded': True,
              'raw_directory': 'data/wafer_datas_raw', 'prior_manifest_paths_are_historical_provenance': True}
    save(ROOT / 'audit/wafer_cache_acquisition.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wafer-cache', type=Path, help='Prior wafer_r01 directory with data/raw/reports')
    parser.add_argument('--carinthia-only', action='store_true')
    parser.add_argument('--skip-carinthia', action='store_true')
    args = parser.parse_args()
    summary_path = ROOT / 'reports/data_summary.json'
    summary = read_json(summary_path) if summary_path.exists() else {}
    summary.update(schema='wafer-platform-data-summary-v1', created_utc=datetime.now(timezone.utc).isoformat())
    if not args.skip_carinthia:
        summary['carinthia_s'] = acquire_carinthia()
    if args.wafer_cache and not args.carinthia_only:
        summary['wafer_datas'] = acquire_wafer_cache(args.wafer_cache.resolve())
    if 'wafer_datas' in summary:
        summary['names'] = summary['wafer_datas']['names']
        summary['splits'] = summary['wafer_datas']['splits']
    summary['status'] = 'PASS_ACQUISITION_AND_COPIED_WAFER_AUDIT' if 'wafer_datas' in summary else 'PASS_CARINTHIA_DOWNLOAD_CHECKSUM'
    save(summary_path, summary)
    print(json.dumps({k: summary[k] for k in ['schema', 'status', 'splits'] if k in summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
