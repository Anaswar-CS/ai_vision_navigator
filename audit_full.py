import os, glob, random, sys
from pathlib import Path

def read_yaml():
    yaml_path = Path('dataset/data.yaml')
    if yaml_path.exists():
        print('--- data.yaml contents ---')
        print(yaml_path.read_text())
    else:
        print('data.yaml not found')

def count_files():
    splits = ['train', 'val', 'test']
    for split in splits:
        img_dir = Path('dataset/images') / split
        label_dir = Path('dataset/labels') / split
        img_count = sum(1 for _ in img_dir.rglob('*') if _.is_file())
        label_count = sum(1 for _ in label_dir.rglob('*.txt') if _.is_file())
        print(f'{split} images: {img_count}, labels: {label_count}')
        if img_count != label_count:
            print(f'!! MISMATCH in {split}: images vs labels')

def per_class_counts():
    ids = list(range(6))
    splits = ['train', 'val', 'test']
    for split in splits:
        label_dir = Path('dataset/labels') / split
        counts = {i:0 for i in ids}
        for txt_file in label_dir.glob('*.txt'):
            present = set()
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            cid = int(parts[0])
                            present.add(cid)
                        except ValueError:
                            continue
            for cid in present:
                counts[cid] += 1
        print(f'--- {split} per-class label file counts ---')
        for i in ids:
            print(f'{i}: {counts[i]}')

def integrity_check():
    ids = list(range(6))
    splits = ['train', 'val', 'test']
    rng = random.Random(42)
    for split in splits:
        label_dir = Path('dataset/labels') / split
        files = list(label_dir.glob('*.txt'))
        # group files by class present
        class_to_files = {i: [] for i in ids}
        for f in files:
            with open(f, 'r') as fh:
                for line in fh:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cid = int(parts[0])
                        if cid in ids:
                            class_to_files[cid].append(f)
                    except ValueError:
                        continue
        for cid, flist in class_to_files.items():
            if not flist:
                continue
            sample = rng.sample(flist, min(5, len(flist)))
            for fp in sample:
                with open(fp, 'r') as fh:
                    for ln_no, line in enumerate(fh, 1):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            print(f'Integrity FAIL {split} class {cid} {fp} line {ln_no}: expected 5 values, got {len(parts)}')
                            continue
                        try:
                            cid_val = int(parts[0])
                            if not (0 <= cid_val <=5):
                                print(f'Integrity FAIL {split} class {cid} {fp} line {ln_no}: class_id {cid_val} out of range')
                        except ValueError:
                            print(f'Integrity FAIL {split} class {cid} {fp} line {ln_no}: class_id not integer')
                        # check normalized floats
                        for val in parts[1:]:
                            try:
                                fval = float(val)
                                if not (0.0 <= fval <= 1.0):
                                    print(f'Integrity FAIL {split} class {cid} {fp} line {ln_no}: value {val} out of [0,1]')
                            except ValueError:
                                print(f'Integrity FAIL {split} class {cid} {fp} line {ln_no}: value {val} not a float')

def duplicate_check():
    splits = ['train', 'val', 'test']
    filename_map = {}
    for split in splits:
        img_dir = Path('dataset/images') / split
        for img_path in img_dir.rglob('*'):
            if img_path.is_file():
                name = img_path.name
                filename_map.setdefault(name, []).append(split)
    dupes = {name: locs for name, locs in filename_map.items() if len(locs) > 1}
    if dupes:
        print('Duplicate filenames across splits:')
        for name, locs in dupes.items():
            print(f'{name}: {", ".join(locs)}')
    else:
        print('No duplicate filenames across splits')

def model_status():
    model_path = Path('models/custom/best.pt')
    if model_path.exists():
        size = model_path.stat().st_size
        mtime = model_path.stat().st_mtime
        from datetime import datetime
        print(f'Model exists: size={size} bytes, modified={datetime.fromtimestamp(mtime)}')
    else:
        print('Model file models/custom/best.pt does not exist')

if __name__ == '__main__':
    read_yaml()
    count_files()
    per_class_counts()
    integrity_check()
    duplicate_check()
    model_status()
