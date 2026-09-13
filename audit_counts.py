import os, glob, collections
ids = range(6)
base = os.path.join('dataset', 'labels')
for split in ['train', 'val', 'test']:
    counts = collections.Counter()
    split_path = os.path.join(base, split)
    for filepath in glob.glob(os.path.join(split_path, '*.txt')):
        present = set()
        with open(filepath, 'r') as f:
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
    print('---', split, '---')
    for i in ids:
        print(i, counts.get(i, 0))
