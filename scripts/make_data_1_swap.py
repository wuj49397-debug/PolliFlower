import json
import random
import shutil
from pathlib import Path

ROOT = Path("/root/autodl-tmp/flower_baseline")
SRC = ROOT / "data"
DST = ROOT / "data_1"

SEED = 20260522
N_SWAP = 500
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TASKS = ["yolo_det", "yolo_stigma_seg", "yolo_pollination_pose"]

if DST.exists():
    raise SystemExit(f"{DST} already exists. Please remove or rename it first.")

random.seed(SEED)

def files_by_stem(folder, exts):
    out = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            if p.stem in out:
                raise RuntimeError(f"Duplicate stem in {folder}: {p.stem}")
            out[p.stem] = p
    return out

raw = {}
for split in ["train", "val", "test"]:
    raw[split] = {
        "images": files_by_stem(SRC / split / "images", IMG_EXTS),
        "anns": files_by_stem(SRC / split / "annotations_with_ids", {".json"}),
    }

yolo = {}
for task in TASKS:
    yolo[task] = {}
    for split in ["train", "val", "test"]:
        yolo[task][split] = {
            "images": files_by_stem(SRC / task / "images" / split, IMG_EXTS),
            "labels": files_by_stem(SRC / task / "labels" / split, {".txt"}),
        }

def eligible(split):
    stems = set(raw[split]["images"]) & set(raw[split]["anns"])
    for task in TASKS:
        stems &= set(yolo[task][split]["images"]) & set(yolo[task][split]["labels"])
    return sorted(stems)

train_eligible = eligible("train")
test_eligible = eligible("test")

print(f"Eligible train samples: {len(train_eligible)}")
print(f"Eligible test samples:  {len(test_eligible)}")

train_to_test = set(random.sample(train_eligible, N_SWAP))
test_to_train = set(random.sample(test_eligible, N_SWAP))

def source_split(new_split, stem):
    if new_split == "val":
        return "val"
    if new_split == "train":
        return "test" if stem in test_to_train else "train"
    if new_split == "test":
        return "train" if stem in train_to_test else "test"
    raise ValueError(new_split)

raw_stems = {
    split: set(raw[split]["images"]) & set(raw[split]["anns"])
    for split in ["train", "val", "test"]
}

new_raw = {
    "train": (raw_stems["train"] - train_to_test) | test_to_train,
    "val": raw_stems["val"],
    "test": (raw_stems["test"] - test_to_train) | train_to_test,
}

def task_stems(task, split):
    return set(yolo[task][split]["images"]) & set(yolo[task][split]["labels"])

new_yolo = {}
for task in TASKS:
    new_yolo[task] = {
        "train": (task_stems(task, "train") - train_to_test) | test_to_train,
        "val": task_stems(task, "val"),
        "test": (task_stems(task, "test") - test_to_train) | train_to_test,
    }

def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

print("Creating raw data_1...")
for split in ["train", "val", "test"]:
    for stem in sorted(new_raw[split]):
        ss = source_split(split, stem)
        img = raw[ss]["images"][stem]
        ann = raw[ss]["anns"][stem]
        copy_file(img, DST / split / "images" / img.name)
        copy_file(ann, DST / split / "annotations_with_ids" / ann.name)

print("Creating converted YOLO-style datasets...")
for task in TASKS:
    for split in ["train", "val", "test"]:
        for stem in sorted(new_yolo[task][split]):
            ss = source_split(split, stem)
            img = yolo[task][ss]["images"][stem]
            lab = yolo[task][ss]["labels"][stem]
            copy_file(img, DST / task / "images" / split / img.name)
            copy_file(lab, DST / task / "labels" / split / lab.name)

yaml_map = {
    "yolo_det/polliflower_det.yaml": "yolo_det",
    "yolo_stigma_seg/polliflower_stigma_seg.yaml": "yolo_stigma_seg",
    "yolo_pollination_pose/polliflower_pollination_pose.yaml": "yolo_pollination_pose",
}

for rel, task in yaml_map.items():
    src_yaml = SRC / rel
    dst_yaml = DST / rel
    lines = src_yaml.read_text(encoding="utf-8").splitlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith("path:"):
            new_lines.append(f"path: {DST / task}")
        else:
            new_lines.append(line)
    dst_yaml.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

manifest = {
    "seed": SEED,
    "n_swap_each_direction": N_SWAP,
    "train_to_test_stems": sorted(train_to_test),
    "test_to_train_stems": sorted(test_to_train),
    "raw_counts": {k: len(v) for k, v in new_raw.items()},
    "yolo_counts": {
        task: {split: len(new_yolo[task][split]) for split in ["train", "val", "test"]}
        for task in TASKS
    },
    "note": "Validation split is unchanged. For formal results, retrain all baselines on data_1/train and data_1/val, then evaluate once on data_1/test."
}

manifest_path = DST / f"split_swap_{N_SWAP}_seed{SEED}.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

def count_files(folder):
    return len([p for p in folder.iterdir() if p.is_file()])

print("\nRaw data_1 counts:")
for split in ["train", "val", "test"]:
    print(
        split,
        "images=", count_files(DST / split / "images"),
        "annotations=", count_files(DST / split / "annotations_with_ids"),
    )

print("\nYOLO-style data_1 counts:")
for task in TASKS:
    for split in ["train", "val", "test"]:
        print(
            task,
            split,
            "images=", count_files(DST / task / "images" / split),
            "labels=", count_files(DST / task / "labels" / split),
        )

print(f"\nSaved manifest: {manifest_path}")
