import json
import random
import re
import shutil
from pathlib import Path

ROOT = Path("/root/autodl-tmp/flower_baseline")
DATA = ROOT / "data_1"

TARGET_MOVE_IMAGES = 1000
SEED = 20260523

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TASKS = ["yolo_det", "yolo_stigma_seg", "yolo_pollination_pose"]

random.seed(SEED)

def files_by_stem(folder, exts):
    out = {}
    if not folder.exists():
        return out
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            out[p.stem] = p
    return out

def base_stem(stem):
    s = stem
    patterns = [
        r"_brightness_[+-]?\d+$",
        r"_brightness_plus\d+$",
        r"_brightness_minus\d+$",
        r"_brightness$",
        r"_rotate_[+-]?\d+$",
        r"_rotated$",
        r"_rot[+-]?\d+$",
    ]

    changed = True
    while changed:
        changed = False
        for pat in patterns:
            ns = re.sub(pat, "", s, flags=re.IGNORECASE)
            if ns != s:
                s = ns
                changed = True
    return s

def split_stems(split):
    imgs = files_by_stem(DATA / split / "images", IMG_EXTS)
    anns = files_by_stem(DATA / split / "annotations_with_ids", {".json"})
    return sorted(set(imgs) & set(anns))

def count_files(split):
    return len([p for p in (DATA / split / "images").iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])

train_stems = split_stems("train")
val_stems = split_stems("val")
test_stems = split_stems("test")

groups = {}
for stem in train_stems:
    groups.setdefault(base_stem(stem), []).append(stem)

group_items = list(groups.items())
random.shuffle(group_items)

selected_groups = []
selected_stems = []
count = 0

for g, stems in group_items:
    selected_groups.append(g)
    selected_stems.extend(stems)
    count += len(stems)
    if count >= TARGET_MOVE_IMAGES:
        break

selected_stems = sorted(set(selected_stems))

print("Before:")
print(f"Train images: {count_files('train')}")
print(f"Val images:   {count_files('val')}")
print(f"Test images:  {count_files('test')}")

print("\nSelected:")
print(f"Groups selected: {len(selected_groups)}")
print(f"Images selected: {len(selected_stems)}")

expected_train = count_files("train") - len(selected_stems)
expected_test = count_files("test") + len(selected_stems)

print("\nExpected after:")
print(f"Train images: {expected_train}")
print(f"Val images:   {count_files('val')}")
print(f"Test images:  {expected_test}")

manifest = {
    "mode": "in-place move from data_1/train to data_1/test",
    "seed": SEED,
    "target_move_images": TARGET_MOVE_IMAGES,
    "selected_groups": selected_groups,
    "selected_stems": selected_stems,
    "before": {
        "train_images": count_files("train"),
        "val_images": count_files("val"),
        "test_images": count_files("test"),
    },
    "expected_after": {
        "train_images": expected_train,
        "val_images": count_files("val"),
        "test_images": expected_test,
    },
}

backup_dir = ROOT / "split_backups" / "data_1_expand_test_inplace"
backup_dir.mkdir(parents=True, exist_ok=True)
manifest_path = backup_dir / "move_manifest.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"\nSaved move manifest: {manifest_path}")

def move_file(src, dst):
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    shutil.move(str(src), str(dst))
    return True

def move_raw(stem):
    img = files_by_stem(DATA / "train" / "images", IMG_EXTS).get(stem)
    ann = files_by_stem(DATA / "train" / "annotations_with_ids", {".json"}).get(stem)

    moved = {}
    if img:
        moved["raw_image"] = move_file(img, DATA / "test" / "images" / img.name)
    else:
        moved["raw_image"] = False

    if ann:
        moved["raw_annotation"] = move_file(ann, DATA / "test" / "annotations_with_ids" / ann.name)
    else:
        moved["raw_annotation"] = False

    return moved

def move_yolo(task, stem):
    img = files_by_stem(DATA / task / "images" / "train", IMG_EXTS).get(stem)
    lab = files_by_stem(DATA / task / "labels" / "train", {".txt"}).get(stem)

    moved = {}
    if img:
        moved[f"{task}_image"] = move_file(img, DATA / task / "images" / "test" / img.name)
    else:
        moved[f"{task}_image"] = False

    if lab:
        moved[f"{task}_label"] = move_file(lab, DATA / task / "labels" / "test" / lab.name)
    else:
        moved[f"{task}_label"] = False

    return moved

move_log = []

print("\nMoving files...")
for idx, stem in enumerate(selected_stems, 1):
    record = {"stem": stem}
    record.update(move_raw(stem))
    for task in TASKS:
        record.update(move_yolo(task, stem))
    move_log.append(record)

    if idx % 100 == 0:
        print(f"Moved {idx}/{len(selected_stems)}")

log_path = backup_dir / "move_log.json"
log_path.write_text(json.dumps(move_log, ensure_ascii=False, indent=2), encoding="utf-8")

print("\nAfter:")
print(f"Train images: {count_files('train')}")
print(f"Val images:   {count_files('val')}")
print(f"Test images:  {count_files('test')}")

print(f"\nSaved move log: {log_path}")
print("Done.")
