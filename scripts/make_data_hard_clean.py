import json
import random
import re
import shutil
from pathlib import Path

ROOT = Path("/root/autodl-tmp/flower_baseline")
SRC = ROOT / "data"
DST = ROOT / "data_hard"

SEED = 20260525
TARGET_VAL = 2285

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TASKS = ["yolo_det", "yolo_stigma_seg", "yolo_pollination_pose"]

if DST.exists():
    raise SystemExit(f"{DST} already exists. Please rename or remove it first.")

random.seed(SEED)

def base_stem(stem):
    s = stem
    if ".rf." in s:
        s = s.split(".rf.")[0]

    s = re.sub(r"\s*-\s*副本fz$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*-\s*副本$", "", s, flags=re.IGNORECASE)

    patterns = [
        r"_brightness_[+-]?\d+$",
        r"_brightness_plus\d+$",
        r"_brightness_minus\d+$",
        r"_brightness$",
        r"_dark$",
        r"_rotate_[+-]?\d+$",
        r"_rotate$",
        r"_rotated_[+-]?\d+$",
        r"_rotated[+-]?\d+$",
        r"_rotated$",
        r"_rot[+-]?\d+$",
        r"_rot_[+-]?\d+$",
        r"fz$",
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

def image_files(split):
    img_dir = SRC / split / "images"
    files = []
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)

def count_instances_for_image(img_path):
    ann = SRC / img_path.parent.parent.name / "annotations_with_ids" / f"{img_path.stem}.json"
    if not ann.exists():
        return 0

    try:
        data = json.loads(ann.read_text(encoding="utf-8"))
    except Exception:
        return 0

    count = 0
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).lower()
        shape_type = str(shape.get("shape_type", "")).lower()
        attrs = shape.get("attributes", {}) or {}
        role = str(attrs.get("instance_role", "")).lower()

        if "stigma" in label or "pollination" in label:
            continue
        if shape_type == "rectangle" or role == "rectangle":
            count += 1
    return count

def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

print("Loading old hard data from data/train,val,test ...")

all_images = []
for split in ["train", "val", "test"]:
    for img in image_files(split):
        all_images.append(img)

groups = {}
for img in all_images:
    g = base_stem(img.stem)
    groups.setdefault(g, []).append(img)

old_test_groups = {base_stem(img.stem) for img in image_files("test")}

test_groups = set(old_test_groups)
remaining_groups = sorted(set(groups) - test_groups)

random.shuffle(remaining_groups)

val_groups = set()
val_count = 0
for g in remaining_groups:
    if val_count >= TARGET_VAL:
        break
    val_groups.add(g)
    val_count += len(groups[g])

train_groups = set(remaining_groups) - val_groups

split_groups = {
    "train": train_groups,
    "val": val_groups,
    "test": test_groups,
}

split_images = {}
for split, gs in split_groups.items():
    imgs = []
    for g in sorted(gs):
        imgs.extend(groups[g])
    split_images[split] = sorted(imgs, key=lambda p: p.name)

print("\nNew data_hard split plan:")
for split in ["train", "val", "test"]:
    imgs = split_images[split]
    inst = sum(count_instances_for_image(p) for p in imgs)
    avg = inst / len(imgs) if imgs else 0
    print(f"{split}: images={len(imgs)}, instances={inst}, avg={avg:.3f}, groups={len(split_groups[split])}")

print("\nCopying raw data_hard ...")
for split in ["train", "val", "test"]:
    for img in split_images[split]:
        src_split = img.parent.parent.name
        ann = SRC / src_split / "annotations_with_ids" / f"{img.stem}.json"

        copy_file(img, DST / split / "images" / img.name)
        if ann.exists():
            copy_file(ann, DST / split / "annotations_with_ids" / ann.name)
        else:
            print(f"[WARN] missing raw annotation: {ann}")

def load_task_maps(task):
    img_map = {}
    label_map = {}

    for split in ["train", "val", "test"]:
        img_dir = SRC / task / "images" / split
        lab_dir = SRC / task / "labels" / split

        if img_dir.exists():
            for p in img_dir.iterdir():
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    img_map[p.stem] = p

        if lab_dir.exists():
            for p in lab_dir.glob("*.txt"):
                label_map[p.stem] = p

    return img_map, label_map

print("\nCopying YOLO-style task data ...")

for task in TASKS:
    img_map, label_map = load_task_maps(task)

    for split in ["train", "val", "test"]:
        copied = 0
        missing_img = 0
        missing_label = 0

        for raw_img in split_images[split]:
            stem = raw_img.stem

            if stem not in img_map:
                missing_img += 1
                continue

            copy_file(img_map[stem], DST / task / "images" / split / img_map[stem].name)

            if stem in label_map:
                copy_file(label_map[stem], DST / task / "labels" / split / label_map[stem].name)
            else:
                missing_label += 1
                (DST / task / "labels" / split).mkdir(parents=True, exist_ok=True)
                (DST / task / "labels" / split / f"{stem}.txt").write_text("", encoding="utf-8")

            copied += 1

        print(f"{task}/{split}: copied={copied}, missing_img={missing_img}, missing_label={missing_label}")

    yaml_files = list((SRC / task).glob("*.yaml"))
    for yf in yaml_files:
        text = yf.read_text(encoding="utf-8")
        text = re.sub(r"(?m)^path:.*$", f"path: {DST / task}", text)
        (DST / task / yf.name).write_text(text, encoding="utf-8")

manifest = {
    "source": str(SRC),
    "destination": str(DST),
    "seed": SEED,
    "rule": "old data/test base groups are kept as the hard test core; all remaining base groups are re-split into train and val.",
    "target_val_images": TARGET_VAL,
    "splits": {
        split: {
            "images": len(split_images[split]),
            "groups": len(split_groups[split]),
            "instances": sum(count_instances_for_image(p) for p in split_images[split]),
        }
        for split in ["train", "val", "test"]
    },
}
(DST / "data_hard_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

print("\nSaved manifest:", DST / "data_hard_manifest.json")
print("Done.")
