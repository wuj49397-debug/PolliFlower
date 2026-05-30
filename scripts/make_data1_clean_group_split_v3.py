import json
import random
import re
import shutil
from pathlib import Path

ROOT = Path("/root/autodl-tmp/flower_baseline")
SRC = ROOT / "data"
DST = ROOT / "data_1"

SEED = 20260525
TARGET_TEST = 2242
TARGET_VAL = 2285

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TASKS = ["yolo_det", "yolo_stigma_seg", "yolo_pollination_pose"]

if DST.exists():
    raise SystemExit(f"{DST} already exists. Remove it first: rm -rf {DST}")

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

def get_instance_id(shape):
    attrs = shape.get("attributes", {}) or {}
    if "instance_id" in attrs:
        return str(attrs["instance_id"])
    if shape.get("group_id", None) is not None:
        return str(shape["group_id"])
    return None

def is_flower_box(shape):
    label = str(shape.get("label", "")).lower()
    shape_type = str(shape.get("shape_type", "")).lower()
    attrs = shape.get("attributes", {}) or {}
    role = str(attrs.get("instance_role", "")).lower()
    if "stigma" in label or "pollination" in label:
        return False
    return shape_type == "rectangle" or role == "rectangle"

def count_instances(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = set()
    for shape in data.get("shapes", []):
        if is_flower_box(shape):
            iid = get_instance_id(shape)
            if iid is not None:
                ids.add(iid)
    return len(ids)

def files_by_stem(folder, exts):
    out = {}
    if not folder.exists():
        return out
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            out[p.stem] = p
    return out

print("Loading raw images and annotations...", flush=True)

items = {}
for old_split in ["train", "val", "test"]:
    img_map = files_by_stem(SRC / old_split / "images", IMG_EXTS)
    ann_map = files_by_stem(SRC / old_split / "annotations_with_ids", {".json"})
    for stem in sorted(set(img_map) & set(ann_map)):
        if stem in items:
            continue
        ann = ann_map[stem]
        items[stem] = {
            "stem": stem,
            "base": base_stem(stem),
            "image": img_map[stem],
            "annotation": ann,
            "instances": count_instances(ann),
            "old_split": old_split,
        }

groups = {}
for stem, item in items.items():
    groups.setdefault(item["base"], []).append(stem)

group_list = []
for base, stems in groups.items():
    inst = sum(items[s]["instances"] for s in stems)
    group_list.append({
        "base": base,
        "stems": stems,
        "images": len(stems),
        "instances": inst,
        "density": inst / len(stems),
    })

total_images = sum(g["images"] for g in group_list)
total_instances = sum(g["instances"] for g in group_list)
global_avg = total_instances / total_images
TARGET_TRAIN = total_images - TARGET_VAL - TARGET_TEST

print(f"Total images={total_images}, instances={total_instances}, avg={global_avg:.3f}", flush=True)
print(f"Target images: train={TARGET_TRAIN}, val={TARGET_VAL}, test={TARGET_TEST}", flush=True)

random.shuffle(group_list)

def pick_groups(pool, target_images, target_avg, name):
    selected = []
    cur_img = 0
    cur_inst = 0

    pool = list(pool)

    while cur_img < target_images and pool:
        best_i = None
        best_score = None

        for i, g in enumerate(pool):
            ni = cur_img + g["images"]
            ns = cur_inst + g["instances"]
            avg = ns / ni

            img_err = abs(ni - target_images)
            avg_err = abs(avg - target_avg) * target_images
            over_penalty = max(0, ni - target_images) * 3

            score = img_err + avg_err + over_penalty

            if best_score is None or score < best_score:
                best_score = score
                best_i = i

        g = pool.pop(best_i)
        selected.append(g)
        cur_img += g["images"]
        cur_inst += g["instances"]

    # If the last group makes image-count much worse, remove it.
    improved = True
    while improved and selected:
        improved = False
        current_gap = abs(cur_img - target_images)
        for i, g in enumerate(selected):
            ni = cur_img - g["images"]
            if ni <= 0:
                continue
            new_gap = abs(ni - target_images)
            if new_gap < current_gap:
                pool.append(g)
                cur_img -= g["images"]
                cur_inst -= g["instances"]
                selected.pop(i)
                improved = True
                break

    print(f"{name}: images={cur_img}, instances={cur_inst}, avg={cur_inst/cur_img:.3f}, groups={len(selected)}", flush=True)
    return selected, pool

# Pick test first, then val, train gets the rest.
test_groups, remaining = pick_groups(group_list, TARGET_TEST, global_avg, "test")
val_groups, remaining = pick_groups(remaining, TARGET_VAL, global_avg, "val")
train_groups = remaining

assigned = {
    "train": train_groups,
    "val": val_groups,
    "test": test_groups,
}

split_stems = {}
for split, gs in assigned.items():
    stems = []
    for g in gs:
        stems.extend(g["stems"])
    split_stems[split] = set(stems)

print("\nFinal raw split:", flush=True)
for split in ["train", "val", "test"]:
    n = len(split_stems[split])
    inst = sum(items[s]["instances"] for s in split_stems[split])
    print(f"{split}: images={n}, instances={inst}, avg={inst/n:.3f}, groups={len(assigned[split])}", flush=True)

def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

print("\nCopying raw data_1...", flush=True)
for split in ["train", "val", "test"]:
    print(f"Copy raw {split}: {len(split_stems[split])}", flush=True)
    for stem in sorted(split_stems[split]):
        item = items[stem]
        copy_file(item["image"], DST / split / "images" / item["image"].name)
        copy_file(item["annotation"], DST / split / "annotations_with_ids" / item["annotation"].name)

print("\nLoading YOLO maps...", flush=True)
yolo = {}
for task in TASKS:
    yolo[task] = {}
    for old_split in ["train", "val", "test"]:
        yolo[task][old_split] = {
            "images": files_by_stem(SRC / task / "images" / old_split, IMG_EXTS),
            "labels": files_by_stem(SRC / task / "labels" / old_split, {".txt"}),
        }

def find_yolo(task, stem, kind):
    for old_split in ["train", "val", "test"]:
        if stem in yolo[task][old_split][kind]:
            return yolo[task][old_split][kind][stem]
    return None

task_report = {}

print("\nCopying YOLO-style data_1...", flush=True)
for task in TASKS:
    task_report[task] = {}
    for split in ["train", "val", "test"]:
        copied = 0
        missing = 0
        print(f"Copy {task}/{split}: target={len(split_stems[split])}", flush=True)

        for stem in sorted(split_stems[split]):
            img = find_yolo(task, stem, "images")
            lab = find_yolo(task, stem, "labels")
            if img is None or lab is None:
                missing += 1
                continue
            copy_file(img, DST / task / "images" / split / img.name)
            copy_file(lab, DST / task / "labels" / split / lab.name)
            copied += 1

        task_report[task][split] = {"copied": copied, "missing": missing}
        print(f"{task}/{split}: copied={copied}, missing={missing}", flush=True)

yaml_map = {
    "yolo_det/polliflower_det.yaml": "yolo_det",
    "yolo_stigma_seg/polliflower_stigma_seg.yaml": "yolo_stigma_seg",
    "yolo_pollination_pose/polliflower_pollination_pose.yaml": "yolo_pollination_pose",
}

print("\nWriting YAML files...", flush=True)
for rel, task in yaml_map.items():
    src_yaml = SRC / rel
    dst_yaml = DST / rel
    lines = src_yaml.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        if line.strip().startswith("path:"):
            out.append(f"path: {DST / task}")
        else:
            out.append(line)
    dst_yaml.parent.mkdir(parents=True, exist_ok=True)
    dst_yaml.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(dst_yaml, flush=True)

# Leakage assertion by base group.
split_bases = {
    split: {items[s]["base"] for s in split_stems[split]}
    for split in ["train", "val", "test"]
}
overlap_report = {
    "train-val": sorted(split_bases["train"] & split_bases["val"]),
    "train-test": sorted(split_bases["train"] & split_bases["test"]),
    "val-test": sorted(split_bases["val"] & split_bases["test"]),
}

manifest = {
    "seed": SEED,
    "strategy": "Clean group-level split. Original images and offline augmented variants are assigned to only one split.",
    "target_images": {
        "train": TARGET_TRAIN,
        "val": TARGET_VAL,
        "test": TARGET_TEST,
    },
    "split": {
        split: {
            "images": len(split_stems[split]),
            "instances": sum(items[s]["instances"] for s in split_stems[split]),
            "avg_inst_per_image": sum(items[s]["instances"] for s in split_stems[split]) / len(split_stems[split]),
            "groups": len(assigned[split]),
        }
        for split in ["train", "val", "test"]
    },
    "leakage_overlap": {k: len(v) for k, v in overlap_report.items()},
    "task_report": task_report,
}

manifest_path = DST / "clean_group_split_manifest_v3.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"\nSaved manifest: {manifest_path}", flush=True)
print("Leakage overlap:", {k: len(v) for k, v in overlap_report.items()}, flush=True)

if any(overlap_report.values()):
    raise SystemExit("ERROR: leakage still exists.")
else:
    print("Done. No base-group leakage in generated data_1.", flush=True)
