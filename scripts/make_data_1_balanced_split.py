import json
import shutil
from pathlib import Path

ROOT = Path("/root/autodl-tmp/flower_baseline")
SRC = ROOT / "data"
DST = ROOT / "data_1"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TASKS = ["yolo_det", "yolo_stigma_seg", "yolo_pollination_pose"]

if DST.exists():
    raise SystemExit(f"{DST} already exists. Please remove it first: rm -rf {DST}")

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

    if "stigma" in label:
        return False
    if "pollination" in label:
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
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            if p.stem in out:
                raise RuntimeError(f"Duplicate stem in {folder}: {p.stem}")
            out[p.stem] = p
    return out

def load_raw(split):
    img_map = files_by_stem(SRC / split / "images", IMG_EXTS)
    ann_map = files_by_stem(SRC / split / "annotations_with_ids", {".json"})

    out = {}
    for stem in sorted(set(img_map) & set(ann_map)):
        out[stem] = {
            "image": img_map[stem],
            "annotation": ann_map[stem],
            "instances": count_instances(ann_map[stem]),
            "source_split": split,
        }
    return out

print("Loading raw data and counting instances...")
raw = {
    "train": load_raw("train"),
    "val": load_raw("val"),
    "test": load_raw("test"),
}

train_n = len(raw["train"])
val_n = len(raw["val"])
test_n = len(raw["test"])

train_inst = sum(x["instances"] for x in raw["train"].values())
val_inst = sum(x["instances"] for x in raw["val"].values())
test_inst = sum(x["instances"] for x in raw["test"].values())

print("\nBefore balancing:")
print(f"Train: images={train_n}, instances={train_inst}, avg={train_inst/train_n:.3f}")
print(f"Val:   images={val_n}, instances={val_inst}, avg={val_inst/val_n:.3f}")
print(f"Test:  images={test_n}, instances={test_inst}, avg={test_inst/test_n:.3f}")

target_avg = (train_inst + test_inst) / (train_n + test_n)
target_test_inst = round(target_avg * test_n)
needed_reduction = test_inst - target_test_inst

print(f"\nTarget shared Train/Test avg: {target_avg:.3f}")
print(f"Target test instances: {target_test_inst}")
print(f"Needed test-instance reduction: {needed_reduction}")

train_sorted = sorted(raw["train"].items(), key=lambda kv: kv[1]["instances"])
test_sorted = sorted(raw["test"].items(), key=lambda kv: kv[1]["instances"], reverse=True)

swaps = []
reduction = 0

i = 0
j = 0

while i < len(test_sorted) and j < len(train_sorted):
    test_stem, test_item = test_sorted[i]
    train_stem, train_item = train_sorted[j]

    diff = test_item["instances"] - train_item["instances"]

    if diff <= 0:
        break

    current_gap = abs(reduction - needed_reduction)
    next_gap = abs((reduction + diff) - needed_reduction)

    if next_gap > current_gap:
        break

    swaps.append({
        "train_to_test_stem": train_stem,
        "test_to_train_stem": test_stem,
        "train_instances": train_item["instances"],
        "test_instances": test_item["instances"],
        "reduction": diff,
    })

    reduction += diff
    i += 1
    j += 1

train_stems = set(raw["train"])
val_stems = set(raw["val"])
test_stems = set(raw["test"])

for s in swaps:
    train_stems.remove(s["train_to_test_stem"])
    test_stems.remove(s["test_to_train_stem"])
    train_stems.add(s["test_to_train_stem"])
    test_stems.add(s["train_to_test_stem"])

def get_raw_item(stem):
    for split in ["train", "val", "test"]:
        if stem in raw[split]:
            return raw[split][stem]
    raise KeyError(stem)

def stats(stems):
    inst = sum(get_raw_item(s)["instances"] for s in stems)
    return len(stems), inst, inst / len(stems)

train_img_n, train_inst_n, train_avg = stats(train_stems)
val_img_n, val_inst_n, val_avg = stats(val_stems)
test_img_n, test_inst_n, test_avg = stats(test_stems)

print("\nAfter balancing:")
print(f"Train: images={train_img_n}, instances={train_inst_n}, avg={train_avg:.3f}")
print(f"Val:   images={val_img_n}, instances={val_inst_n}, avg={val_avg:.3f}")
print(f"Test:  images={test_img_n}, instances={test_inst_n}, avg={test_avg:.3f}")
print(f"\nSwap pairs: {len(swaps)}")
print(f"Actual test-instance reduction: {reduction}")

def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def copy_raw_split(split, stems):
    print(f"\n[RAW] Copying {split}: {len(stems)} images")
    for stem in sorted(stems):
        item = get_raw_item(stem)
        copy_file(item["image"], DST / split / "images" / item["image"].name)
        copy_file(item["annotation"], DST / split / "annotations_with_ids" / item["annotation"].name)

copy_raw_split("train", train_stems)
copy_raw_split("val", val_stems)
copy_raw_split("test", test_stems)

print("\nRaw data_1 completed.")

# Build maps for YOLO-style dirs.
print("\nLoading YOLO-style source maps...")
yolo = {}
for task in TASKS:
    yolo[task] = {}
    for split in ["train", "val", "test"]:
        yolo[task][split] = {
            "images": files_by_stem(SRC / task / "images" / split, IMG_EXTS),
            "labels": files_by_stem(SRC / task / "labels" / split, {".txt"}),
        }

def find_yolo_file(task, stem, kind):
    for split in ["train", "val", "test"]:
        if stem in yolo[task][split][kind]:
            return yolo[task][split][kind][stem]
    return None

def copy_yolo_task(task, split, stems):
    copied = 0
    missing = 0
    print(f"\n[{task}] Copying {split}: target raw samples={len(stems)}")
    for stem in sorted(stems):
        img = find_yolo_file(task, stem, "images")
        lab = find_yolo_file(task, stem, "labels")

        if img is None or lab is None:
            missing += 1
            continue

        copy_file(img, DST / task / "images" / split / img.name)
        copy_file(lab, DST / task / "labels" / split / lab.name)
        copied += 1

    print(f"[{task}] {split}: copied={copied}, missing={missing}")
    return copied, missing

task_report = {}
for task in TASKS:
    task_report[task] = {}
    for split, stems in [
        ("train", train_stems),
        ("val", val_stems),
        ("test", test_stems),
    ]:
        copied, missing = copy_yolo_task(task, split, stems)
        task_report[task][split] = {
            "copied": copied,
            "missing": missing,
        }

yaml_map = {
    "yolo_det/polliflower_det.yaml": "yolo_det",
    "yolo_stigma_seg/polliflower_stigma_seg.yaml": "yolo_stigma_seg",
    "yolo_pollination_pose/polliflower_pollination_pose.yaml": "yolo_pollination_pose",
}

print("\nWriting YAML files...")
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
    print(dst_yaml)

manifest = {
    "source": str(SRC),
    "output": str(DST),
    "strategy": "Validation split unchanged. Train/test balanced according to flower-instance density by swapping high-instance test images with low-instance train images.",
    "before": {
        "train": {"images": train_n, "instances": train_inst, "avg": train_inst / train_n},
        "val": {"images": val_n, "instances": val_inst, "avg": val_inst / val_n},
        "test": {"images": test_n, "instances": test_inst, "avg": test_inst / test_n},
    },
    "after": {
        "train": {"images": train_img_n, "instances": train_inst_n, "avg": train_avg},
        "val": {"images": val_img_n, "instances": val_inst_n, "avg": val_avg},
        "test": {"images": test_img_n, "instances": test_inst_n, "avg": test_avg},
    },
    "swap_pairs": len(swaps),
    "actual_test_instance_reduction": reduction,
    "swaps": swaps,
    "yolo_task_report": task_report,
}

manifest_path = DST / "balanced_split_manifest.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"\nSaved manifest: {manifest_path}")

print("\nFinal check:")
print("| Split | Images | Instances | Avg. Inst./Image |")
print("|---|---:|---:|---:|")
print(f"| Train | {train_img_n} | {train_inst_n} | {train_avg:.3f} |")
print(f"| Val | {val_img_n} | {val_inst_n} | {val_avg:.3f} |")
print(f"| Test | {test_img_n} | {test_inst_n} | {test_avg:.3f} |")
print(f"| Total | {train_img_n + val_img_n + test_img_n} | {train_inst_n + val_inst_n + test_inst_n} | {(train_inst_n + val_inst_n + test_inst_n)/(train_img_n + val_img_n + test_img_n):.3f} |")
