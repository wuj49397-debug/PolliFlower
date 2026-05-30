import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/flower_baseline/data_1")
SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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

def main():
    rows = []
    total_images = 0
    total_instances = 0

    for split in SPLITS:
        img_dir = ROOT / split / "images"
        ann_dir = ROOT / split / "annotations_with_ids"

        images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
        anns = sorted(ann_dir.glob("*.json"))

        instances = 0
        missing_ann = 0

        for img in images:
            ann = ann_dir / f"{img.stem}.json"
            if not ann.exists():
                missing_ann += 1
                continue
            instances += count_instances(ann)

        avg = instances / len(images) if images else 0

        rows.append((split.capitalize(), len(images), instances, avg, missing_ann))
        total_images += len(images)
        total_instances += instances

    print("| Split | Images | Instances | Avg. Inst./Image |")
    print("|---|---:|---:|---:|")
    for split, images, instances, avg, missing_ann in rows:
        print(f"| {split} | {images} | {instances} | {avg:.3f} |")

    print(f"| Total | {total_images} | {total_instances} | {total_instances / total_images:.3f} |")

    print("\nMissing annotations:")
    for split, images, instances, avg, missing_ann in rows:
        print(f"{split}: {missing_ann}")

if __name__ == "__main__":
    main()
