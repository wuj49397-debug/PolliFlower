import argparse
import json
import re
from pathlib import Path
from collections import Counter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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

def is_flower_instance(shape):
    label = str(shape.get("label", "")).lower()
    shape_type = str(shape.get("shape_type", "")).lower()
    attrs = shape.get("attributes", {}) or {}
    role = str(attrs.get("instance_role", "")).lower()

    if "stigma" in label:
        return False
    if "pollination" in label:
        return False

    return shape_type == "rectangle" or role == "rectangle"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    root = Path(args.data)
    splits = ["train", "val", "test"]

    summary = {}
    split_files = {}
    split_groups = {}

    for split in splits:
        img_dir = root / split / "images"
        ann_dir = root / split / "annotations_with_ids"

        images = sorted([
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ])

        instances = 0
        missing = 0
        labels = Counter()
        groups = set()

        for img in images:
            groups.add(base_stem(img.stem))
            ann = ann_dir / f"{img.stem}.json"

            if not ann.exists():
                missing += 1
                continue

            data = json.loads(ann.read_text(encoding="utf-8"))
            for shape in data.get("shapes", []):
                if is_flower_instance(shape):
                    instances += 1
                    labels[str(shape.get("label", "unknown"))] += 1

        summary[split] = {
            "images": len(images),
            "base_groups": len(groups),
            "instances": instances,
            "avg_instances_per_image": instances / len(images) if images else 0,
            "missing_annotations": missing,
            "labels": dict(labels),
        }

        split_files[split] = {p.name for p in images}
        split_groups[split] = groups

    print("| Split | Images | Base groups | Instances | Avg. Inst./Image | Missing annotations |")
    print("|---|---:|---:|---:|---:|---:|")

    total_images = 0
    total_instances = 0
    total_missing = 0
    total_groups = set()

    for split in splits:
        s = summary[split]
        total_images += s["images"]
        total_instances += s["instances"]
        total_missing += s["missing_annotations"]
        total_groups |= split_groups[split]

        print(
            f"| {split.capitalize()} | {s['images']} | {s['base_groups']} | "
            f"{s['instances']} | {s['avg_instances_per_image']:.3f} | "
            f"{s['missing_annotations']} |"
        )

    print(
        f"| Total | {total_images} | {len(total_groups)} | "
        f"{total_instances} | {total_instances / total_images:.3f} | {total_missing} |"
    )

    print("\nExact filename overlap:")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        print(f"{a}-{b}: {len(split_files[a] & split_files[b])}")

    print("\nBase-group overlap:")
    total_overlap = 0
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = split_groups[a] & split_groups[b]
        total_overlap += len(overlap)
        print(f"{a}-{b}: {len(overlap)}")

    if total_overlap == 0:
        print("\nConclusion: no base-group leakage detected.")
    else:
        print(f"\nConclusion: detected {total_overlap} base-group overlaps across splits.")

    out = root / "split_stats.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved split statistics to: {out}")

if __name__ == "__main__":
    main()
