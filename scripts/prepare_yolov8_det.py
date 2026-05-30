import json
import shutil
from pathlib import Path
from PIL import Image

ROOT = Path("/root/autodl-tmp/flower_baseline")
SRC_DATA = ROOT / "data"
OUT_DATA = ROOT / "data" / "yolo_det"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def is_flower_box(shape):
    label = str(shape.get("label", "")).lower()
    shape_type = str(shape.get("shape_type", "")).lower()
    attrs = shape.get("attributes", {}) or {}
    role = str(attrs.get("instance_role", "")).lower()

    if label == "stigma region":
        return False
    if "stigma" in label:
        return False
    if "pollination" in label:
        return False

    if shape_type == "rectangle":
        return True
    if role == "rectangle":
        return True
    if "flower" in label and shape_type in {"rectangle", "polygon"}:
        return True

    return False

def find_image(image_dir, image_path):
    p = image_dir / image_path
    if p.exists():
        return p

    stem = Path(image_path).stem
    for ext in IMG_EXTS:
        q = image_dir / f"{stem}{ext}"
        if q.exists():
            return q

    matches = list(image_dir.glob(stem + ".*"))
    matches = [m for m in matches if m.suffix.lower() in IMG_EXTS]
    if matches:
        return matches[0]

    return None

def shape_to_box(points, width, height):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]

    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(width), max(xs))
    y2 = min(float(height), max(ys))

    if x2 <= x1 or y2 <= y1:
        return None

    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height

    return xc, yc, bw, bh

def convert_split(split):
    ann_dir = SRC_DATA / split / "annotations_with_ids"
    img_dir = SRC_DATA / split / "images"

    out_img_dir = OUT_DATA / "images" / split
    out_lab_dir = OUT_DATA / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lab_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(ann_dir.glob("*.json"))
    image_count = 0
    box_count = 0
    empty_count = 0

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_path = data.get("imagePath", jf.with_suffix(".png").name)
        img_src = find_image(img_dir, image_path)
        if img_src is None:
            print(f"[WARN] image not found for {jf.name}: {image_path}")
            continue

        width = int(data.get("imageWidth", 0))
        height = int(data.get("imageHeight", 0))

        if width <= 0 or height <= 0:
            with Image.open(img_src) as im:
                width, height = im.size

        lines = []
        for shape in data.get("shapes", []):
            if not is_flower_box(shape):
                continue

            pts = shape.get("points", [])
            if len(pts) < 2:
                continue

            box = shape_to_box(pts, width, height)
            if box is None:
                continue

            xc, yc, bw, bh = box
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        dst_img = out_img_dir / img_src.name
        if not dst_img.exists():
            shutil.copy2(img_src, dst_img)

        label_path = out_lab_dir / (img_src.stem + ".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        image_count += 1
        box_count += len(lines)
        if not lines:
            empty_count += 1

    print(f"{split}: images={image_count}, boxes={box_count}, empty_labels={empty_count}")

def main():
    for split in ["train", "val", "test"]:
        convert_split(split)

    yaml_path = OUT_DATA / "polliflower_det.yaml"
    yaml_text = f"""path: {OUT_DATA}
train: images/train
val: images/val
test: images/test

names:
  0: flower
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"Saved dataset yaml to: {yaml_path}")

if __name__ == "__main__":
    main()
