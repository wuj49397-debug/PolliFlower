import json
import shutil
from pathlib import Path
from PIL import Image

ROOT = Path("/root/autodl-tmp/flower_baseline")
SRC_DATA = ROOT / "data"
OUT_DATA = ROOT / "data" / "yolo_stigma_seg"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def is_stigma_polygon(shape):
    label = str(shape.get("label", "")).lower().strip()
    shape_type = str(shape.get("shape_type", "")).lower().strip()
    return label == "stigma region" and shape_type == "polygon"

def find_image(image_dir, image_path):
    p = image_dir / image_path
    if p.exists():
        return p

    stem = Path(image_path).stem
    for ext in IMG_EXTS:
        q = image_dir / f"{stem}{ext}"
        if q.exists():
            return q

    matches = [m for m in image_dir.glob(stem + ".*") if m.suffix.lower() in IMG_EXTS]
    if matches:
        return matches[0]

    return None

def polygon_to_yolo(points, width, height):
    coords = []
    for p in points:
        if len(p) < 2:
            continue
        x = float(p[0])
        y = float(p[1])
        x = min(max(x, 0.0), float(width))
        y = min(max(y, 0.0), float(height))
        coords.append(x / width)
        coords.append(y / height)

    if len(coords) < 6:
        return None

    return coords

def convert_split(split):
    ann_dir = SRC_DATA / split / "annotations_with_ids"
    img_dir = SRC_DATA / split / "images"

    out_img_dir = OUT_DATA / "images" / split
    out_lab_dir = OUT_DATA / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lab_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(ann_dir.glob("*.json"))

    image_count = 0
    inst_count = 0
    empty_count = 0
    missing_img = 0

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_path = data.get("imagePath", jf.with_suffix(".png").name)
        img_src = find_image(img_dir, image_path)
        if img_src is None:
            print(f"[WARN] image not found for {jf.name}: {image_path}")
            missing_img += 1
            continue

        width = int(data.get("imageWidth", 0))
        height = int(data.get("imageHeight", 0))
        if width <= 0 or height <= 0:
            with Image.open(img_src) as im:
                width, height = im.size

        lines = []
        for shape in data.get("shapes", []):
            if not is_stigma_polygon(shape):
                continue

            pts = shape.get("points", [])
            coords = polygon_to_yolo(pts, width, height)
            if coords is None:
                continue

            coord_text = " ".join(f"{v:.6f}" for v in coords)
            lines.append(f"0 {coord_text}")

        dst_img = out_img_dir / img_src.name
        if not dst_img.exists():
            shutil.copy2(img_src, dst_img)

        label_path = out_lab_dir / (img_src.stem + ".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        image_count += 1
        inst_count += len(lines)
        if not lines:
            empty_count += 1

    print(f"{split}: images={image_count}, stigma_instances={inst_count}, empty_labels={empty_count}, missing_images={missing_img}")

def main():
    for split in ["train", "val", "test"]:
        convert_split(split)

    yaml_path = OUT_DATA / "polliflower_stigma_seg.yaml"
    yaml_text = f"""path: {OUT_DATA}
train: images/train
val: images/val
test: images/test

names:
  0: stigma
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"Saved dataset yaml to: {yaml_path}")

if __name__ == "__main__":
    main()
