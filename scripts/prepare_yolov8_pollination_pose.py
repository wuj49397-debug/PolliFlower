import json
import shutil
from pathlib import Path
from PIL import Image

ROOT = Path("/root/autodl-tmp/flower_baseline")
SRC_DATA = ROOT / "data"
OUT_DATA = ROOT / "data" / "yolo_pollination_pose"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def get_instance_id(shape):
    attrs = shape.get("attributes", {}) or {}
    if "instance_id" in attrs:
        return attrs["instance_id"]
    if shape.get("group_id", None) is not None:
        return shape.get("group_id")
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

def is_pollination_point(shape):
    label = str(shape.get("label", "")).lower().strip()
    shape_type = str(shape.get("shape_type", "")).lower().strip()
    return label == "pollination point" and shape_type == "point"

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

def box_from_points(points, width, height):
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

def point_from_shape(shape, width, height):
    pts = shape.get("points", [])
    if len(pts) < 1:
        return None

    x = float(pts[0][0])
    y = float(pts[0][1])

    x = min(max(x, 0.0), float(width))
    y = min(max(y, 0.0), float(height))

    return x / width, y / height

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
    missing_pair = 0

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

        boxes = {}
        points = {}

        for shape in data.get("shapes", []):
            iid = get_instance_id(shape)
            if iid is None:
                continue
            iid = str(iid)

            if is_flower_box(shape):
                box = box_from_points(shape.get("points", []), width, height)
                if box is not None:
                    boxes[iid] = box

            elif is_pollination_point(shape):
                pt = point_from_shape(shape, width, height)
                if pt is not None:
                    points[iid] = pt

        lines = []
        all_ids = sorted(set(boxes.keys()) | set(points.keys()), key=lambda x: int(float(x)) if x.replace(".", "", 1).isdigit() else x)

        for iid in all_ids:
            if iid not in boxes or iid not in points:
                missing_pair += 1
                continue

            xc, yc, bw, bh = boxes[iid]
            px, py = points[iid]

            # class box keypoint_x keypoint_y visibility
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {px:.6f} {py:.6f} 2")

        dst_img = out_img_dir / img_src.name
        if not dst_img.exists():
            shutil.copy2(img_src, dst_img)

        label_path = out_lab_dir / (img_src.stem + ".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        image_count += 1
        inst_count += len(lines)
        if not lines:
            empty_count += 1

    print(f"{split}: images={image_count}, pose_instances={inst_count}, empty_labels={empty_count}, missing_images={missing_img}, missing_pairs={missing_pair}")

def main():
    for split in ["train", "val", "test"]:
        convert_split(split)

    yaml_path = OUT_DATA / "polliflower_pollination_pose.yaml"
    yaml_text = f"""path: {OUT_DATA}
train: images/train
val: images/val
test: images/test

kpt_shape: [1, 3]
flip_idx: [0]

names:
  0: flower
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"Saved dataset yaml to: {yaml_path}")

if __name__ == "__main__":
    main()
