import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


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


def is_stigma_polygon(shape):
    label = str(shape.get("label", "")).lower().strip()
    shape_type = str(shape.get("shape_type", "")).lower().strip()
    return label == "stigma region" and shape_type == "polygon"


def box_xyxy_from_points(points, width, height):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(width), max(xs))
    y2 = min(float(height), max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def point_xy_from_shape(shape, width, height):
    pts = shape.get("points", [])
    if len(pts) < 1:
        return None
    x = min(max(float(pts[0][0]), 0.0), float(width))
    y = min(max(float(pts[0][1]), 0.0), float(height))
    return np.array([x, y], dtype=np.float32)


def polygon_area_from_shape(shape, width, height):
    pts = []
    for p in shape.get("points", []):
        if len(p) < 2:
            continue
        x = min(max(float(p[0]), 0.0), float(width - 1))
        y = min(max(float(p[1]), 0.0), float(height - 1))
        pts.append([x, y])

    if len(pts) < 3:
        return 0.0

    mask = np.zeros((height, width), dtype=np.uint8)
    arr = np.asarray(pts, dtype=np.int32)
    cv2.fillPoly(mask, [arr], 1)
    return float(mask.sum())


def box_iou(box1, box2):
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def load_gt_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    width = int(data.get("imageWidth"))
    height = int(data.get("imageHeight"))

    boxes = {}
    points = {}
    stigma_areas = {}

    for shape in data.get("shapes", []):
        iid = get_instance_id(shape)
        if iid is None:
            continue
        iid = str(iid)

        if is_flower_box(shape):
            box = box_xyxy_from_points(shape.get("points", []), width, height)
            if box is not None:
                boxes[iid] = box

        elif is_pollination_point(shape):
            pt = point_xy_from_shape(shape, width, height)
            if pt is not None:
                points[iid] = pt

        elif is_stigma_polygon(shape):
            area = polygon_area_from_shape(shape, width, height)
            if area > 0:
                stigma_areas[iid] = area

    gt_items = []
    for iid in sorted(set(boxes.keys()) & set(points.keys()) & set(stigma_areas.keys()), key=lambda x: int(float(x)) if x.replace(".", "", 1).isdigit() else x):
        gt_items.append({
            "instance_id": iid,
            "box": boxes[iid],
            "point": points[iid],
            "stigma_area": stigma_areas[iid],
        })

    return gt_items


def find_json(ann_dir, img_path):
    p = ann_dir / f"{img_path.stem}.json"
    if p.exists():
        return p
    matches = list(ann_dir.glob(img_path.stem + "*.json"))
    if matches:
        return matches[0]
    return None


def greedy_match(pred_items, gt_items, iou_thr):
    matched_pred = set()
    matches = []

    for gi, gt in enumerate(gt_items):
        best_j = -1
        best_iou = 0.0

        for pj, pred in enumerate(pred_items):
            if pj in matched_pred:
                continue
            iou = box_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_j = pj

        if best_j >= 0 and best_iou >= iou_thr:
            matched_pred.add(best_j)
            matches.append((gt, pred_items[best_j], best_iou))
        else:
            matches.append((gt, None, best_iou))

    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", default="/root/autodl-tmp/flower_baseline/data/test/images")
    parser.add_argument("--annotations", default="/root/autodl-tmp/flower_baseline/data/test/annotations_with_ids")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--box-iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    image_dir = Path(args.images)
    ann_dir = Path(args.annotations)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

    model = YOLO(args.model)

    per_instance = []
    total_gt = 0
    total_pred = 0
    matched = 0

    for idx, img_path in enumerate(image_paths, 1):
        json_path = find_json(ann_dir, img_path)
        if json_path is None:
            print(f"[WARN] missing json for {img_path.name}")
            continue

        gt_items = load_gt_from_json(json_path)

        result = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]

        pred_items = []

        if result.boxes is not None and result.keypoints is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            kpts = result.keypoints.xy.cpu().numpy()

            for box, score, kp in zip(boxes, scores, kpts):
                if len(kp) < 1:
                    continue
                pred_items.append({
                    "box": box.astype(np.float32),
                    "point": np.array([kp[0][0], kp[0][1]], dtype=np.float32),
                    "score": float(score),
                })

        matches = greedy_match(pred_items, gt_items, args.box_iou)

        total_gt += len(gt_items)
        total_pred += len(pred_items)

        for gt, pred, match_iou in matches:
            if pred is None:
                dist = None
                strict_dist = None
                is_matched = False
            else:
                dist = float(np.linalg.norm(pred["point"] - gt["point"]))
                strict_dist = dist / math.sqrt(max(gt["stigma_area"], 1e-6))
                is_matched = True
                matched += 1

            per_instance.append({
                "image": img_path.name,
                "instance_id": gt["instance_id"],
                "matched": is_matched,
                "box_iou": float(match_iou),
                "dist": dist,
                "strict_dist": strict_dist,
                "stigma_area": float(gt["stigma_area"]),
            })

        if idx % 100 == 0 or idx == len(image_paths):
            print(f"Processed {idx}/{len(image_paths)} images | GT={total_gt} | Pred={total_pred} | Matched={matched}", flush=True)

    matched_items = [x for x in per_instance if x["matched"]]
    all_items = per_instance

    # Unmatched GT instances are treated as failures for PCK.
    mpe = float(np.mean([x["dist"] for x in matched_items])) if matched_items else 0.0
    strict_dist = float(np.mean([x["strict_dist"] for x in matched_items])) if matched_items else 0.0

    strict_pck_005 = sum(1 for x in all_items if x["matched"] and x["strict_dist"] <= 0.05) / max(len(all_items), 1)
    strict_pck_010 = sum(1 for x in all_items if x["matched"] and x["strict_dist"] <= 0.10) / max(len(all_items), 1)

    # Also report matched-only PCK for analysis, but use full PCK in the paper table.
    matched_pck_005 = sum(1 for x in matched_items if x["strict_dist"] <= 0.05) / max(len(matched_items), 1)
    matched_pck_010 = sum(1 for x in matched_items if x["strict_dist"] <= 0.10) / max(len(matched_items), 1)

    summary = {
        "images": len(image_paths),
        "gt_instances": total_gt,
        "pred_instances": total_pred,
        "matched_instances": matched,
        "conf": args.conf,
        "box_iou_threshold": args.box_iou,
        "MPE_matched_px": mpe,
        "StrictDist_matched": strict_dist,
        "StrictPCK@0.05": strict_pck_005,
        "StrictPCK@0.10": strict_pck_010,
        "StrictPCK@0.05_matched_only": matched_pck_005,
        "StrictPCK@0.10_matched_only": matched_pck_010,
    }

    out = {
        "summary": summary,
        "per_instance": per_instance,
    }

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\nFinal Results")
    print(json.dumps(summary, indent=2))
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
