import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def polygon_to_mask(poly, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(poly) < 3:
        return mask
    pts = np.asarray(poly, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    pts = pts.astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def load_gt_masks(label_path, h, w):
    masks = []

    if not label_path.exists():
        return masks

    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 7:
            continue

        cls = int(float(parts[0]))
        if cls != 0:
            continue

        coords = [float(x) for x in parts[1:]]
        if len(coords) % 2 != 0:
            continue

        poly = []
        for i in range(0, len(coords), 2):
            x = coords[i] * w
            y = coords[i + 1] * h
            poly.append([x, y])

        masks.append(polygon_to_mask(poly, h, w))

    return masks


def masks_iou_and_coverage(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    gt_area = gt.sum()

    if union == 0:
        iou = 0.0
    else:
        iou = inter / union

    if gt_area == 0:
        coverage = 0.0
    else:
        coverage = inter / gt_area

    return float(iou), float(coverage)


def greedy_match(gt_masks, pred_masks):
    matched_pred = set()
    per_gt = []

    for gt in gt_masks:
        best_iou = 0.0
        best_cov = 0.0
        best_j = -1

        for j, pred in enumerate(pred_masks):
            if j in matched_pred:
                continue

            iou, cov = masks_iou_and_coverage(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_cov = cov
                best_j = j

        if best_j >= 0:
            matched_pred.add(best_j)

        strict_iou = best_iou if best_cov >= 0.8 else 0.0

        per_gt.append({
            "iou": best_iou,
            "coverage": best_cov,
            "strict_iou": strict_iou,
            "matched": best_j >= 0,
        })

    return per_gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", default="/root/autodl-tmp/flower_baseline/data/yolo_stigma_seg/images/test")
    parser.add_argument("--labels", default="/root/autodl-tmp/flower_baseline/data/yolo_stigma_seg/labels/test")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default="/root/autodl-tmp/flower_baseline/outputs/yolov8_stigma_seg_eval/yolov8s_seg_1024_test/iou_strictiou.json")
    args = parser.parse_args()

    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

    model = YOLO(args.model)

    all_items = []
    total_gt = 0
    total_pred = 0

    for idx, img_path in enumerate(image_paths, 1):
        with Image.open(img_path) as im:
            w, h = im.size

        label_path = label_dir / f"{img_path.stem}.txt"
        gt_masks = load_gt_masks(label_path, h, w)

        result = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            retina_masks=True,
            verbose=False,
        )[0]

        pred_masks = []
        if result.masks is not None:
            for poly in result.masks.xy:
                pred_masks.append(polygon_to_mask(poly, h, w))

        per_gt = greedy_match(gt_masks, pred_masks)

        total_gt += len(gt_masks)
        total_pred += len(pred_masks)

        for item in per_gt:
            item["image"] = img_path.name
            all_items.append(item)

        if idx % 100 == 0 or idx == len(image_paths):
            print(f"Processed {idx}/{len(image_paths)} images | GT={total_gt} | Pred={total_pred}", flush=True)

    if len(all_items) == 0:
        mean_iou = 0.0
        mean_strict_iou = 0.0
        mean_cov = 0.0
    else:
        mean_iou = float(np.mean([x["iou"] for x in all_items]))
        mean_strict_iou = float(np.mean([x["strict_iou"] for x in all_items]))
        mean_cov = float(np.mean([x["coverage"] for x in all_items]))

    summary = {
        "images": len(image_paths),
        "gt_instances": total_gt,
        "pred_instances": total_pred,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "mean_iou": mean_iou,
        "mean_coverage": mean_cov,
        "mean_strict_iou": mean_strict_iou,
    }

    out = {
        "summary": summary,
        "per_instance": all_items,
    }

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\nFinal Results")
    print(f"Images: {summary['images']}")
    print(f"GT instances: {summary['gt_instances']}")
    print(f"Pred instances: {summary['pred_instances']}")
    print(f"IoU: {summary['mean_iou']:.4f}")
    print(f"StrictIoU: {summary['mean_strict_iou']:.4f}")
    print(f"Mean coverage: {summary['mean_coverage']:.4f}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
