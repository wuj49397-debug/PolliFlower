import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from train_keypoint_rcnn_pollination import (
    PollinationKeypointDataset,
    build_model,
    collate_fn,
    load_instances,
    greedy_match,
)


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = [float(x) for x in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def build_coco_gt(dataset):
    images = []
    annotations = []
    ann_id = 1

    for image_id, img_path in enumerate(dataset.images):
        with Image.open(img_path) as im:
            w, h = im.size

        images.append({
            "id": image_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        gt_items = load_instances(dataset.ann_dir / f"{img_path.stem}.json")

        for gt in gt_items:
            box_xywh = xyxy_to_xywh(gt["box"])
            x, y = gt["point"]

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": box_xywh,
                "area": box_xywh[2] * box_xywh[3],
                "iscrowd": 0,
                "keypoints": [float(x), float(y), 2],
                "num_keypoints": 1,
            })
            ann_id += 1

    coco = COCO()
    coco.dataset = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "flower",
                "keypoints": ["pollination_point"],
                "skeleton": [],
            }
        ],
    }
    coco.createIndex()
    return coco


def run_coco_eval(coco_gt, preds, iou_type, kpt_sigma=0.05):
    if len(preds) == 0:
        return 0.0, 0.0

    coco_dt = coco_gt.loadRes(preds)
    evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())

    if iou_type == "keypoints":
        evaluator.params.kpt_oks_sigmas = np.array([kpt_sigma], dtype=np.float32)

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    map_5095 = float(evaluator.stats[0])
    map_50 = float(evaluator.stats[1])
    return map_50, map_5095


@torch.no_grad()
def evaluate_all(model, dataset, loader, device, conf=0.25, box_iou_thr=0.5, kpt_sigma=0.05):
    model.eval()

    bbox_preds = []
    kpt_preds = []

    per_instance = []
    total_gt = 0
    total_pred_conf = 0
    matched = 0

    img_offset = 0

    for images, targets in loader:
        images_gpu = [img.to(device) for img in images]
        outputs = model(images_gpu)

        for bi, out in enumerate(outputs):
            image_id = img_offset + bi
            img_path = dataset.images[image_id]

            gt_items = load_instances(dataset.ann_dir / f"{img_path.stem}.json")

            scores = out["scores"].detach().cpu().numpy()
            boxes = out["boxes"].detach().cpu().numpy()
            keypoints = out["keypoints"].detach().cpu().numpy()

            pred_items_for_strict = []

            for score, box, kp in zip(scores, boxes, keypoints):
                score = float(score)
                box_xywh = xyxy_to_xywh(box)
                px, py = float(kp[0][0]), float(kp[0][1])

                if score >= 0.001:
                    bbox_preds.append({
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": box_xywh,
                        "score": score,
                    })

                    kpt_preds.append({
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": box_xywh,
                        "keypoints": [px, py, 2],
                        "score": score,
                    })

                if score >= conf:
                    pred_items_for_strict.append({
                        "box": box.astype(np.float32),
                        "point": np.array([px, py], dtype=np.float32),
                        "score": score,
                    })

            matches = greedy_match(pred_items_for_strict, gt_items, box_iou_thr)

            total_gt += len(gt_items)
            total_pred_conf += len(pred_items_for_strict)

            for gt, pred, match_iou in matches:
                if pred is None:
                    per_instance.append({
                        "matched": False,
                        "dist": None,
                        "strict_dist": None,
                        "box_iou": float(match_iou),
                    })
                else:
                    dist = float(np.linalg.norm(pred["point"] - gt["point"]))
                    strict_dist = dist / math.sqrt(max(gt["stigma_area"], 1e-6))
                    matched += 1

                    per_instance.append({
                        "matched": True,
                        "dist": dist,
                        "strict_dist": strict_dist,
                        "box_iou": float(match_iou),
                    })

        img_offset += len(images)

    coco_gt = build_coco_gt(dataset)

    box_map50, box_map5095 = run_coco_eval(coco_gt, bbox_preds, "bbox")
    pose_map50, pose_map5095 = run_coco_eval(coco_gt, kpt_preds, "keypoints", kpt_sigma=kpt_sigma)

    matched_items = [x for x in per_instance if x["matched"]]

    mpe = float(np.mean([x["dist"] for x in matched_items])) if matched_items else 0.0
    strict_dist = float(np.mean([x["strict_dist"] for x in matched_items])) if matched_items else 0.0

    pck005 = sum(1 for x in per_instance if x["matched"] and x["strict_dist"] <= 0.05) / max(len(per_instance), 1)
    pck010 = sum(1 for x in per_instance if x["matched"] and x["strict_dist"] <= 0.10) / max(len(per_instance), 1)

    return {
        "images": len(dataset),
        "gt_instances": total_gt,
        "pred_instances_conf025": total_pred_conf,
        "matched_instances": matched,
        "conf": conf,
        "box_iou_threshold": box_iou_thr,
        "Box mAP50": box_map50,
        "Box mAP50-95": box_map5095,
        "Pose mAP50": pose_map50,
        "Pose mAP50-95": pose_map5095,
        "MPE": mpe,
        "StrictDist": strict_dist,
        "StrictPCK@0.05": pck005,
        "StrictPCK@0.10": pck010,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/root/autodl-tmp/flower_baseline/data")
    parser.add_argument("--pretrained", default="/root/autodl-tmp/flower_baseline/checkpoints/keypointrcnn_resnet50_fpn_coco.pth")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--box-iou", type=float, default=0.5)
    parser.add_argument("--kpt-sigma", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = PollinationKeypointDataset(args.data, "test", hflip=0.0, filter_empty=False)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    model = build_model(args.pretrained, num_classes=2, num_keypoints=1, imgsz=args.imgsz)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    results = evaluate_all(
        model=model,
        dataset=dataset,
        loader=loader,
        device=device,
        conf=args.conf,
        box_iou_thr=args.box_iou,
        kpt_sigma=args.kpt_sigma,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Final Test Results")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
