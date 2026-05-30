import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.transforms import functional as F


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
    if "stigma" in label or "pollination" in label:
        return False
    return shape_type == "rectangle" or role == "rectangle"


def is_pollination_point(shape):
    return str(shape.get("label", "")).lower().strip() == "pollination point" and str(shape.get("shape_type", "")).lower().strip() == "point"


def is_stigma_polygon(shape):
    return str(shape.get("label", "")).lower().strip() == "stigma region" and str(shape.get("shape_type", "")).lower().strip() == "polygon"


def box_from_points(points, width, height):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(width), max(xs))
    y2 = min(float(height), max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def point_from_shape(shape, width, height):
    pts = shape.get("points", [])
    if len(pts) < 1:
        return None
    x = min(max(float(pts[0][0]), 0.0), float(width))
    y = min(max(float(pts[0][1]), 0.0), float(height))
    return np.array([x, y], dtype=np.float32)


def stigma_area_from_shape(shape, width, height):
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
    cv2.fillPoly(mask, [np.asarray(pts, dtype=np.int32)], 1)
    return float(mask.sum())


def load_instances(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    width = int(data["imageWidth"])
    height = int(data["imageHeight"])

    boxes, points, stigma_areas = {}, {}, {}

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
        elif is_stigma_polygon(shape):
            area = stigma_area_from_shape(shape, width, height)
            if area > 0:
                stigma_areas[iid] = area

    instances = []
    valid_ids = set(boxes.keys()) & set(points.keys()) & set(stigma_areas.keys())
    for iid in sorted(valid_ids, key=lambda x: int(float(x)) if x.replace(".", "", 1).isdigit() else x):
        instances.append({
            "instance_id": iid,
            "box": boxes[iid],
            "point": points[iid],
            "stigma_area": stigma_areas[iid],
        })

    return instances


def box_iou(box1, box2):
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class PollinationKeypointDataset(Dataset):
    def __init__(self, root, split, hflip=0.0, filter_empty=False):
        self.root = Path(root)
        self.split = split
        self.hflip = hflip
        self.image_dir = self.root / split / "images"
        self.ann_dir = self.root / split / "annotations_with_ids"

        self.images = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

        if filter_empty:
            kept = []
            for p in self.images:
                jp = self.ann_dir / f"{p.stem}.json"
                if jp.exists() and len(load_instances(jp)) > 0:
                    kept.append(p)
            self.images = kept

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        json_path = self.ann_dir / f"{img_path.stem}.json"

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        instances = load_instances(json_path)

        boxes = []
        keypoints = []
        areas = []
        stigma_areas = []
        gt_points = []

        for ins in instances:
            boxes.append(ins["box"])
            keypoints.append([[ins["point"][0], ins["point"][1], 2.0]])
            gt_points.append(ins["point"])
            stigma_areas.append(ins["stigma_area"])
            areas.append((ins["box"][2] - ins["box"][0]) * (ins["box"][3] - ins["box"][1]))

        if self.split == "train" and self.hflip > 0 and random.random() < self.hflip:
            image = ImageOps.mirror(image)
            for b in boxes:
                old_x1, old_x2 = b[0], b[2]
                b[0] = w - old_x2
                b[2] = w - old_x1
            for kp in keypoints:
                kp[0][0] = w - kp[0][0]
            for pt in gt_points:
                pt[0] = w - pt[0]

        image = F.to_tensor(image)

        if len(boxes) > 0:
            boxes = torch.as_tensor(np.asarray(boxes), dtype=torch.float32)
            keypoints = torch.as_tensor(np.asarray(keypoints), dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            stigma_areas = torch.as_tensor(stigma_areas, dtype=torch.float32)
            gt_points = torch.as_tensor(np.asarray(gt_points), dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            keypoints = torch.zeros((0, 1, 3), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            stigma_areas = torch.zeros((0,), dtype=torch.float32)
            gt_points = torch.zeros((0, 2), dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "keypoints": keypoints,
            "image_id": torch.tensor([idx]),
            "area": areas,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
            "stigma_areas": stigma_areas,
            "gt_points": gt_points,
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(pretrained_path, num_classes=2, num_keypoints=1, imgsz=1024):
    model = keypointrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=imgsz,
        max_size=imgsz,
    )

    if pretrained_path and Path(pretrained_path).exists():
        state = torch.load(pretrained_path, map_location="cpu")
        current = model.state_dict()
        compatible = {k: v for k, v in state.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
        current.update(compatible)
        model.load_state_dict(current)
        print(f"Loaded compatible COCO pretrained weights: {len(compatible)}/{len(current)} tensors")
    else:
        print("WARNING: pretrained weights not found. Training from scratch.")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_channels = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_channels, num_keypoints)

    return model


def greedy_match(pred_items, gt_items, iou_thr):
    matched_pred = set()
    results = []

    for gt in gt_items:
        best_j, best_iou = -1, 0.0
        for j, pred in enumerate(pred_items):
            if j in matched_pred:
                continue
            iou = box_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_j >= 0 and best_iou >= iou_thr:
            matched_pred.add(best_j)
            results.append((gt, pred_items[best_j], best_iou))
        else:
            results.append((gt, None, best_iou))

    return results


@torch.no_grad()
def evaluate_strict(model, dataset, loader, device, conf=0.25, box_iou_thr=0.5):
    model.eval()

    per_instance = []
    total_gt, total_pred, matched = 0, 0, 0

    img_offset = 0
    for images, targets in loader:
        images_gpu = [img.to(device) for img in images]
        outputs = model(images_gpu)

        for bi, out in enumerate(outputs):
            img_path = dataset.images[img_offset + bi]
            gt_items = load_instances(dataset.ann_dir / f"{img_path.stem}.json")

            scores = out["scores"].detach().cpu().numpy()
            boxes = out["boxes"].detach().cpu().numpy()
            kpts = out["keypoints"].detach().cpu().numpy()

            pred_items = []
            for score, box, kp in zip(scores, boxes, kpts):
                if float(score) < conf:
                    continue
                pred_items.append({
                    "box": box.astype(np.float32),
                    "point": np.array([kp[0][0], kp[0][1]], dtype=np.float32),
                    "score": float(score),
                })

            matches = greedy_match(pred_items, gt_items, box_iou_thr)

            total_gt += len(gt_items)
            total_pred += len(pred_items)

            for gt, pred, match_iou in matches:
                if pred is None:
                    per_instance.append({"matched": False, "dist": None, "strict_dist": None, "box_iou": float(match_iou)})
                else:
                    dist = float(np.linalg.norm(pred["point"] - gt["point"]))
                    strict_dist = dist / math.sqrt(max(gt["stigma_area"], 1e-6))
                    matched += 1
                    per_instance.append({"matched": True, "dist": dist, "strict_dist": strict_dist, "box_iou": float(match_iou)})

        img_offset += len(images)

    matched_items = [x for x in per_instance if x["matched"]]

    mpe = float(np.mean([x["dist"] for x in matched_items])) if matched_items else 0.0
    strict_dist = float(np.mean([x["strict_dist"] for x in matched_items])) if matched_items else 0.0

    pck005 = sum(1 for x in per_instance if x["matched"] and x["strict_dist"] <= 0.05) / max(len(per_instance), 1)
    pck010 = sum(1 for x in per_instance if x["matched"] and x["strict_dist"] <= 0.10) / max(len(per_instance), 1)

    return {
        "gt_instances": total_gt,
        "pred_instances": total_pred,
        "matched_instances": matched,
        "MPE": mpe,
        "StrictDist": strict_dist,
        "StrictPCK@0.05": pck005,
        "StrictPCK@0.10": pck010,
    }


def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train()
    running = 0.0
    start = time.time()

    for step, (images, targets) in enumerate(loader, 1):
        images = [img.to(device) for img in images]
        train_targets = []
        for t in targets:
            train_targets.append({k: v.to(device) for k, v in t.items() if k not in {"stigma_areas", "gt_points"}})

        loss_dict = model(images, train_targets)
        loss = sum(v for v in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        running += float(loss.item())

        if step % 50 == 0 or step == len(loader):
            print(f"Epoch {epoch} | {step}/{len(loader)} | loss {running / step:.4f} | {time.time() - start:.1f}s", flush=True)

    return running / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/root/autodl-tmp/flower_baseline/data")
    parser.add_argument("--pretrained", default="/root/autodl-tmp/flower_baseline/checkpoints/keypointrcnn_resnet50_fpn_coco.pth")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--box-iou", type=float, default=0.5)
    parser.add_argument("--output", default="/root/autodl-tmp/flower_baseline/outputs/keypoint_rcnn_pollination/keypointrcnn_r50_fpn_1024")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = PollinationKeypointDataset(args.data, "train", hflip=0.5, filter_empty=True)
    val_set = PollinationKeypointDataset(args.data, "val", hflip=0.0, filter_empty=False)
    test_set = PollinationKeypointDataset(args.data, "test", hflip=0.0, filter_empty=False)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    print(f"Train images: {len(train_set)} | Val images: {len(val_set)} | Test images: {len(test_set)}")

    model = build_model(args.pretrained, num_classes=2, num_keypoints=1, imgsz=args.imgsz)
    model.to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[27, 36], gamma=0.1)

    best_score = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()

        val_res = evaluate_strict(model, val_set, val_loader, device, conf=args.conf, box_iou_thr=args.box_iou)
        val_res["epoch"] = epoch
        val_res["train_loss"] = loss
        history.append(val_res)

        print(
            f"Epoch {epoch} Val | MPE {val_res['MPE']:.4f} | StrictDist {val_res['StrictDist']:.4f} | "
            f"PCK@0.05 {val_res['StrictPCK@0.05']:.4f} | PCK@0.10 {val_res['StrictPCK@0.10']:.4f}",
            flush=True,
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val": val_res,
            "args": vars(args),
        }

        torch.save(ckpt, out_dir / "last.pt")
        torch.save(ckpt, out_dir / f"epoch_{epoch:03d}.pt")

        if val_res["StrictPCK@0.10"] > best_score:
            best_score = val_res["StrictPCK@0.10"]
            torch.save(ckpt, out_dir / "best.pt")
            print(f"Saved best checkpoint at epoch {epoch}: StrictPCK@0.10={best_score:.4f}", flush=True)

        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("Loading best checkpoint for strict test evaluation...")
    best = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])

    test_res = evaluate_strict(model, test_set, test_loader, device, conf=args.conf, box_iou_thr=args.box_iou)
    (out_dir / "test_results.json").write_text(json.dumps(test_res, indent=2), encoding="utf-8")

    print("Final Strict Test Results")
    print(json.dumps(test_res, indent=2))


if __name__ == "__main__":
    main()
