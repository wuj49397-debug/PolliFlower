import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def polygon_to_mask(points, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) < 3:
        return mask
    pts = np.asarray(points, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def mask_to_box(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


class StigmaSegDataset(Dataset):
    def __init__(self, root, split, hflip=0.0):
        self.root = Path(root)
        self.split = split
        self.hflip = hflip
        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        self.images = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        label_path = self.label_dir / f"{img_path.stem}.txt"
        masks = []
        boxes = []

        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                cls = int(float(parts[0]))
                if cls != 0:
                    continue
                coords = [float(x) for x in parts[1:]]
                if len(coords) % 2 != 0:
                    continue
                pts = []
                for i in range(0, len(coords), 2):
                    pts.append([coords[i] * w, coords[i + 1] * h])
                mask = polygon_to_mask(pts, h, w)
                box = mask_to_box(mask)
                if box is not None:
                    masks.append(mask)
                    boxes.append(box)

        image = F.to_tensor(image)

        if len(masks) > 0:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(masks),), dtype=torch.int64)
        else:
            masks = torch.zeros((0, h, w), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        if self.split == "train" and self.hflip > 0 and random.random() < self.hflip:
            image = torch.flip(image, dims=[2])
            masks = torch.flip(masks, dims=[2])
            if boxes.numel() > 0:
                old_x1 = boxes[:, 0].clone()
                old_x2 = boxes[:, 2].clone()
                boxes[:, 0] = w - old_x2
                boxes[:, 2] = w - old_x1

        area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes, pretrained_path, min_size=1024, max_size=1024):
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=min_size, max_size=max_size)

    if pretrained_path and Path(pretrained_path).exists():
        state = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        print(f"Loaded COCO pretrained weights from {pretrained_path}")
    else:
        print("WARNING: COCO pretrained weights not found. Training from scratch.")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, hidden_layer, num_classes)
    return model


def mask_iou_and_cov(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    gt_area = gt.sum()
    iou = inter / union if union > 0 else 0.0
    cov = inter / gt_area if gt_area > 0 else 0.0
    return float(iou), float(cov)


def greedy_iou(gt_masks, pred_masks):
    matched = set()
    ious, strict_ious, covs = [], [], []

    for gt in gt_masks:
        best_iou, best_cov, best_j = 0.0, 0.0, -1
        for j, pred in enumerate(pred_masks):
            if j in matched:
                continue
            iou, cov = mask_iou_and_cov(pred, gt)
            if iou > best_iou:
                best_iou, best_cov, best_j = iou, cov, j
        if best_j >= 0:
            matched.add(best_j)
        ious.append(best_iou)
        covs.append(best_cov)
        strict_ious.append(best_iou if best_cov >= 0.5 else 0.0)

    return ious, strict_ious, covs


@torch.no_grad()
def evaluate(model, loader, device, conf=0.25):
    model.eval()
    metric = MeanAveragePrecision(iou_type="segm", class_metrics=False)

    all_ious, all_strict, all_cov = [], [], []
    total_images, total_gt, total_pred = 0, 0, 0

    for images, targets in loader:
        images_gpu = [img.to(device) for img in images]
        outputs = model(images_gpu)

        preds_for_map = []
        targets_for_map = []

        for out, tgt in zip(outputs, targets):
            gt_masks = tgt["masks"].cpu().numpy().astype(np.uint8)
            gt_labels = tgt["labels"].cpu()

            masks_prob = out["masks"].detach().cpu()
            scores = out["scores"].detach().cpu()
            labels = out["labels"].detach().cpu()

            pred_bool_all = (masks_prob[:, 0] >= 0.5) if masks_prob.numel() > 0 else torch.zeros((0, *gt_masks.shape[1:]), dtype=torch.bool)

            preds_for_map.append({
                "masks": pred_bool_all,
                "scores": scores,
                "labels": labels,
            })
            targets_for_map.append({
                "masks": torch.as_tensor(gt_masks.astype(bool)),
                "labels": gt_labels,
            })

            keep = scores >= conf
            pred_masks = pred_bool_all[keep].numpy().astype(np.uint8)

            ious, strict, covs = greedy_iou(gt_masks, pred_masks)
            all_ious.extend(ious)
            all_strict.extend(strict)
            all_cov.extend(covs)

            total_images += 1
            total_gt += len(gt_masks)
            total_pred += len(pred_masks)

        metric.update(preds_for_map, targets_for_map)

    res = metric.compute()
    return {
        "images": total_images,
        "gt_instances": total_gt,
        "pred_instances_conf025": total_pred,
        "IoU": float(np.mean(all_ious)) if all_ious else 0.0,
        "StrictIoU": float(np.mean(all_strict)) if all_strict else 0.0,
        "MeanCoverage": float(np.mean(all_cov)) if all_cov else 0.0,
        "mAP@0.5": float(res["map_50"]),
        "mAP@0.5:0.95": float(res["map"]),
    }


def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train()
    running = 0.0
    start = time.time()

    for step, (images, targets) in enumerate(loader, 1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
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
    parser.add_argument("--data", default="/root/autodl-tmp/flower_baseline/data/yolo_stigma_seg")
    parser.add_argument("--pretrained", default="/root/autodl-tmp/flower_baseline/checkpoints/maskrcnn_resnet50_fpn_coco.pth")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--hflip", type=float, default=0.5)
    parser.add_argument("--output", default="/root/autodl-tmp/flower_baseline/outputs/mask_rcnn_stigma_seg/maskrcnn_r50_fpn_1024")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = StigmaSegDataset(args.data, "train", hflip=args.hflip)
    val_set = StigmaSegDataset(args.data, "val", hflip=0.0)
    test_set = StigmaSegDataset(args.data, "test", hflip=0.0)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    print(f"Train images: {len(train_set)} | Val images: {len(val_set)} | Test images: {len(test_set)}")

    model = build_model(num_classes=2, pretrained_path=args.pretrained, min_size=1024, max_size=1024)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[27, 36], gamma=0.1)

    best_map50 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()

        val_res = evaluate(model, val_loader, device, conf=0.25)
        val_res["epoch"] = epoch
        val_res["train_loss"] = loss
        history.append(val_res)

        print(f"Epoch {epoch} Val | IoU {val_res['IoU']:.4f} | StrictIoU {val_res['StrictIoU']:.4f} | mAP50 {val_res['mAP@0.5']:.4f} | mAP50-95 {val_res['mAP@0.5:0.95']:.4f}", flush=True)

        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "val": val_res, "args": vars(args)}
        torch.save(ckpt, out_dir / "last.pt")

        if val_res["mAP@0.5"] > best_map50:
            best_map50 = val_res["mAP@0.5"]
            torch.save(ckpt, out_dir / "best.pt")
            print(f"Saved best checkpoint at epoch {epoch}: mAP50={best_map50:.4f}", flush=True)

        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("Loading best checkpoint for test evaluation...")
    best = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])

    test_res = evaluate(model, test_loader, device, conf=0.25)
    (out_dir / "test_results.json").write_text(json.dumps(test_res, indent=2), encoding="utf-8")

    print("Final Test Results")
    print(json.dumps(test_res, indent=2))


if __name__ == "__main__":
    main()
