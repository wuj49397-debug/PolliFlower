import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


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


def resize_mask(mask, size):
    return cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)


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


class StigmaMask2FormerDataset(Dataset):
    def __init__(self, root, split, imgsz=1024, hflip=0.0):
        self.root = Path(root)
        self.split = split
        self.imgsz = imgsz
        self.hflip = hflip

        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split

        self.images = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

    def __len__(self):
        return len(self.images)

    def load_masks(self, img_path, h, w):
        label_path = self.label_dir / f"{img_path.stem}.txt"
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

            pts = []
            for i in range(0, len(coords), 2):
                pts.append([coords[i] * w, coords[i + 1] * h])

            mask = polygon_to_mask(pts, h, w)
            if mask.sum() > 0:
                masks.append(mask)

        return masks

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        masks_orig = self.load_masks(img_path, h, w)

        if self.split == "train" and self.hflip > 0 and random.random() < self.hflip:
            image = ImageOps.mirror(image)
            masks_orig = [np.fliplr(m).copy() for m in masks_orig]

        image_resized = image.resize((self.imgsz, self.imgsz), resample=Image.BILINEAR)

        if len(masks_orig) > 0:
            train_masks = [resize_mask(m, self.imgsz) for m in masks_orig]
            train_masks = torch.as_tensor(np.stack(train_masks), dtype=torch.float32)
            class_labels = torch.zeros((len(train_masks),), dtype=torch.long)
            gt_masks_orig = torch.as_tensor(np.stack(masks_orig).astype(bool), dtype=torch.bool)
            gt_labels = torch.zeros((len(masks_orig),), dtype=torch.long)
        else:
            train_masks = torch.zeros((0, self.imgsz, self.imgsz), dtype=torch.float32)
            class_labels = torch.zeros((0,), dtype=torch.long)
            gt_masks_orig = torch.zeros((0, h, w), dtype=torch.bool)
            gt_labels = torch.zeros((0,), dtype=torch.long)

        return {
            "image": image_resized,
            "mask_labels": train_masks,
            "class_labels": class_labels,
            "gt_masks_orig": gt_masks_orig,
            "gt_labels": gt_labels,
            "orig_size": (h, w),
            "image_name": img_path.name,
        }


def build_collate_fn(processor):
    def collate_fn(batch):
        images = [x["image"] for x in batch]

        inputs = processor(
            images=images,
            return_tensors="pt",
            do_resize=False,
        )

        inputs["mask_labels"] = [x["mask_labels"] for x in batch]
        inputs["class_labels"] = [x["class_labels"] for x in batch]

        meta = {
            "gt_masks_orig": [x["gt_masks_orig"] for x in batch],
            "gt_labels": [x["gt_labels"] for x in batch],
            "orig_size": [x["orig_size"] for x in batch],
            "image_name": [x["image_name"] for x in batch],
        }

        return inputs, meta

    return collate_fn


def move_batch_to_device(inputs, device):
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, list):
            out[k] = [x.to(device) if isinstance(x, torch.Tensor) else x for x in v]
        else:
            out[k] = v
    return out


def extract_pred_masks(processed):
    if processed.get("segmentation", None) is None:
        return (
            torch.zeros((0, 1, 1), dtype=torch.bool),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
        )

    seg = processed["segmentation"]
    info = processed.get("segments_info", [])

    if isinstance(seg, torch.Tensor) and seg.ndim == 3:
        masks = seg.bool().cpu()
        scores = torch.as_tensor([float(x["score"]) for x in info], dtype=torch.float32)
        labels = torch.as_tensor([int(x["label_id"]) for x in info], dtype=torch.long)
        return masks, scores, labels

    if isinstance(seg, torch.Tensor) and seg.ndim == 2:
        masks = []
        scores = []
        labels = []
        for item in info:
            sid = int(item["id"])
            masks.append((seg == sid).cpu())
            scores.append(float(item["score"]))
            labels.append(int(item["label_id"]))
        if len(masks) == 0:
            h, w = seg.shape
            return (
                torch.zeros((0, h, w), dtype=torch.bool),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
            )
        return torch.stack(masks).bool(), torch.as_tensor(scores, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.long)

    return (
        torch.zeros((0, 1, 1), dtype=torch.bool),
        torch.zeros((0,), dtype=torch.float32),
        torch.zeros((0,), dtype=torch.long),
    )


@torch.no_grad()
def evaluate(model, processor, loader, device, conf=0.25, ap_threshold=0.05):
    model.eval()
    metric = MeanAveragePrecision(iou_type="segm", class_metrics=False)

    all_ious, all_strict, all_cov = [], [], []
    total_images, total_gt, total_pred_conf = 0, 0, 0

    for inputs, meta in loader:
        batch = move_batch_to_device(inputs, device)

        outputs = model(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch.get("pixel_mask", None),
        )

        processed = processor.post_process_instance_segmentation(
            outputs,
            threshold=ap_threshold,
            mask_threshold=0.5,
            overlap_mask_area_threshold=0.8,
            target_sizes=meta["orig_size"],
            return_binary_maps=True,
        )

        preds_for_map = []
        targets_for_map = []

        for pred_item, gt_masks, gt_labels in zip(processed, meta["gt_masks_orig"], meta["gt_labels"]):
            pred_masks, scores, labels = extract_pred_masks(pred_item)

            preds_for_map.append({
                "masks": pred_masks.cpu().bool(),
                "scores": scores.cpu(),
                "labels": labels.cpu(),
            })

            targets_for_map.append({
                "masks": gt_masks.cpu().bool(),
                "labels": gt_labels.cpu(),
            })

            keep = scores >= conf
            pred_keep = pred_masks[keep].cpu().numpy().astype(np.uint8)
            gt_np = gt_masks.cpu().numpy().astype(np.uint8)

            ious, strict, covs = greedy_iou(gt_np, pred_keep)
            all_ious.extend(ious)
            all_strict.extend(strict)
            all_cov.extend(covs)

            total_images += 1
            total_gt += len(gt_np)
            total_pred_conf += int(keep.sum().item())

        metric.update(preds_for_map, targets_for_map)

    res = metric.compute()

    return {
        "images": total_images,
        "gt_instances": total_gt,
        "pred_instances_conf025": total_pred_conf,
        "IoU": float(np.mean(all_ious)) if all_ious else 0.0,
        "StrictIoU": float(np.mean(all_strict)) if all_strict else 0.0,
        "MeanCoverage": float(np.mean(all_cov)) if all_cov else 0.0,
        "mAP@0.5": float(res["map_50"]),
        "mAP@0.5:0.95": float(res["map"]),
    }


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, grad_accum=1, use_amp=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    running = 0.0
    start = time.time()

    for step, (inputs, meta) in enumerate(loader, 1):
        batch = move_batch_to_device(inputs, device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % grad_accum == 0 or step == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        running += float(loss.item()) * grad_accum

        if step % 50 == 0 or step == len(loader):
            print(f"Epoch {epoch} | {step}/{len(loader)} | loss {running / step:.4f} | {time.time() - start:.1f}s", flush=True)

    return running / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/root/autodl-tmp/flower_baseline/data/yolo_stigma_seg")
    parser.add_argument("--model-name", default="/root/autodl-tmp/flower_baseline/checkpoints/mask2former-swin-tiny-coco-instance")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--hflip", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--ap-threshold", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", default="/root/autodl-tmp/flower_baseline/outputs/mask2former_stigma_seg/mask2former_swin_tiny_1024")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model:", args.model_name)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(args.model_name)

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model_name,
        id2label={0: "stigma"},
        label2id={"stigma": 0},
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_set = StigmaMask2FormerDataset(args.data, "train", imgsz=args.imgsz, hflip=args.hflip)
    val_set = StigmaMask2FormerDataset(args.data, "val", imgsz=args.imgsz, hflip=0.0)
    test_set = StigmaMask2FormerDataset(args.data, "test", imgsz=args.imgsz, hflip=0.0)

    collate_fn = build_collate_fn(processor)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    print(f"Train images: {len(train_set)} | Val images: {len(val_set)} | Test images: {len(test_set)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_map50 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            grad_accum=args.grad_accum,
            use_amp=args.amp,
        )
        scheduler.step()

        val_res = evaluate(model, processor, val_loader, device, conf=args.conf, ap_threshold=args.ap_threshold)
        val_res["epoch"] = epoch
        val_res["train_loss"] = loss
        history.append(val_res)

        print(
            f"Epoch {epoch} Val | IoU {val_res['IoU']:.4f} | StrictIoU {val_res['StrictIoU']:.4f} | "
            f"mAP50 {val_res['mAP@0.5']:.4f} | mAP50-95 {val_res['mAP@0.5:0.95']:.4f}",
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

        if val_res["mAP@0.5"] > best_map50:
            best_map50 = val_res["mAP@0.5"]
            torch.save(ckpt, out_dir / "best.pt")
            print(f"Saved best checkpoint at epoch {epoch}: mAP50={best_map50:.4f}", flush=True)

        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("Loading best checkpoint for test evaluation...")
    best = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])

    test_res = evaluate(model, processor, test_loader, device, conf=args.conf, ap_threshold=args.ap_threshold)
    (out_dir / "test_results.json").write_text(json.dumps(test_res, indent=2), encoding="utf-8")

    print("Final Test Results")
    print(json.dumps(test_res, indent=2))


if __name__ == "__main__":
    main()
