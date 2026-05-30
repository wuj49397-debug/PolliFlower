import argparse
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class FlowerDetectionDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.images = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in exts])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []

        if label_path.exists():
            for line in label_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                cls, xc, yc, bw, bh = map(float, line.split())
                x1 = (xc - bw / 2.0) * w
                y1 = (yc - bh / 2.0) * h
                x2 = (xc + bw / 2.0) * w
                y2 = (yc + bh / 2.0) * h

                x1 = max(0.0, min(x1, w - 1))
                y1 = max(0.0, min(y1, h - 1))
                x2 = max(0.0, min(x2, w))
                y2 = max(0.0, min(y2, h))

                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(1)

        image = F.to_tensor(image)

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes=2, min_size=640, max_size=640):
    model = fasterrcnn_resnet50_fpn(
        weights="DEFAULT",
        min_size=min_size,
        max_size=max_size,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=False,
    )

    total_images = 0
    total_instances = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets_cpu = []
        for t in targets:
            targets_cpu.append({
                "boxes": t["boxes"].cpu(),
                "labels": t["labels"].cpu(),
            })
            total_instances += len(t["labels"])

        outputs = model(images)
        preds_cpu = []
        for out in outputs:
            preds_cpu.append({
                "boxes": out["boxes"].detach().cpu(),
                "scores": out["scores"].detach().cpu(),
                "labels": out["labels"].detach().cpu(),
            })

        metric.update(preds_cpu, targets_cpu)
        total_images += len(images)

    res = metric.compute()
    map50 = float(res["map_50"])
    map5095 = float(res["map"])
    precision = float("nan")
    recall = float("nan")

    return {
        "images": total_images,
        "instances": total_instances,
        "mAP50": map50,
        "mAP50-95": map5095,
        "precision": precision,
        "recall": recall,
    }


def train_one_epoch(model, optimizer, loader, device, epoch, print_freq=50):
    model.train()
    running_loss = 0.0
    start = time.time()

    for step, (images, targets) in enumerate(loader, 1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running_loss += float(loss.item())

        if step % print_freq == 0 or step == len(loader):
            avg_loss = running_loss / step
            elapsed = time.time() - start
            print(f"Epoch {epoch} | step {step}/{len(loader)} | loss {avg_loss:.4f} | time {elapsed:.1f}s", flush=True)

    return running_loss / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/root/autodl-tmp/flower_baseline/data/yolo_det")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--output", default="/root/autodl-tmp/flower_baseline/outputs/faster_rcnn_det/fasterrcnn_r50_fpn_640")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set = FlowerDetectionDataset(args.data, "train")
    val_set = FlowerDetectionDataset(args.data, "val")
    test_set = FlowerDetectionDataset(args.data, "test")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Train images: {len(train_set)}")
    print(f"Val images: {len(val_set)}")
    print(f"Test images: {len(test_set)}")

    model = build_model(num_classes=2, min_size=640, max_size=640)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.67), int(args.epochs * 0.89)],
        gamma=0.1,
    )

    best_map50 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()

        val_res = evaluate(model, val_loader, device)
        val_res["epoch"] = epoch
        val_res["train_loss"] = loss
        history.append(val_res)

        print(
            f"Epoch {epoch} Val | images {val_res['images']} | instances {val_res['instances']} | "
            f"mAP50 {val_res['mAP50']:.4f} | mAP50-95 {val_res['mAP50-95']:.4f}",
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

        torch.save(ckpt, output_dir / "last.pt")

        if val_res["mAP50"] > best_map50:
            best_map50 = val_res["mAP50"]
            torch.save(ckpt, output_dir / "best.pt")
            print(f"Saved best checkpoint at epoch {epoch}: mAP50={best_map50:.4f}", flush=True)

        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("Loading best checkpoint for test evaluation...")
    best = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])

    test_res = evaluate(model, test_loader, device)
    with open(output_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_res, f, indent=2)

    print("Final Test Results")
    print(
        f"Images: {test_res['images']} | Instances: {test_res['instances']} | "
        f"mAP50: {test_res['mAP50']:.4f} | mAP50-95: {test_res['mAP50-95']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
