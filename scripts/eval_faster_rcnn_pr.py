import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


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

        image = F.to_tensor(image)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        return image, {"boxes": boxes, "image_id": torch.tensor([idx])}


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes=2, min_size=640, max_size=640):
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=min_size,
        max_size=max_size,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


@torch.no_grad()
def evaluate_pr(model, loader, device, conf=0.25, iou_thr=0.5):
    model.eval()

    tp = 0
    fp = 0
    fn = 0
    total_images = 0
    total_instances = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            gt_boxes = tgt["boxes"].cpu()
            total_instances += len(gt_boxes)

            scores = out["scores"].detach().cpu()
            pred_boxes = out["boxes"].detach().cpu()

            keep = scores >= conf
            pred_boxes = pred_boxes[keep]
            scores = scores[keep]

            order = torch.argsort(scores, descending=True)
            pred_boxes = pred_boxes[order]

            matched_gt = set()

            for pb in pred_boxes:
                if len(gt_boxes) == 0:
                    fp += 1
                    continue

                ious = box_iou(pb.unsqueeze(0), gt_boxes).squeeze(0)
                best_iou, best_idx = torch.max(ious, dim=0)

                if float(best_iou) >= iou_thr and int(best_idx) not in matched_gt:
                    tp += 1
                    matched_gt.add(int(best_idx))
                else:
                    fp += 1

            fn += len(gt_boxes) - len(matched_gt)
            total_images += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {
        "images": total_images,
        "instances": total_instances,
        "conf": conf,
        "iou_thr": iou_thr,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/root/autodl-tmp/flower_baseline/data/yolo_det")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2, min_size=640, max_size=640)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    dataset = FlowerDetectionDataset(args.data, "test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    res = evaluate_pr(model, loader, device, conf=args.conf, iou_thr=args.iou)
    print(res)
    print(
        f"Images: {res['images']} | Instances: {res['instances']} | "
        f"P: {res['precision']:.4f} | R: {res['recall']:.4f} | "
        f"TP: {res['TP']} | FP: {res['FP']} | FN: {res['FN']}"
    )


if __name__ == "__main__":
    main()
