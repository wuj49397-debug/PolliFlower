import argparse
import json
from pathlib import Path

import torch

from train_mask2former_stigma import (
    StigmaMask2FormerDataset,
    build_collate_fn,
    evaluate,
)
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/root/autodl-tmp/flower_baseline/data/yolo_stigma_seg")
    parser.add_argument("--model-name", default="/root/autodl-tmp/flower_baseline/checkpoints/mask2former-swin-tiny-coco-instance")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--ap-threshold", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    processor = AutoImageProcessor.from_pretrained(args.model_name)

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model_name,
        id2label={0: "stigma"},
        label2id={"stigma": 0},
        num_labels=1,
        ignore_mismatched_sizes=True,
    )

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    test_set = StigmaMask2FormerDataset(args.data, "test", imgsz=args.imgsz, hflip=0.0)
    collate_fn = build_collate_fn(processor)

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_res = evaluate(
        model,
        processor,
        test_loader,
        device,
        conf=args.conf,
        ap_threshold=args.ap_threshold,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(test_res, indent=2), encoding="utf-8")

    print("Final Test Results")
    print(json.dumps(test_res, indent=2))


if __name__ == "__main__":
    main()
