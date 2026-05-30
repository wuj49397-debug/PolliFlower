import argparse
import json
import shutil
import subprocess
from pathlib import Path

def find_metric(obj):
    if isinstance(obj, dict):
        for key in ["Pose mAP50-95", "pose_mAP50_95", "pose_map50_95", "Pose mAP@0.5:0.95"]:
            if key in obj:
                return float(obj[key])
        for v in obj.values():
            r = find_metric(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_metric(v)
            if r is not None:
                return r
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--box-iou", type=float, default=0.5)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    eval_dir = ckpt_dir / "val_map_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    best_ckpt = None
    records = []

    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpts:
        raise SystemExit(f"No epoch checkpoints found in {ckpt_dir}")

    for ckpt in ckpts:
        out_json = eval_dir / f"{ckpt.stem}.json"
        cmd = [
            "python", "scripts/eval_keypoint_rcnn_all_metrics.py",
            "--data", args.eval_data,
            "--ckpt", str(ckpt),
            "--imgsz", str(args.imgsz),
            "--batch", str(args.batch),
            "--workers", str(args.workers),
            "--conf", str(args.conf),
            "--box-iou", str(args.box_iou),
            "--out", str(out_json),
        ]
        print("Evaluating", ckpt.name, flush=True)
        subprocess.run(cmd, check=True)

        data = json.loads(out_json.read_text(encoding="utf-8"))
        score = find_metric(data)
        if score is None:
            raise SystemExit(f"Cannot find Pose mAP50-95 in {out_json}")

        rec = {"ckpt": str(ckpt), "Pose mAP50-95": score}
        records.append(rec)
        print(rec, flush=True)

        if score > best_score:
            best_score = score
            best_ckpt = ckpt

    shutil.copy2(best_ckpt, ckpt_dir / "best_map.pt")

    summary = {
        "selection_metric": "validation Pose mAP50-95",
        "best_ckpt": str(best_ckpt),
        "best_score": best_score,
        "records": records,
    }
    (ckpt_dir / "map_best_selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSelected best_map.pt")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
