# PolliFlower Baseline Reproducibility README

This README records the commands used to reproduce the standard-split and challenging-split baseline experiments. All commands assume the project root is:

```bash
/root/autodl-tmp/flower_baseline
```

The standard split is `data_standard`, and the challenging split is `data_hard`.

## 1. Environment

```bash
cd /root/autodl-tmp/flower_baseline

source /root/miniconda3/etc/profile.d/conda.sh
conda activate polliflower_yolo
export OMP_NUM_THREADS=1
```

Set the split to run:

```bash
# Standard split
DATASET=data_standard
OUTDIR=outputs_data1_clean

# Challenging split
# DATASET=data_hard
# OUTDIR=outputs_data_hard
```

Check the split before training:

```bash
python scripts/check_split_leakage_by_group_v2.py \
  --data /root/autodl-tmp/flower_baseline/${DATASET} \
  --out /root/autodl-tmp/flower_baseline/${DATASET}/split_leakage_report_v2.json

python scripts/count_split_stats.py \
  --data /root/autodl-tmp/flower_baseline/${DATASET}
```

## 2. Required Custom Scripts

The following scripts should exist before running all baselines:

```text
scripts/eval_yolov8_stigma_iou_tau080.py
scripts/train_mask_rcnn_stigma_tau080.py
scripts/train_mask2former_stigma_tau080.py
scripts/eval_yolov8_pose_point_metrics.py
scripts/train_keypoint_rcnn_pollination_save_epochs.py
scripts/select_keypoint_rcnn_map_best.py
scripts/eval_keypoint_rcnn_all_metrics.py
```

`eval_yolov8_stigma_iou_tau080.py`, `train_mask_rcnn_stigma_tau080.py`, and `train_mask2former_stigma_tau080.py` use `StrictIoU@0.8`. 

## 3. Flower Instance Detection

### 3.1 YOLOv8s-det

```bash
yolo detect train \
  model=checkpoints/yolov8s.pt \
  data=/root/autodl-tmp/flower_baseline/${DATASET}/yolo_det/polliflower_det.yaml \
  imgsz=640 \
  epochs=50 \
  batch=8 \
  workers=8 \
  device=0 \
  project=/root/autodl-tmp/flower_baseline/${OUTDIR}/yolov8_det \
  name=yolov8s_640 \
  exist_ok=True
```

```bash
yolo detect val \
  model=/root/autodl-tmp/flower_baseline/${OUTDIR}/yolov8_det/yolov8s_640/weights/best.pt \
  data=/root/autodl-tmp/flower_baseline/${DATASET}/yolo_det/polliflower_det.yaml \
  split=test \
  imgsz=640 \
  batch=8 \
  device=0 \
  project=/root/autodl-tmp/flower_baseline/${OUTDIR}/yolov8_det_eval \
  name=yolov8s_640_test \
  exist_ok=True
```

### 3.2 RT-DETR

```bash
yolo detect train \
  model=checkpoints/rtdetr-l.pt \
  data=/root/autodl-tmp/flower_baseline/${DATASET}/yolo_det/polliflower_det.yaml \
  imgsz=640 \
  epochs=40 \
  batch=6 \
  workers=6 \
  device=0 \
  amp=False \
  optimizer=AdamW \
  lr0=0.0001 \
  weight_decay=0.0001 \
  cos_lr=True \
  mosaic=0 \
  mixup=0 \
  copy_paste=0 \
  erasing=0 \
  project=/root/autodl-tmp/flower_baseline/${OUTDIR}/rtdetr_det \
  name=rtdetr_l_640_stable_b6 \
  exist_ok=True
```

```bash
yolo detect val \
  model=/root/autodl-tmp/flower_baseline/${OUTDIR}/rtdetr_det/rtdetr_l_640_stable_b6/weights/best.pt \
  data=/root/autodl-tmp/flower_baseline/${DATASET}/yolo_det/polliflower_det.yaml \
  split=test \
  imgsz=640 \
  batch=6 \
  device=0 \
  project=/root/autodl-tmp/flower_baseline/${OUTDIR}/rtdetr_det_eval \
  name=rtdetr_l_640_stable_b6_test \
  exist_ok=True
```

### 3.3 Faster R-CNN

```bash
python scripts/train_faster_rcnn_det.py \
  --data /root/autodl-tmp/flower_baseline/${DATASET}/yolo_det \
  --epochs 40 \
  --batch 6 \
  --workers 6 \
  --lr 0.0025 \
  --output /root/autodl-tmp/flower_baseline/${OUTDIR}/faster_rcnn_det/fasterrcnn_r50_fpn_640
```

```bash
python scripts/eval_faster_rcnn_pr.py \
  --data /root/autodl-tmp/flower_baseline/${DATASET}/yolo_det \
  --ckpt /root/autodl-tmp/flower_baseline/${OUTDIR}/faster_rcnn_det/fasterrcnn_r50_fpn_640/best.pt \
  --batch 4 \
  --workers 6 \
  --conf 0.25 \
  --iou 0.5
```

## 4. Stigma Instance Segmentation

The main table reports `IoU`, `StrictIoU@0.8`, `mAP@0.5`, and `mAP@0.5:0.95`.

### 4.1 YOLOv8s-seg

```bash
yolo segment train \
  model=checkpoints/yolov8s-seg.pt \
  data=/root/autodl-tmp/flower_baseline/${DATASET}/yolo_stigma_seg/polliflower_stigma_seg.yaml \
  imgsz=1024 \
  epochs=40 \
  batch=8 \
  workers=6 \
  device=0 \
  project=/root/autodl-tmp/flower_baseline/${OUTDIR}/yolov8_stigma_seg \
  name=yolov8s_seg_1024 \
  exist_ok=True
```

```bash
yolo segment val \
  model=/root/autodl-tmp/flower_baseline/${OUTDIR}/yolov8_stigma_seg/yolov8s_seg_1024/weights/best.pt \
  data=/root/autodl-tmp/flower_baseline/${DATASET}/yolo_stigma_seg/polliflower_stigma_seg.yaml \
  split=test \
  imgsz=1024 \
  batch=8 \
  device=0 \
  project=/root/autodl-tmp/flower_baseline/${OUTDIR}/yolov8_stigma_seg_eval \
  name=yolov8s_seg_1024_test \
  exist_ok=True
```

```bash
python scripts/eval_yolov8_stigma_iou_tau080.py \
  --model /root/autodl-tmp/flower_baseline/${OUTDIR}/yolov8_stigma_seg/yolov8s_seg_1024/weights/best.pt \
  --images /root/autodl-tmp/flower_baseline/${DATASET}/yolo_stigma_seg/images/test \
  --labels /root/autodl-tmp/flower_baseline/${DATASET}/yolo_stigma_seg/labels/test \
  --imgsz 1024 \
  --conf 0.25 \
  --device 0 \
  --out /root/autodl-tmp/flower_baseline/${OUTDIR}/yolov8_stigma_seg_eval/yolov8s_seg_1024_test/stigma_metrics_tau080.json
```

### 4.2 Mask R-CNN

```bash
python scripts/train_mask_rcnn_stigma_tau080.py \
  --data /root/autodl-tmp/flower_baseline/${DATASET}/yolo_stigma_seg \
  --pretrained /root/autodl-tmp/flower_baseline/checkpoints/maskrcnn_resnet50_fpn_coco.pth \
  --epochs 40 \
  --batch 4 \
  --workers 6 \
  --lr 0.0025 \
  --output /root/autodl-tmp/flower_baseline/${OUTDIR}/mask_rcnn_stigma_seg/maskrcnn_r50_fpn_1024
```

### 4.3 Mask2Former

Use the same Mask2Former setting for both splits. The configuration below follows the final efficient setting.

```bash
python scripts/train_mask2former_stigma_tau080.py \
  --data /root/autodl-tmp/flower_baseline/${DATASET}/yolo_stigma_seg \
  --model-name /root/autodl-tmp/flower_baseline/checkpoints/mask2former-swin-tiny-coco-instance \
  --imgsz 1024 \
  --epochs 30 \
  --batch 2 \
  --grad-accum 1 \
  --workers 4 \
  --lr 5e-6 \
  --weight-decay 0.05 \
  --output /root/autodl-tmp/flower_baseline/${OUTDIR}/mask2former_stigma_seg/mask2former_swin_tiny_1024_b2_e30
```

## 5. Pollination Point Localization

The main table reports `Pose mAP@0.5`, `StrictDist`, `StrictPCK@0.05`, and `StrictPCK@0.10`. 

### 5.1 YOLOv8s-pose

```bash
yolo pose train \
  model=checkpoints/yolov8s-pose.pt \
  data=/root/autodl-tmp/flower_baseline/${DATASET}/yolo_pollination_pose/polliflower_pollination_pose.yaml \
  imgsz=1024 \
  epochs=40 \
  batch=8 \
  workers=6 \
  device=0 \
  project=/root/autodl-tmp/flower_baseline/${OUTDIR}/yolov8_pollination_pose \
  name=yolov8s_pose_1024 \
  exist_ok=True
```

```bash
yolo pose val \
  model=/root/autodl-tmp/flower_baseline/${OUTDIR}/yolov8_pollination_pose/yolov8s_pose_1024/weights/best.pt \
  data=/root/autodl-tmp/flower_baseline/${DATASET}/yolo_pollination_pose/polliflower_pollination_pose.yaml \
  split=test \
  imgsz=1024 \
  batch=8 \
  device=0 \
  project=/root/autodl-tmp/flower_baseline/${OUTDIR}/yolov8_pollination_pose_eval \
  name=yolov8s_pose_1024_test \
  exist_ok=True
```


### 5.2 YOLO11s-pose

```bash
yolo pose train \
  model=checkpoints/yolo11s-pose.pt \
  data=/root/autodl-tmp/flower_baseline/${DATASET}/yolo_pollination_pose/polliflower_pollination_pose.yaml \
  imgsz=1024 \
  epochs=40 \
  batch=8 \
  workers=6 \
  device=0 \
  project=/root/autodl-tmp/flower_baseline/${OUTDIR}/yolo11_pollination_pose \
  name=yolo11s_pose_1024 \
  exist_ok=True
```

```bash
yolo pose val \
  model=/root/autodl-tmp/flower_baseline/${OUTDIR}/yolo11_pollination_pose/yolo11s_pose_1024/weights/best.pt \
  data=/root/autodl-tmp/flower_baseline/${DATASET}/yolo_pollination_pose/polliflower_pollination_pose.yaml \
  split=test \
  imgsz=1024 \
  batch=8 \
  device=0 \
  project=/root/autodl-tmp/flower_baseline/${OUTDIR}/yolo11_pollination_pose_eval \
  name=yolo11s_pose_1024_test \
  exist_ok=True
```

```bash
python scripts/eval_yolov8_pose_point_metrics.py \
  --model /root/autodl-tmp/flower_baseline/${OUTDIR}/yolo11_pollination_pose/yolo11s_pose_1024/weights/best.pt \
  --images /root/autodl-tmp/flower_baseline/${DATASET}/test/images \
  --annotations /root/autodl-tmp/flower_baseline/${DATASET}/test/annotations_with_ids \
  --imgsz 1024 \
  --conf 0.25 \
  --box-iou 0.5 \
  --device 0 \
  --out /root/autodl-tmp/flower_baseline/${OUTDIR}/yolo11_pollination_pose_eval/yolo11s_pose_1024_test/point_metrics_rawtest.json
```

### 5.3 Keypoint R-CNN

For fair checkpoint selection, Keypoint R-CNN uses validation `Pose mAP@0.5` to select `best_map.pt`. The task-specific strict metrics are reported only on the final test set.

```bash
KPT_OUT=/root/autodl-tmp/flower_baseline/${OUTDIR}/keypoint_rcnn_pollination/keypointrcnn_r50_fpn_1024_mapbest

python scripts/train_keypoint_rcnn_pollination_save_epochs.py \
  --data /root/autodl-tmp/flower_baseline/${DATASET} \
  --pretrained /root/autodl-tmp/flower_baseline/checkpoints/keypointrcnn_resnet50_fpn_coco.pth \
  --imgsz 1024 \
  --epochs 40 \
  --batch 4 \
  --workers 6 \
  --lr 0.0025 \
  --output ${KPT_OUT}
```

Create a validation-as-test view for mAP-based checkpoint selection:

```bash
mkdir -p eval_views/${DATASET}_val_as_test/test
rm -f eval_views/${DATASET}_val_as_test/test/images
rm -f eval_views/${DATASET}_val_as_test/test/annotations_with_ids

ln -sfn /root/autodl-tmp/flower_baseline/${DATASET}/val/images \
  eval_views/${DATASET}_val_as_test/test/images

ln -sfn /root/autodl-tmp/flower_baseline/${DATASET}/val/annotations_with_ids \
  eval_views/${DATASET}_val_as_test/test/annotations_with_ids
```

Select `best_map.pt` using validation `Pose mAP@0.5`:

```bash
python scripts/select_keypoint_rcnn_map_best.py \
  --eval-data /root/autodl-tmp/flower_baseline/eval_views/${DATASET}_val_as_test \
  --ckpt-dir ${KPT_OUT} \
  --imgsz 1024 \
  --batch 4 \
  --workers 6 \
  --conf 0.25 \
  --box-iou 0.5
```

Evaluate the selected checkpoint on the test set:

```bash
python scripts/eval_keypoint_rcnn_all_metrics.py \
  --data /root/autodl-tmp/flower_baseline/${DATASET} \
  --ckpt ${KPT_OUT}/best_map.pt \
  --imgsz 1024 \
  --batch 4 \
  --workers 6 \
  --conf 0.25 \
  --box-iou 0.5 \
  --out ${KPT_OUT}/test_results_mapbest.json
```

## 6. Notes

- Do not mix standard-split and challenging-split outputs.
- Use `outputs_data1_clean` only for `data_standard`.
- Use `outputs_data_hard` only for `data_hard`.
- Avoid spaces after the line-continuation character `\`.
- YOLO pose custom metrics should use raw split images, e.g. `${DATASET}/test/images`, rather than `${DATASET}/yolo_pollination_pose/images/test`.
- `StrictIoU` for stigma segmentation is reported as `StrictIoU@0.8`.

