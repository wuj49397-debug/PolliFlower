# PolliFlower

PolliFlower is a multi-crop hierarchical dataset and benchmark for precision pollination perception. It is designed for operation-oriented agricultural vision, where a model must not only localize flowers but also identify visible stigma regions and actionable pollination points.

## Benchmark Tasks

PolliFlower defines three benchmark tasks:

1. **Flower Instance Detection**  
   Detect flower instances in each image using bounding boxes.

2. **Stigma Instance Segmentation**  
   Segment the visible stigma region for each flower instance.

3. **Pollination Point Localization**  
   Localize the target point for robotic pollination on each valid flower instance.

## Repository Contents

This repository currently provides:

- sample images and annotations;
- baseline training and evaluation scripts;
- sample data for the Standard and Challenging protocols;
- checkpoint preparation instructions;
- data access and usage terms.

The full PolliFlower dataset is not publicly released before publication. Researchers who need access to the complete dataset should contact the maintainers and sign the PolliFlower Data Use Agreement.

## Sample Data Structure

Sample data are provided under `samples/`:

```text
samples/
  data_standard/
    train/
      images/
      annotations_with_ids/
    val/
      images/
      annotations_with_ids/
    test/
      images/
      annotations_with_ids/
  data_hard/
    train/
      images/
      annotations_with_ids/
    val/
      images/
      annotations_with_ids/
    test/
      images/
      annotations_with_ids/
```

`data_standard` corresponds to the Standard Protocol, while `data_hard` corresponds to the Challenging Protocol.

## Annotation Format

Each image is paired with a JSON annotation file in `annotations_with_ids`. The annotations include:

- flower bounding boxes;
- visible stigma region polygons;
- pollination point annotations;
- instance IDs linking the flower, stigma region, and pollination point.

The instance ID enables hierarchical instance-level evaluation across the three tasks.

## Baseline Models

The benchmark includes representative baselines for each task:

- **Flower instance detection:** YOLOv8, RT-DETR, Faster R-CNN
- **Stigma instance segmentation:** YOLOv8-seg, Mask R-CNN, Mask2Former
- **Pollination point localization:** YOLOv8-pose, YOLO11-pose, Keypoint R-CNN

## Checkpoints

Model weights are not included in this public repository.

Please manually download or prepare the required pretrained weights before running the baseline scripts. The `checkpoints/` directory can be used to store pretrained weights locally.

## Data Access

The complete PolliFlower dataset is available upon request for academic research only. To request access, please provide:

1. Name
2. Affiliation
3. Position
4. Institutional email
5. Research purpose
6. Agreement not to redistribute the dataset
7. Agreement to cite the PolliFlower paper after publication

Please see [DATA_USE_AGREEMENT.md](DATA_USE_AGREEMENT.md) for the data usage terms.

## Citation

The citation will be added after publication.

## Contact

Please contact the maintainers for full dataset access.

Contact email: wuj49397@gmail.com
