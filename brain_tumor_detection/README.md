# Brain Tumor Detection with YOLOv8

Object detection of brain tumors in MRI images with a fine-tuned **YOLOv8n**
model (Ultralytics). Bounding boxes were drawn in [Roboflow](https://roboflow.com),
and the images were split 80/20 into train and validation sets by
[`split_train_test.ipynb`](split_train_test.ipynb), and used to train a
single-class detector (`tumor`).

> **Research and educational use only.** This model is not a medical device and
> must not be used for diagnosis or any clinical decision.

## Results

Validation set: 28 images, 31 tumor instances. Final model (epoch 100,
`best.pt`):

| Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|
| 0.991 | 0.806 | 0.844 | 0.732 |

Training progress (validation metrics):

| Epoch | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| 10 | 0.672 | 0.594 | 0.626 | 0.417 |
| 30 | 0.864 | 0.710 | 0.785 | 0.607 |
| 50 | 0.925 | 0.799 | 0.851 | 0.667 |
| 70 | 0.959 | 0.763 | 0.813 | 0.676 |
| 100 | 0.991 | 0.806 | 0.844 | 0.732 |

Training took about 30 minutes on CPU (AMD Ryzen 5 7600).

**Qualitative test:** on 4 new images (`Y1` to `Y4`), the model detected one
tumor in each at the default confidence threshold of 0.25. These images have no
ground-truth labels in the notebook, so this is a sanity check, not a metric.

Read these numbers with care:

- The validation set is small (28 images, 31 boxes), so metrics will move a lot
  with a handful of images.
- The same validation set was used to monitor training and to report results,
  and there is no separate labeled test set, so the figures are likely
  optimistic.
- Precision is very high (0.99) while recall is lower (0.81): the model rarely
  raises false alarms, but it misses roughly one in five tumors. For medical
  screening, missed tumors are usually the costlier error.
- Images were split randomly at image level. If several slices come from the
  same patient, they can land in both train and validation and inflate results.

## Dataset

- **Classes:** 1 (`0: tumor`), defined in [`data.yaml`](data.yaml)
- **Annotations:** bounding boxes drawn in Roboflow and exported as YOLO-format `.txt` files (one box per line: `class x_center y_center width height`, normalised)
- **Split:** 136 labeled images, shuffled with `random.seed(42)`, 80% train (108) and 20% validation (28)
- **Cleaning:** Ultralytics removed one duplicate label from each of two training images (`Y102`, `Y11`)

<!-- TODO: add the source of the original images, their license and citation. -->

Expected layout after running the split notebook:

```
dataset/
├── images/                  # all source .jpg images
├── labels/                  # matching YOLO .txt labels
└── train_dataset/
    ├── train/{images,labels}/   # 108 images
    ├── val/{images,labels}/     # 28 images
    └── test/                    # new images for inference (no labels needed)
```

`data.yaml`:

```yaml
path: C:/Users/micha/Documents/llabelImg/My_First_Project/dataset/train_dataset
train: train/images
val: val/images
names:
  0: tumor
```

The `path` is an absolute Windows path. Change it to a relative one (for example
`path: dataset/train_dataset`) so the project runs on other machines.

## Method

1. **Split** (`split_train_test.ipynb`, cell 1): shuffle images with a fixed seed,
   copy 80% to `train/` and 20% to `val/`, along with each image's label file.
2. **Train:** fine-tune the pretrained `yolov8n.pt` (nano, 3.0M parameters,
   8.2 GFLOPs):

   ```python
   from ultralytics import YOLO

   model = YOLO("yolov8n.pt")
   model.train(data="data.yaml", epochs=100, imgsz=640, batch=4)
   ```

   Other settings were Ultralytics defaults, including automatic optimizer
   selection (AdamW, lr 0.002) and default augmentation (mosaic, horizontal
   flip, HSV jitter).
3. **Inspect** the folder structure (cell 2) to confirm the split.
4. **Predict** on new images with the best checkpoint:

   ```python
   model = YOLO("runs/detect/train/weights/best.pt")
   results = model.predict(
       source="dataset/train_dataset/test/",
       conf=0.25, save=True, save_txt=True,
   )
   ```

   Annotated images and label files are written to `runs/detect/predict/`.

## Getting started

Environment used: Python 3.10, Ultralytics 8.4.160, PyTorch (CPU build).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install ultralytics jupyter
jupyter notebook split_train_test.ipynb
```

Place your images in `dataset/images/` and YOLO labels in `dataset/labels/`,
then run the notebook top to bottom. Training uses the GPU automatically if one
is available.

Or train from the command line once the split exists:

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=4
```

## Project structure

```
.
├── split_train_test.ipynb       # split, train, inspect, predict
├── data.yaml                    # dataset config for Ultralytics
├── LICENSE                      # MIT
├── dataset/                     # images, labels and train/val/test folders
└── runs/detect/
    ├── train/weights/best.pt    # trained detector
    └── predict/                 # inference outputs
```

## Limitations and next steps

- Grow and diversify the dataset, and hold out a **labeled test set** that is
  never used during training or tuning.
- Split by **patient** to avoid leakage.
- Report per-image sensitivity and false-negative examples; tune the confidence
  threshold for the recall you need rather than using 0.25 by default.
- Try larger models (`yolov8s`/`yolov8m`) or more epochs, now that the nano
  model's training curves show mAP still improving late in training.
- Validate on scans from other scanners and MRI sequences.
- Add `path`-independent config and a `requirements.txt` with pinned versions.

## License

Released under the [MIT License](LICENSE). This covers the code and
documentation in this repository. The MRI images keep the license of their
original source, so check it before redistributing the dataset.

## Acknowledgements

- [Roboflow](https://roboflow.com) for bounding-box annotation
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8

Note that Ultralytics YOLO is itself licensed under AGPL-3.0, which may affect
how you can distribute or deploy models and code built on it.

## Author

Michael Ohaya
