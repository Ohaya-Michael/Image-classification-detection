# llabelImg: Brain Tumor Classification and Detection

Two complementary deep-learning approaches to brain tumor analysis on MRI images,
built in one workspace:

| Project | Question it answers | Approach | Folder |
|---|---|---|---|
| **Classification** | Does this scan show a tumor? | EfficientNetB0 transfer learning (TensorFlow/Keras) | [`brain_tumor_classification/`](brain_tumor_classification) |
| **Detection** | Where is the tumor? | YOLOv8n with Roboflow-drawn bounding boxes (Ultralytics) | [`brain_tumor_detection/`](brain_tumor_detection) |

> **Research and educational use only.** Not a medical device. Do not use for
> diagnosis or any clinical decision.

## Repository layout

Paths are relative to `~/Documents/llabelImg/`.

```
llabelImg/
├── README.md                          # this page
├── LICENSE                            # MIT
├── brain_tumor_classification/
│   ├── README.md
│   ├── train.ipynb                    # augment, train, evaluate
│   ├── S3_package_for_subArticle.py   # helper functions used by the notebook
│   ├── brain/{train,val,test}/        # image dataset (negative / positive)
│   ├── buildEfficientB0_Model/
│   │   └── trained_model_finetuned.keras
│   └── assets/sample_predictions.png
└── brain_tumor_detection/
    ├── README.md
    ├── data.yaml                      # dataset config for Ultralytics
    ├── split_train_test.ipynb         # split, train, inspect, predict
    ├── dataset/
    │   ├── images/  labels/           # source images and YOLO labels
    │   └── train_dataset/{train,val,test}/
    └── runs/detect/
        ├── train/weights/best.pt      # trained detector
        └── predict/                   # annotated predictions
```

## Results at a glance

### Classification (EfficientNetB0)

- **Data:** 948 train, 374 validation (after augmentation), 36 test images
- **Validation accuracy:** 73.3% after fine-tuning (from 67.7% with a frozen backbone)
- **Test accuracy:** 91.67% (33/36), test loss 0.209
- The 36-image test set is tiny: the 95% interval is roughly 78% to 97%.

### Detection (YOLOv8n)

- **Data:** 136 annotated images, 80/20 split (108 train, 28 validation)
- **Validation:** precision 0.991, recall 0.806, mAP@50 0.844, mAP@50-95 0.732
- The validation set (28 images, 31 boxes) also guided training, so the figures
  are likely optimistic. Recall is lower than precision, meaning some tumors are missed.

See each project's README for the full method, tables and caveats.

## Quick start

The two projects use different frameworks (TensorFlow vs. PyTorch), so give each
its own virtual environment.

**Classification**

```bash
cd ~/Documents/llabelImg/brain_tumor_classification
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install tensorflow numpy matplotlib tqdm jupyter
jupyter notebook train.ipynb
```

**Detection**

```bash
cd ~/Documents/llabelImg/brain_tumor_detection
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install ultralytics jupyter
jupyter notebook split_train_test.ipynb
```

Detection can also be run from the command line once the dataset is split:

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=4
yolo detect predict model=runs/detect/train/weights/best.pt \
     source=dataset/train_dataset/test conf=0.25 save=True save_txt=True
```

Before running detection on another machine, edit the `path:` line in
`brain_tumor_detection/data.yaml`; it currently points to an absolute Windows path.

## Documentation

| Document | Contents |
|---|---|
| [`brain_tumor_classification/README.md`](brain_tumor_classification/README.md) | Classification data, method, results, limitations |
| [`brain_tumor_detection/README.md`](brain_tumor_detection/README.md) | Detection data, method, results, limitations |
| [`LICENSE`](LICENSE) | MIT License |

## Limitations

- Small datasets and small evaluation sets in both projects
- Random image-level splits, so slices from one patient could appear in both train and evaluation sets
- Detection recall (0.81) trails precision (0.99); classification confidences are modest (about 64% to 73% on sample predictions)
- No external validation on other scanners or MRI sequences

## License

Code and documentation are released under the [MIT License](LICENSE). The MRI
images keep the license of their original source. Ultralytics YOLO is licensed
separately under AGPL-3.0, which may affect distribution and deployment.

## Acknowledgements

- [Roboflow](https://roboflow.com) for bounding-box annotation
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [TensorFlow / Keras](https://keras.io) for the EfficientNetB0 classifier

## Author

Michael Ohaya
