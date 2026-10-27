# Bengali License Plate Recognition

A two-stage deep learning pipeline for recognizing Bangladeshi (Bengali) vehicle license plates: Real-ESRGAN super-resolution to enhance low-quality plate crops, followed by a MobileNetV2 transfer-learning classifier that labels each crop as a digit, a district/city name, or a Bengali series letter.

## Repository contents

```
.
├── MobileNet_ALPR_FINAL.ipynb      # Classifier training & evaluation notebook
└── Real_ESRGAN.ipynb               # Super-resolution preprocessing notebook
```

## Dataset

The dataset (raw `Data.zip` and the Real-ESRGAN-enhanced `enhanced_dataset.zip`) is hosted on Google Drive rather than committed to this repo:

**[Download the dataset](https://drive.google.com/drive/folders/1DO0eGGp9gkg1TJ6XYFuIQiAFgbzsK7M4?usp=sharing)**

Both zips contain the same structure — `training_data/`, `val_data/`, `testing_data/` — split across 17 classes (see below). `enhanced_dataset.zip` is the enhanced output actually produced and consumed by these two notebooks.

## Pipeline

### 1. Image enhancement — `Real_ESRGAN.ipynb`
- Clones the official [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) repo and downloads the pretrained `RealESRGAN_x4plus` weights.
- Unzips `Data.zip` into `training_data/`, `val_data/`, `testing_data/` subfolders.
- Runs 4x super-resolution inference (`inference_realesrgan.py`) on every image, preserving folder structure.
- Plots side-by-side original-vs-enhanced comparisons for spot-checking.
- Packages the result as `enhanced_dataset.zip`.

### 2. Classification — `MobileNet_ALPR_FINAL.ipynb`
- Loads the enhanced train/val/test splits with Keras `ImageDataGenerator` at 224×224 (train: shear/zoom/horizontal-flip augmentation; val/test: rescale only).
- Model: **MobileNetV2** (ImageNet weights, frozen) → `GlobalAveragePooling2D` → `Dense(256, relu)` → `Dropout(0.3)` → `Dense(num_classes, softmax)`.
- Training: Adam (lr=1e-4), categorical cross-entropy, up to 20 epochs, with `ReduceLROnPlateau` and `EarlyStopping` callbacks.
- Evaluation: confusion matrix (seaborn heatmap), `classification_report`, and final test loss/accuracy.

## Classes (17)

| Type | Labels |
|---|---|
| Digits | `0`–`9` |
| District/city | `DHAKA`, `NARAYANGANJ` |
| Series letters/words | `JHA`, `LA`, `MATRO`, `ga`, `k` |

These correspond to the individual segments printed on a standard Bangladeshi license plate (city/district name, Bengali series letter, and digit groups), classified independently rather than as a single end-to-end plate OCR.

## Requirements

- Python 3, TensorFlow/Keras, OpenCV, scikit-learn, seaborn, matplotlib, NumPy
- For `Real_ESRGAN.ipynb`: `basicsr`, `facexlib`, `gfpgan`, `realesrgan` (installed via pip in the notebook)
- Both notebooks were originally written for **Google Colab** (Google Drive mounting, `/content/...` paths, `google.colab.files.download`)

## Usage

1. Download `Data.zip` and/or `enhanced_dataset.zip` from the [dataset link above](#dataset) and unzip so that `training_data/`, `val_data/`, and `testing_data/` sit under a single dataset directory.
2. *(Optional)* Re-run `Real_ESRGAN.ipynb` on `Data.zip` to regenerate the enhanced dataset from scratch.
3. In `MobileNet_ALPR_FINAL.ipynb`, update `train_Path`, `val_Path`, and `test_Path` to point at your local dataset location (they default to a Google Drive path).
4. Run `MobileNet_ALPR_FINAL.ipynb` end-to-end to train and evaluate the classifier.

## Notes

- Paths and Colab-specific calls (Drive mount, `/content/...`, `files.download`) need adjusting for local or other cloud environments.
- Class imbalance and per-class sample counts aren't addressed in the notebooks as-is — worth checking before drawing conclusions from the reported accuracy.
