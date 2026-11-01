# Facial Emotion Recognition

A deep-learning project for recognizing facial expressions from grayscale face images and overlaying emotion predictions on video. The project is built around the FER2013 emotion classes and includes a custom Keras CNN, dataset loaders, evaluation plots, notebooks, and a face-tracking video demo.

## Recognized emotions

The classifier predicts seven classes, encoded alphabetically:

`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

## Project layout

The implementation is in [`DL_facial_emotion_recognition/`](./DL_facial_emotion_recognition/). The `_project` directory contains an identical copy of the implementation currently included in the workspace.

```text
DL_facial_emotion_recognition/
├── data/                         # Local FER2013 data; ignored by Git
│   ├── train/<emotion>/*.jpg
│   └── test/<emotion>/*.jpg
├── model/
│   ├── model.json                # Saved model architecture
│   └── model.weights.h5          # Saved model weights
├── data_preprocessing.py         # Class distributions and class weights
├── load_data.py                  # Dataset loading and label encoding
├── model_training.py             # CNN definition and training entry point
├── utils.py                      # Metrics, confusion matrices, model saving
├── video_emotion_tracker.py      # Face detection, tracking, and video output
├── emotion_recognition_colab.ipynb
├── fine_tuning_kaggle.ipynb
├── video_demo.mp4
└── requirements.txt
```

## Installation

Python 3.10+ is recommended. From the project directory:

```bash
cd DL_facial_emotion_recognition
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

TensorFlow may require a platform-specific installation, especially when using GPU acceleration. See the official TensorFlow installation guide if the standard installation does not work on your system.

## Dataset

The Python loaders expect a folder-structured FER2013 dataset. Each split must contain one directory per emotion, with 48×48 JPG images inside:

```text
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

By default, [`load_data.py`](./DL_facial_emotion_recognition/load_data.py) looks for `data/` beside the source files. To use a dataset stored elsewhere, set `DATA_DIR` to its root:

```powershell
$env:DATA_DIR = 'D:\datasets\fer2013'
```

The included notebooks show how to download the folder-structured dataset with `kagglehub` in Google Colab or Kaggle. Dataset licensing and access requirements are determined by the dataset provider.

## Train the CNN

From `DL_facial_emotion_recognition/`, run:

```bash
python model_training.py
```

The training script:

- loads and normalizes grayscale images to `[0, 1]`;
- applies random zoom, translation, rotation, and horizontal flipping;
- trains a four-block convolutional neural network;
- uses class weights, label smoothing, early stopping, and learning-rate reduction;
- saves the architecture and weights to `model/model.json` and `model/model.weights.h5`;
- displays confusion matrices and weighted F1 scores for the training and test splits.

Training settings such as batch size, number of epochs, learning rate, and augmentation strength are defined near the top of [`model_training.py`](./DL_facial_emotion_recognition/model_training.py).

## Run the video demo

The saved model can be used to detect faces in an MP4, classify each face, smooth predictions across frames, and assign simple IoU-based track IDs:

```bash
python video_emotion_tracker.py
```

By default, the script uses `video_demo.mp4` and the saved model in `model/`. You can provide a different video or save an annotated copy:

```bash
python video_emotion_tracker.py \
  --video path/to/input.mp4 \
  --output outputs/annotated.mp4
```

Useful options include `--model-dir`, `--max-missed`, `--iou-threshold`, and `--min-face-size`. Press `q` or `Esc` to stop playback.

## Notebooks

- [`emotion_recognition_colab.ipynb`](./DL_facial_emotion_recognition/emotion_recognition_colab.ipynb) provides an end-to-end FER2013 workflow for Google Colab, including dataset download, training, and evaluation.
- [`fine_tuning_kaggle.ipynb`](./DL_facial_emotion_recognition/fine_tuning_kaggle.ipynb) benchmarks pretrained VGG16, VGG19, and ResNet18-style transfer-learning models on Kaggle.

## Notes

- The checked-in model files make it possible to run the video demo without retraining.
- Training data is excluded by `.gitignore`; place it locally or configure `DATA_DIR`.
- The model is intended for experimentation and demonstration. Facial-expression predictions can be uncertain and should not be treated as reliable measures of a person’s internal emotional state.

