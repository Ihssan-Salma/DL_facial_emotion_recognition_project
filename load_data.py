import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# Emotion labels sorted alphabetically to ensure consistent integer encoding
LABELS = sorted(["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"])
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}


def _load_split(split_dir: str):
    """
    Load all images from a split directory (train or test).

    Directory structure expected:
        split_dir/
            {emotion_class}/
                *.jpg

    Returns:
        X : np.ndarray of shape (N, 48, 48, 1), dtype float32, values in [0, 1]
        y : np.ndarray of shape (N, num_classes), dtype float32 — one-hot encoded labels
    """
    images = []
    labels = []

    # we iterate over all of the label , we exist in the data directory
    for label in LABELS:
        class_dir = os.path.join(split_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for img_path in glob.glob(os.path.join(class_dir, "*.jpg")):
            img = Image.open(img_path).convert("L")  # we use L for gray , because they are monochrome images
            arr = np.array(img, dtype=np.float32) / 255   # normalize to [0, 1]
            images.append(arr)
            labels.append(LABEL_TO_IDX[label])

    X = np.array(images, dtype=np.float32)[..., np.newaxis]  # (N, 48, 48, 1)
    y_idx = np.array(labels, dtype=np.int64)                 # (N,)
    y = np.eye(len(LABELS), dtype=np.float32)[y_idx]         # (N, num_classes) one-hot
    return X, y


def load_training_dataset():
    """Load the training dataset from the path given by the DATA_DIR env variable.

    Returns:
        X : np.ndarray  shape (N, 48, 48, 1)  — pixel values in [0, 1]
        y : np.ndarray  shape (N, num_classes)  — one-hot encoded labels (see LABELS)
    """
    data_dir = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
    train_dir = os.path.join(data_dir, "train")
    return _load_split(train_dir)


def load_testing_dataset():
    """Load the testing dataset from the path given by the DATA_DIR env variable.

    Returns:
        X : np.ndarray  shape (N, 48, 48, 1)  — pixel values in [0, 1]
        y : np.ndarray  shape (N, num_classes)  — one-hot encoded labels (see LABELS)
    """
    data_dir = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
    test_dir = os.path.join(data_dir, "test")
    return _load_split(test_dir)

