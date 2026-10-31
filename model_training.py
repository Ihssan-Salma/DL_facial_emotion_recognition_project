import random
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, BatchNormalization,
                          Dense, Flatten, Input, Dropout)
from keras.layers import RandomZoom, RandomTranslation, RandomRotation, RandomFlip
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.losses import CategoricalCrossentropy
from load_data import load_training_dataset, load_testing_dataset
from data_preprocessing import get_class_weights
from utils import generate_confusion_matrix , save_model

# ──────────────────────────────────────────────
#  Input
# ──────────────────────────────────────────────
INPUT_SHAPE = (48, 48, 1)
NUM_CLASSES = 7

# ──────────────────────────────────────────────
#  Data Augmentation
# ──────────────────────────────────────────────
AUGMENT_ZOOM = 0.05
AUGMENT_TRANSLATION = 0.05
AUGMENT_ROTATION = 0.05

# ──────────────────────────────────────────────
#  Convolutional Blocks
# ──────────────────────────────────────────────
CONV_KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
POOL_STRIDES = (2, 2)
LEAKY_RELU_ALPHA = 0.01

BLOCK1_FILTERS = 64
BLOCK2_FILTERS = 96
BLOCK3_FILTERS = 128
BLOCK4_FILTERS = 256

# ──────────────────────────────────────────────
#  Dense Head
# ──────────────────────────────────────────────
DENSE_UNITS = 512
DENSE_L2_LAMBDA = 0.001
DROPOUT_RATE = 0.5

# ──────────────────────────────────────────────
#  Reproducibility
# ──────────────────────────────────────────────
SEED = 42

def set_seeds(seed: int = SEED) -> None:
    """Fix all relevant RNG seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ──────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────
LEARNING_RATE = 0.0003
BATCH_SIZE = 64
EPOCHS = 75
LABEL_SMOOTHING = 0.1
MAX_CLASS_WEIGHT = 4

# ──────────────────────────────────────────────
#  Callbacks
# ──────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 15
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 4
LR_REDUCE_MIN = 1e-6


## Model Creation

def create_model(dropout_rate: float = DROPOUT_RATE,
                 l2_lambda: float = DENSE_L2_LAMBDA,
                 learning_rate: float = LEARNING_RATE) -> Sequential:
    """Build, compile, and return the CNN model."""
    set_seeds()
    model = Sequential([
        Input(shape=INPUT_SHAPE),

        ## Data Augmentation
        RandomZoom(height_factor=(-AUGMENT_ZOOM, AUGMENT_ZOOM)),
        RandomTranslation(height_factor=(-AUGMENT_TRANSLATION, AUGMENT_TRANSLATION),
                          width_factor=(-AUGMENT_TRANSLATION, AUGMENT_TRANSLATION)),
        RandomRotation(factor=(-AUGMENT_ROTATION, AUGMENT_ROTATION)),
        RandomFlip("horizontal"),

        ## Block 1
        Conv2D(BLOCK1_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=LEAKY_RELU_ALPHA),
        Conv2D(BLOCK1_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=LEAKY_RELU_ALPHA),
        MaxPooling2D(POOL_SIZE, strides=POOL_STRIDES),

        ## Block 2
        Conv2D(BLOCK2_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=LEAKY_RELU_ALPHA),
        Conv2D(BLOCK2_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=LEAKY_RELU_ALPHA),
        Conv2D(BLOCK2_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=LEAKY_RELU_ALPHA),
        MaxPooling2D(POOL_SIZE, strides=POOL_STRIDES),

        ## Block 3
        Conv2D(BLOCK3_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=LEAKY_RELU_ALPHA),
        Conv2D(BLOCK3_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=LEAKY_RELU_ALPHA),
        Conv2D(BLOCK3_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=LEAKY_RELU_ALPHA),
        MaxPooling2D(POOL_SIZE, strides=POOL_STRIDES),

        ## Block 4
        Conv2D(BLOCK4_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=LEAKY_RELU_ALPHA),
        Conv2D(BLOCK4_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=LEAKY_RELU_ALPHA),
        Conv2D(BLOCK4_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=LEAKY_RELU_ALPHA),
        MaxPooling2D(POOL_SIZE, strides=POOL_STRIDES),

        ## Dense Head
        Flatten(),

        Dense(DENSE_UNITS, kernel_regularizer=l2(l2_lambda)),
        BatchNormalization(),
        LeakyReLU(alpha=LEAKY_RELU_ALPHA),
        Dropout(dropout_rate),

        Dense(DENSE_UNITS, kernel_regularizer=l2(l2_lambda)),
        BatchNormalization(),
        LeakyReLU(alpha=LEAKY_RELU_ALPHA),
        Dropout(dropout_rate),

        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"]
    )
    return model

def train_model():
    set_seeds()

    print("Loading Training Data...")
    XTrain, YTrain = load_training_dataset()
    print("Loading Testing Data...")
    XTest, YTest = load_testing_dataset()

    class_weights = get_class_weights(YTrain, max_weight=MAX_CLASS_WEIGHT)
    model = create_model()

    early_stopping_callback = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE,
        min_lr=LR_REDUCE_MIN,
        verbose=1
    )

    print("Commencing Training...")
    history = model.fit(
        x=XTrain, y=YTrain,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        class_weight=class_weights,
        validation_data=(XTest, YTest),
        shuffle=True,
        callbacks=[early_stopping_callback, lr_scheduler]
    )
    print("Training complete.")

    save_model(model)

    generate_confusion_matrix(model, XTrain, YTrain, "Validation Matrix on Training")
    generate_confusion_matrix(model, XTest, YTest, "Validation Matrix on Testing")

    return model, history

train_model()