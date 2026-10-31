import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from load_data import LABELS , load_training_dataset
from sklearn.utils.class_weight import compute_class_weight


#TESTED
def plot_class_distribution(y: np.ndarray, title: str = "Emotion Class Distribution") -> None:
    """
    Plot a bar chart showing the number of samples per emotion class.

    Args:
        y     : np.ndarray of shape (N, num_classes) — one-hot encoded labels
        title : title of the plot
    """
    counts = y.sum(axis=0).astype(int)  # (num_classes,)

    colors = ["#e74c3c", "#8e44ad", "#2980b9", "#f1c40f",
              "#95a5a6", "#2ecc71", "#e67e22"]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(LABELS, counts, color=colors, edgecolor="black", linewidth=0.8)

    # Annotate each bar with its sample count
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    # Legend mapping each color to its emotion
    legend_handles = [
        Rectangle((0, 0), 1, 1, color=colors[i], edgecolor="black", linewidth=0.8)
        for i in range(len(LABELS))
    ]
    ax.legend(legend_handles, LABELS, title="Emotions", loc="upper right", fontsize=10)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Emotion", fontsize=13)
    ax.set_ylabel("Number of Samples", fontsize=13)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


#TESTED 
def get_class_weights(y: np.ndarray , max_weight: float ):
    """
    This class is going to return the class weights based on the training labels.

    Args: 
        y     : np.ndarray of shape (N, num_classes) — one-hot encoded labels
    """
    y_integers = np.argmax(y, axis=1) ## here it's going the return the index [0-7] which is the label
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    
    return { idx : np.min([weight,max_weight]) for idx,weight in enumerate(weights)}
