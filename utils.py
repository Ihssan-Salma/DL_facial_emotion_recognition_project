import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

# Make sure your LABELS array matches exactly how your data was loaded
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def generate_confusion_matrix(model, X_val, Y_val, title="Validation Confusion Matrix"):
    print(f"\n--- Generating: {title} ---")
    
    # 1. Forward pass: Get the raw probabilities from the model
    predictions = model.predict(X_val)
    
    # 2. Collapse One-Hot arrays back to single integers (e.g., [0, 1, 0, 0, 0, 0, 0] -> 1)
    y_pred_classes = np.argmax(predictions, axis=1)
    y_true_classes = np.argmax(Y_val, axis=1)
    
    # 3. Calculate metrics
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    accuracy = np.mean(y_true_classes == y_pred_classes) * 100
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted') * 100
    
    # 4. Plot it visually
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f"{title}\nAccuracy: {accuracy:.2f}%  |  F1 Score: {f1:.2f}%")
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.show()

def save_model(model) : 
    """
    Save a Keras model's architecture and weights to disk.

    The model's architecture is serialized to JSON and written to
    `./model/model.json`, and the model's weights are saved in H5 format
    to `./model/model.weights.h5`.

    Parameters
    ----------
    model :
        A Keras model instance that provides `to_json()` and `save_weights()`
        methods.
    """
    model_json = model.to_json()
    with open("./model/model.json", "w", encoding="utf-8") as json_file:
        json_file.write(model_json)
    model.save_weights("./model/model.weights.h5")
    print("Saved model architecture to model.json and weights to model.weights.h5")