import matplotlib
matplotlib.use("Agg")

import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import base64
from io import BytesIO
import matplotlib.pyplot as plt

##############################################################################
# Detect device
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    DEVICE = "/GPU:0"
    print("GPU detected. Using GPU.")
else:
    DEVICE = "/CPU:0"
    print("No GPU detected. Using CPU.")
print("TensorFlow version:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())
##############################################################################

def build_model(lr_value: float):
    with tf.device(DEVICE):
        model = Sequential([
            Input(shape=(28, 28)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=lr_value),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        print(f"Model Metrics: {model.metrics}")

    return model

def build_and_train(x_poisoned, y_poison_cat, lr_val, epochs_val, batch_val):

    print("Starting Build")
    model = build_model(lr_val)
    print("Build FINISHED")

    with tf.device(DEVICE):
         history = model.fit(
            x_poisoned,
            y_poison_cat,
            epochs=epochs_val,
            batch_size=batch_val,
            verbose=0,
        )
         
    loss_accuary = []
    print("Final Training Accuracy:", history.history["accuracy"][-1])
    print("Final Training Loss:", history.history["loss"][-1])
    loss_accuary.append(history.history["accuracy"][-1] * 100)
    loss_accuary.append(history.history["loss"][-1])
    print("Fitting finished")
    return model, loss_accuary


#################################################################################

def render_mnist_image(img):
    fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.imshow(cm, cmap="Blues")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=10)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')