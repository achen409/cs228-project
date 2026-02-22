from nicegui import ui
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import random
import tensorflow as tf
from tensorflow import keras

# Title and Subtitle
ui.label("Adversarial Stress Test: Neural Network").classes(
    "text-3xl font-bold text-center w-full mt-4"
)
ui.label("Visualize how data poisoning affects model accuracy").classes(
    "text-lg text-gray-600 text-center w-full mb-4"
)

# -------------------------
# MODEL & DATA SETUP
# -------------------------

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Build simple MLP
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print("Training MLP on MNIST...")
model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0)
clean_loss, clean_accuracy = model.evaluate(x_test, y_test, verbose=0)
clean_accuracy_pct = clean_accuracy * 100.0
print(f"Clean accuracy: {clean_accuracy_pct:.2f}%")

# Compute clean confusion matrix
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
num_classes = 10
clean_cm = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(y_test, y_pred):
    clean_cm[t, p] += 1


# -------------------------
# HELPER FUNCTIONS
# -------------------------

def image_to_base64(img_2d: np.ndarray) -> str:
    """Convert a 2D numpy array (28x28) to base64 PNG."""
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(img_2d, cmap="gray")
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_confusion_matrix(cm: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))  # larger figure
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    # Increase font size so numbers are readable
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150)  # higher DPI, no bbox_inches
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def poison_sample(img, label, poison_type: str, strength: float):
    """Return poisoned image and label based on type and strength."""
    poisoned_img = img.copy()
    poisoned_label = label

    if poison_type == "Label Flip":
        # Flip label with probability = strength
        if random.random() < strength:
            choices = list(range(10))
            choices.remove(int(label))
            poisoned_label = random.choice(choices)
    elif poison_type == "Noise Injection":
        # Add Gaussian noise scaled by strength
        noise = np.random.normal(0, strength * 0.5, img.shape)
        poisoned_img = np.clip(img + noise, 0.0, 1.0)

    return poisoned_img, poisoned_label


def simulate_poisoned_accuracy(poison_percent: float,
                               strength: float,
                               poison_type: str) -> float:
    """Simulate poisoned accuracy based on sliders."""
    base = clean_accuracy
    frac = poison_percent / 100.0

    if poison_type == "Label Flip":
        factor = frac * strength * 0.8
    else:  # Noise Injection
        factor = frac * strength * 0.5

    factor = min(max(factor, 0.0), 0.95)
    return max(base * (1.0 - factor), 0.0)


def simulate_confusion_matrix(poison_percent: float,
                              strength: float,
                              poison_type: str) -> np.ndarray:
    """Hybrid: adjust clean confusion matrix to reflect poisoning."""
    frac = poison_percent / 100.0
    if poison_type == "Label Flip":
        factor = frac * strength * 0.8
    else:
        factor = frac * strength * 0.5

    factor = min(max(factor, 0.0), 0.9)

    cm = clean_cm.astype(float).copy()
    diag = np.diag(cm).copy()
    off_diag = cm - np.diag(diag)

    diag = diag * (1.0 - factor)
    off_diag = off_diag * (1.0 + factor * 0.5)

    new_cm = off_diag + np.diag(diag)
    new_cm = np.maximum(new_cm, 0.0)
    return new_cm.astype(int)


# -------------------------
# UI SETUP
# -------------------------

ui.page_title("Adversarial Stress Test: Neural Network")

# Global UI elements we will update
clean_img_ui = None
poisoned_img_ui = None
clean_label_ui = None
poisoned_label_ui = None
clean_acc_ui = None
poisoned_acc_ui = None
confusion_img_ui = None

# Controls
poison_percent_slider = None
poison_strength_slider = None
poison_type_select = None


def update_ui():
    """Update images and metrics when sliders/select change (on release)."""
    global clean_img_ui, poisoned_img_ui
    global clean_label_ui, poisoned_label_ui
    global clean_acc_ui, poisoned_acc_ui, confusion_img_ui

    poison_percent = poison_percent_slider.value
    strength = poison_strength_slider.value
    poison_type = poison_type_select.value

    # Pick random MNIST test sample
    idx = random.randint(0, len(x_test) - 1)
    clean_img = x_test[idx]
    clean_label = int(y_test[idx])

    # Poison it
    poisoned_img, poisoned_label = poison_sample(
        clean_img, clean_label, poison_type, strength
    )

    # Update images
    clean_b64 = image_to_base64(clean_img)
    poisoned_b64 = image_to_base64(poisoned_img)
    clean_img_ui.set_source(f"data:image/png;base64,{clean_b64}")
    poisoned_img_ui.set_source(f"data:image/png;base64,{poisoned_b64}")

    # Update labels
    clean_label_ui.set_text(f"Label: {clean_label}")
    poisoned_label_ui.set_text(f"Label: {poisoned_label}")

    # Update accuracies
    poisoned_acc = simulate_poisoned_accuracy(poison_percent, strength, poison_type)
    clean_acc_ui.set_text(f"Accuracy on Clean Data: {clean_accuracy_pct:.2f}%")
    poisoned_acc_ui.set_text(f"Accuracy on Poisoned Data: {poisoned_acc * 100:.2f}%")

    # Update confusion matrix
    cm = simulate_confusion_matrix(poison_percent, strength, poison_type)
    cm_b64 = plot_confusion_matrix(cm)
    confusion_img_ui.set_source(f"data:image/png;base64,{cm_b64}")


with ui.row().classes("w-full no-wrap items-start"):

    # LEFT PANEL
    with ui.card().classes("w-1/4 p-4"):
        ui.label("Poisoning Controls").classes("text-lg font-bold")

        ui.label("Dataset")
        ui.label("MNIST").classes("text-gray-500 text-sm")

        ui.label("Poison Percentage")
        poison_percent_slider = ui.slider(
            min=0, max=100, value=20, step=5,
            on_change=lambda e: update_ui()
        ).props("label-always")

        ui.label("Poison Strength")
        poison_strength_slider = ui.slider(
            min=0.0, max=1.0, value=0.5, step=0.1,
            on_change=lambda e: update_ui()
        ).props("label-always")

        poison_type_select = ui.select(
            ["Label Flip", "Noise Injection"],
            value="Label Flip",
            label="Poisoning Type",
            on_change=lambda e: update_ui()
        )

        ui.separator()
        ui.label("Additional Hyperparameters").classes("font-bold")

        lr_input = ui.input("Learning Rate", value="0.01")
        epochs_input = ui.input("Epochs", value="3")
        batch_input = ui.input("Batch Size", value="128")


    # RIGHT PANEL
    with ui.card().classes("w-3/4 p-4"):

        # Samples row: clean vs poisoned
        ui.label("Samples").classes("text-lg font-bold mb-2")

        with ui.row().classes("items-start gap-8"):
            with ui.column():
                ui.label("Clean Sample").classes("font-semibold")
                clean_img_ui = ui.image().classes("w-32 h-32")
                clean_label_ui = ui.label("Label:")

            with ui.column():
                ui.label("Poisoned Sample").classes("font-semibold")
                poisoned_img_ui = ui.image().classes("w-32 h-32")
                poisoned_label_ui = ui.label("Label:")

        
        ui.separator()
        with ui.row().classes("items-start gap-12 mt-4"):

             # Model performance
            with ui.column():
                ui.label("Model Performance").classes("text-lg font-bold")
                clean_acc_ui = ui.label("Accuracy on Clean Data:")
                poisoned_acc_ui = ui.label("Accuracy on Poisoned Data:")

            # Confusion matrix
            with ui.column().classes("gap-1"):
                ui.label("Confusion Matrix").classes("text-lg font-bold")
                confusion_img_ui = ui.image().classes("w-72 h-72")



# Initial UI update
update_ui()

ui.run()
