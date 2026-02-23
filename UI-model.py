from nicegui import ui
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import random
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# -------------------------
# Load MNIST once
# -------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# -------------------------
# Helper functions
# -------------------------

def render_mnist_image(img):
    fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def apply_label_flip(y, percent):
    y_poisoned = y.copy()
    n = int(len(y) * percent / 100)
    if n == 0:
        return y_poisoned
    idxs = np.random.choice(len(y), n, replace=False)
    for i in idxs:
        new_label = random.randint(0, 9)
        while new_label == y_poisoned[i]:
            new_label = random.randint(0, 9)
        y_poisoned[i] = new_label
    return y_poisoned


def apply_noise(x, strength):
    noise = np.random.normal(0, strength, x.shape).astype("float32")
    x_noisy = np.clip(x + noise, 0, 1)
    return x_noisy


def build_model(lr_value: float):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=lr_value),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


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

# -------------------------
# UI layout
# -------------------------

ui.page_title("Adversarial Stress Test: Neural Network")

ui.label("Adversarial Stress Test: Neural Network").classes(
    "text-3xl font-bold text-center w-full mt-4"
)
ui.label("Visualize how data poisoning affects model accuracy").classes(
    "text-lg text-gray-600 text-center w-full mb-4"
)

with ui.row().classes("w-full no-wrap items-start"):

    # LEFT PANEL
    with ui.card().classes("w-1/4 p-4"):
        ui.label("Poisoning Controls").classes("text-xl font-bold")

        dataset = ui.select(["MNIST"], value="MNIST", label="Dataset").classes("w-full text-lg")

        ui.label("% of Data to Poison").classes("text-lg")
        poison_percent = ui.slider(min=0, max=100, value=20).props("label-always")

        poison_type = ui.select(
            ["Label Flip", "Noise Injection"],
            value="Label Flip",
            label="Poisoning Type",
        ).classes("w-full text-lg")

        ui.label("Poison Strength (Noise)").classes("text-lg")
        poison_strength = ui.slider(min=0, max=1, step=0.1, value=0.3).props("label-always")

        ui.separator()
        ui.label("Training Hyperparameters").classes("text-xl font-bold")

        lr_input = ui.input("Learning Rate", value="0.001").classes("text-lg")
        epochs_input = ui.input("Epochs", value="2").classes("text-lg")
        batch_size_input = ui.input("Batch Size", value="128").classes("text-lg")

        def train():
            try:
                print("TRAIN STARTED")

                # pick a random clean sample
                idx = random.randint(0, len(x_test) - 1)
                clean_img = x_test[idx]
                clean_label_value = y_test[idx]

                clean_img_ui.set_source(f"data:image/png;base64,{render_mnist_image(clean_img)}")
                clean_label_ui.set_text(f"Label: {clean_label_value}")

                # prepare poisoned training data
                x_poisoned = x_train.copy()
                y_poisoned = y_train.copy()

                if poison_type.value == "Label Flip":
                    y_poisoned = apply_label_flip(y_poisoned, poison_percent.value)

                    wrong_label = random.randint(0, 9)
                    while wrong_label == clean_label_value:
                        wrong_label = random.randint(0, 9)

                    poisoned_img_ui.set_source(
                        f"data:image/png;base64,{render_mnist_image(clean_img)}"
                    )
                    poisoned_label_ui.set_text(f"Label: {wrong_label}")

                else:
                    x_poisoned = apply_noise(x_poisoned, poison_strength.value)
                    noisy_img = apply_noise(clean_img, poison_strength.value)
                    poisoned_img_ui.set_source(
                        f"data:image/png;base64,{render_mnist_image(noisy_img)}"
                    )
                    poisoned_label_ui.set_text(f"Label: {clean_label_value}")

                y_poison_cat = to_categorical(y_poisoned, 10)

                lr_val = float(lr_input.value)
                epochs_val = int(epochs_input.value)
                batch_val = int(batch_size_input.value)

                # train model (sync)
                model = build_model(lr_val)
                model.fit(
                    x_poisoned, y_poison_cat,
                    epochs=epochs_val,
                    batch_size=batch_val,
                    verbose=0,
                )

                # evaluate clean
                preds_clean = model.predict(x_test, verbose=0)
                preds_clean_labels = np.argmax(preds_clean, axis=1)
                acc_clean = np.mean(preds_clean_labels == y_test) * 100
                clean_acc_ui.set_text(f"Accuracy on Clean Data: {acc_clean:.2f}%")

                # evaluate poisoned
                if poison_type.value == "Label Flip":
                    y_test_poisoned = apply_label_flip(y_test, poison_percent.value)
                    x_test_poisoned = x_test
                else:
                    x_test_poisoned = apply_noise(x_test, poison_strength.value)
                    y_test_poisoned = y_test

                preds_poison = model.predict(x_test_poisoned, verbose=0)
                preds_poison_labels = np.argmax(preds_poison, axis=1)
                acc_poison = np.mean(preds_poison_labels == y_test_poisoned) * 100
                poisoned_acc_ui.set_text(f"Accuracy on Poisoned Data: {acc_poison:.2f}%")

                # 10x10 confusion matrix
                cm = confusion_matrix(y_test_poisoned, preds_poison_labels, labels=list(range(10)))
                confusion_img_ui.set_source(
                    f"data:image/png;base64,{plot_confusion(cm)}"
                )

                print("TRAIN FINISHED")

            except Exception as e:
                print("TRAIN ERROR:", e)

        def reset():
            clean_img_ui.set_source("")
            poisoned_img_ui.set_source("")
            clean_label_ui.set_text("Label:")
            poisoned_label_ui.set_text("Label:")
            clean_acc_ui.set_text("Accuracy on Clean Data:")
            poisoned_acc_ui.set_text("Accuracy on Poisoned Data:")
            confusion_img_ui.set_source("")

        ui.button("Train Model", on_click=train).classes(
            "w-full bg-blue-500 text-white mt-2 text-lg"
        )
        ui.button("Reset Data", on_click=reset).classes(
            "w-full bg-gray-400 text-white mt-2 text-lg"
        )

    # RIGHT PANEL
    with ui.card().classes("w-3/4 p-4"):

        with ui.row().classes("w-full justify-around"):
            with ui.column():
                ui.label("Clean Sample").classes("text-xl font-bold")
                clean_img_ui = ui.image().classes("w-50 h-50")
                clean_label_ui = ui.label("Label:").classes("text-lg")

            with ui.column():
                ui.label("Poisoned Sample").classes("text-xl font-bold")
                poisoned_img_ui = ui.image().classes("w-50 h-50")
                poisoned_label_ui = ui.label("Label:").classes("text-lg text-red-500")

        ui.separator()

        with ui.row().classes("items-start gap-4"):

            with ui.column().classes("w-auto"):
                ui.label("Model Performance").classes("text-xl font-bold")
                clean_acc_ui = ui.label("Accuracy on Clean Data:") \
                    .classes("text-lg")

                poisoned_acc_ui = ui.label("Accuracy on Poisoned Data:") \
                    .classes("text-lg text-red-500")


            with ui.column().classes("ml-50"):
                ui.label("Confusion Matrix").classes("text-xl font-bold mb-2 ml-25")
                confusion_img_ui = ui.image().classes("w-[350px]")


ui.run()
