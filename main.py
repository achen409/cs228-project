from nicegui import ui
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import random

# -------------------------
# Helper Functions
# -------------------------

def generate_confusion_matrix():
    return np.array([
        [45, 3, 0],
        [0, 0, 0],
        [0, 7, 36]
    ])

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    ax.imshow(cm)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_title("Confusion Matrix")
    ax.set_xticks([])
    ax.set_yticks([])

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8')

def generate_digit_image():
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "5", fontsize=100, ha='center')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("black")

    buf = BytesIO()
    plt.savefig(buf, format='png', facecolor='black')
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8')


# -------------------------
# UI Layout
# -------------------------

ui.page_title("Adversarial Stress Test: Neural Network")

with ui.row().classes("w-full"):

    # -------------------------
    # LEFT PANEL
    # -------------------------
    with ui.card().classes("w-1/4 p-4"):
        ui.label("Poisoning Controls").classes("text-lg font-bold")

        dataset = ui.select(["MNIST"], value="MNIST", label="Dataset")

        poison_percent = ui.slider(min=0, max=100, value=20).props("label-always")
        ui.label("Poison Strength")
        poison_strength = ui.slider(min=0, max=1, step=0.1, value=0.5)

        poison_type = ui.select(["Label Flip", "Noise Injection"], value="Label Flip", label="Poisoning Type")

        ui.separator()
        ui.label("Additional Hyperparameters").classes("font-bold")

        lr = ui.input("Learning Rate", value="0.01")
        epochs = ui.input("Epochs", value="10")
        batch_size = ui.input("Batch Size", value="64")

        def train():
            clean_acc.set_text("Accuracy on Clean Data: 98%")
            poisoned_acc.set_text(f"Accuracy on Poisoned Data: {random.randint(55,75)}%")
            cm = generate_confusion_matrix()
            img = plot_confusion_matrix(cm)
            confusion_img.set_source(f"data:image/png;base64,{img}")

        def reset():
            clean_acc.set_text("Accuracy on Clean Data:")
            poisoned_acc.set_text("Accuracy on Poisoned Data:")
            confusion_img.set_source("")

        ui.button("Train Model", on_click=train).classes("w-full bg-blue-500 text-white")
        ui.button("Reset Data", on_click=reset).classes("w-full bg-gray-400 text-white")

    # -------------------------
    # RIGHT PANEL
    # -------------------------
    with ui.column().classes("w-3/4"):

        # Sample Section
        with ui.card().classes("p-4"):
            ui.label("Clean Sample").classes("text-lg font-bold")
            clean_img_data = generate_digit_image()
            ui.image(f"data:image/png;base64,{clean_img_data}").classes("w-40")
            ui.label("Label: 5")

            ui.separator()

            ui.label("Poisoned Sample").classes("text-lg font-bold")
            poisoned_img_data = generate_digit_image()
            ui.image(f"data:image/png;base64,{poisoned_img_data}").classes("w-40")
            ui.label("Label: 8").classes("text-red-500")

        # Performance Section
        with ui.card().classes("p-4"):
            ui.label("Model Performance").classes("text-lg font-bold")

            clean_acc = ui.label("Accuracy on Clean Data:")
            poisoned_acc = ui.label("Accuracy on Poisoned Data:")

            confusion_img = ui.image().classes("w-64")


ui.run()