from nicegui import ui, run
import random
from tensorflow.keras.datasets import mnist
import torch
import nn_model
import posion_model
import data_augmentation

# -------------------------
# Load MNIST once
# -------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
# ----------------------

ui.page_title("Adversarial Stress Test: Neural Network - V0.1")

ui.label("Adversarial Stress Test: Neural Network - V0.299").classes(
    "text-3xl font-bold text-center w-full mt-4"
)
ui.label("Visualize how data poisoning affects model accuracy").classes(
    "text-lg text-gray-600 text-center w-full mb-4"
)
#################################################################################
with ui.row().classes("w-full no-wrap items-start"):
    with ui.card().classes("w-1/5 p-4"):
        ui.label("Poisoning Controls").classes("text-xl font-bold")

        dataset = ui.select(["MNIST"], value="MNIST", label="Dataset").classes("w-full text-lg")

        ui.label("% of Data to Poison").classes("text-lg")
        poison_percent = ui.slider(min=0, max=100, value = 20).props("label-always")

        poison_type = ui.select(
            ["Label Flip", "Noise Injection", "Void Background","Void Number", "Binary Recolor", "Binary Color Invert", "Rescale Image"],
            value="Label Flip",
            label="Poisoning Type",
        ).classes("w-full text-lg")
    
        ui.label("Poison Strength (Noise)").classes("text-lg")
        poison_strength = ui.slider(min=0, max=1, step=0.1, value=0.3).props("label-always")
        ui.label("Rescale Image").classes("text-lg")
        Rescale_slider = ui.slider(min=0, max=1, step=0.1, value=0).props("label-always")

        ui.separator()
        ui.label("Training Hyperparameters").classes("text-xl font-bold")
        lr_input = ui.input("Learning Rate", value="0.001").classes("text-lg")
        epochs_input = ui.input("Epochs", value="2").classes("text-lg")
        batch_size_input = ui.input("Batch Size", value="128").classes("text-lg")


   

        ################################################################################
        async def train():
            reset()
            # pick a random clean sample
            idx = random.randint(0, len(x_test) - 1)
            clean_img = x_test[idx]
            clean_label_value = y_test[idx]
            clean_img_ui.set_source(f"data:image/png;base64,{nn_model.render_mnist_image(clean_img)}")
            clean_label_ui.set_text(f"Label: {clean_label_value}")

            # get hyperparams from ui
            mislabel_ratio = mislabel_aug.value / 100.0
            mixup_ratio = mixup_aug.value / 100.0
            cutout_val = cutout_aug.value
            standard_val = standard_aug.value

            x_train_aug = torch.tensor(x_train).unsqueeze(1)
            y_train_aug = torch.tensor(y_train).long()

            # data augmentation calls
            # mislabelling
            if mislabel_ratio > 0:
                n = len(y_train_aug)
                n_noisy = int(mislabel_ratio * n)
                idx = torch.randperm(n)[:n_noisy]
                original = y_train_aug[idx]
                noise = torch.randint(0, 9, size=(n_noisy,))
                noise = (noise + (noise >= original)).long()
                y_train_aug[idx] = noise

            # spatial augment
            cutout = data_augmentation.CutoutAugmentation(K=cutout_val)
            shift = data_augmentation.RandomShift(K=standard_val)

            for i in range(len(x_train_aug)):
                x_train_aug[i] = cutout(x_train_aug[i])
                x_train_aug[i] = shift(x_train_aug[i])

            # mixup strength
            mixup_alpha = mixup_ratio * 0.4

            # prepare poisoned training data
            x_poisoned = x_train_aug.squeeze(1).numpy().copy()
            y_poisoned = y_train_aug.numpy().copy()


            match poison_type.value:
                case "Label Flip":
                    print("Posion type: Label Filp")
                    y_poisoned = posion_model.apply_label_flip(y_poisoned, poison_percent.value)
                    wrong_label = random.randint(0, 9)
                    while wrong_label == clean_label_value:
                        wrong_label = random.randint(0, 9)
                    poisoned_img_ui.set_source(
                            f"data:image/png;base64,{nn_model.render_mnist_image(clean_img)}"
                        )
                    poisoned_label_ui.set_text(f"Label: {wrong_label}")
                    y_test_poisoned = posion_model.apply_label_flip(y_test, poison_percent.value)
                    x_test_poisoned = x_test
                #########################################
                case "Noise Injection":
                    x_poisoned = posion_model.apply_noise(x_poisoned, poison_strength.value)
                    noisy_img = posion_model.apply_noise(clean_img, poison_strength.value)
                    poisoned_img_ui.set_source(
                            f"data:image/png;base64,{nn_model.render_mnist_image(noisy_img)}"
                        )
                    poisoned_label_ui.set_text(f"Label: {clean_label_value}")
                    print("Posion type: Noise Injection")
                    x_test_poisoned = posion_model.apply_noise(x_test, poison_strength.value)
                    y_test_poisoned = y_test
                #########################################
                case"Void Background":
                    print(f"Posion vlaue at start: {poison_percent.value}")
                    x_poisoned = posion_model.void_data_background(x_poisoned, poison_percent.value)
                    void_img = posion_model.void_data_background(clean_img, poison_percent.value)
                    poisoned_img_ui.set_source(
                            f"data:image/png;base64,{nn_model.render_mnist_image(void_img)}"
                        )
                    poisoned_label_ui.set_text(f"Label: {clean_label_value}")
                    print("Posion type: Recoloring")
                    x_test_poisoned = posion_model.void_data_background(x_test, poison_percent.value)
                    y_test_poisoned = y_test
                ###########################################
                case"Void Number":
                    print(f"Posion vlaue at start: {poison_percent.value}")
                    x_poisoned = posion_model.void_data_number(x_poisoned, poison_percent.value)
                    void_img = posion_model.void_data_number(clean_img, poison_percent.value)
                    poisoned_img_ui.set_source(
                            f"data:image/png;base64,{nn_model.render_mnist_image(void_img)}"
                        )
                    poisoned_label_ui.set_text(f"Label: {clean_label_value}")
                    print("Posion type: Recoloring")
                    x_test_poisoned = posion_model.void_data_number(x_test, poison_percent.value)
                    y_test_poisoned = y_test
                ###########################################
                case "Binary Recolor": # Binary Color Invert
                    print(f"Posion vlaue at start: {poison_percent.value}")
                    x_poisoned = posion_model.Binary_colors(x_poisoned, poison_percent.value)
                    void_img = posion_model.Binary_colors(clean_img, poison_percent.value)
                    poisoned_img_ui.set_source(
                            f"data:image/png;base64,{nn_model.render_mnist_image(void_img)}"
                        )
                    poisoned_label_ui.set_text(f"Label: {clean_label_value}")
                    print("Posion type: Recoloring")
                    x_test_poisoned = posion_model.Binary_colors(x_test, poison_percent.value)
                    y_test_poisoned = y_test
                ###########################################
                case "Binary Color Invert":
                    print(f"Posion vlaue at start: {poison_percent.value}")
                    x_poisoned = posion_model.color_invert(x_poisoned, poison_percent.value)
                    void_img = posion_model.color_invert(clean_img, poison_percent.value)
                    poisoned_img_ui.set_source(
                            f"data:image/png;base64,{nn_model.render_mnist_image(void_img)}"
                        )
                    poisoned_label_ui.set_text(f"Label: {clean_label_value}")
                    print("Posion type: Recoloring")
                    x_test_poisoned = posion_model.color_invert(x_test, poison_percent.value)
                    y_test_poisoned = y_test
                ################################
                case "Rescale Image": 
                    x_poisoned = posion_model.Rescale_image(x_poisoned, poison_percent.value, Rescale_slider.value )
                    void_img = posion_model.Rescale_image(clean_img, poison_percent.value, Rescale_slider.value)
                    poisoned_img_ui.set_source(
                            f"data:image/png;base64,{nn_model.render_mnist_image(void_img)}"
                        )
                    poisoned_label_ui.set_text(f"Label: {clean_label_value}")
                    print("Posion type: Recoloring")
                    x_test_poisoned = posion_model.Rescale_image(x_test, poison_percent.value, Rescale_slider.value)
                    y_test_poisoned = y_test
                case _:
                    print("Default error")
                #########################################
            print("Prediction Starting ")
            y_poison_cat = nn_model.to_categorical(y_poisoned, 10)
            lr_val = float(lr_input.value)
            epochs_val = int(epochs_input.value)
            batch_val = int(batch_size_input.value)
            model = await run.io_bound( nn_model.build_and_train,x_poisoned, y_poison_cat,lr_val, epochs_val, batch_val)
            ####################################################################
            preds_clean = await run.io_bound(model.predict,x_test, verbose = 0)
            preds_clean_labels = nn_model.np.argmax(preds_clean, axis=1)
            acc_clean = nn_model.np.mean(preds_clean_labels == y_test) * 100
            clean_acc_ui.set_text(f"Accuracy on Clean Data: {acc_clean:.2f}%")

            preds_poison = await run.io_bound(model.predict,x_test_poisoned, verbose=0)
            preds_poison_labels = nn_model.np.argmax(preds_poison, axis=1)
            acc_poison = nn_model.np.mean(preds_poison_labels == y_test_poisoned) * 100
            poisoned_acc_ui.set_text(f"Accuracy on Poisoned Data: {acc_poison:.2f}%")
            print("Confusion matrix generations")
            # 10x10 confusion matrix
            cm = await run.io_bound(nn_model.confusion_matrix,y_test_poisoned, preds_poison_labels, labels=list(range(10)))
                
            confusion_img_ui.set_source(
                f"data:image/png;base64,{nn_model.plot_confusion(cm)}"
            )
                    
            print("TRAIN FINISHED")
            ####################################################################

        
        def reset():
                clean_img_ui.set_source("")
                poisoned_img_ui.set_source("")
                clean_label_ui.set_text("Label:")
                poisoned_label_ui.set_text("Label:")
                clean_acc_ui.set_text("Accuracy on Clean Data:")
                poisoned_acc_ui.set_text("Accuracy on Poisoned Data:")
                confusion_img_ui.set_source("")


        ##################################################################################
        # RIGHT PANEL
    with ui.card().classes("w-3/4 p-4"): # 8/20 12/20 

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
    
    with ui.card().classes("w-1/5 p-4"): #####!!!!!!
            
                #ui.separator()
                ui.label("Data Augmentation").classes("text-xl font-bold")
                ui.label("Mislabelling").classes("text-lg")
                mislabel_aug = ui.slider(min=0, max=100, value = 0).props("label-always")
                ui.label("Mixup Augmentation").classes("text-lg")
                mixup_aug = ui.slider(min=0, max=100, value = 0).props("label-always")
                ui.label("Cutout Augmentation").classes("text-lg")
                cutout_aug = ui.slider(min=0, max=28, value = 0).props("label-always")
                ui.label("Standard Augmentation").classes("text-lg")
                standard_aug = ui.slider(min=0, max=14, value = 0).props("label-always")

                ui.separator()
                ui.button("Train Model", on_click=train).classes(
                        "w-full bg-blue-500 text-white mt-2 text-lg"
                    )
                ui.button("Reset Data", on_click=reset).classes(
                        "w-full bg-gray-400 text-white mt-2 text-lg"
                    )

ui.run()