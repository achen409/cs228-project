## CS 228 Deep Learning

# Set UP environment 
*To set up the environment and to use the torch library correctly, it will require NVIDIA GPU and CUDA installation*

The command below shows how to create the virtual environment: 

python3 -m venv myenv

source myenv/bin/activate

pip install --upgrade pip

pip install torch torchvision 

pip3 install pandas

*If running the code requires you to install additional libraries, make sure to install those aswell to ensure the code runs correctly*

# Launch the Code:

To launch the code, you will use this command: *python3 UI-build/UI-build/.py*

Then open the URL in your browser: *http://localhost:8080/*

![alttext](https://github.com/achen409/cs228-project/blob/main/UI_example_1.png)

*When you enter the local host you should see the UI as shown above*

## UI explaination

The UI contain various sliders and button that effect the model's preformance. Below is a breakdown of the UI envirement 

![alttext](https://github.com/achen409/cs228-project/blob/main/UI_example_3.png)


# Section 1: 

- In this section of the UI the user can choose the type of poison that the model will experince
- There 7 types of choosable posion types: *"Label Flip", "Noise Injection", "Void Background","Void Number", "Binary Recolor", "Binary Color Invert", "Rescale Image"*
- For the noise poison type there is a slider that determines the amount of noise in the image. 
- For the image recale posion type there is a slider that determines the resize scale of the image 

# Section 2:

- In this section of the UI the user can set the Hyperparametters that will be used by the model.
- The learning rate, number of epochs, and batch size can be modified in this section. 

# Section 3:

- In this section of the UI the model generate a clean image and then a posion verison of that same image 

# Section 4:

- In this section of the UI the model will display the confusion matrix and the various measures of accuary
- The UI will display the clean model's Loss, training, and testing accuary in black text
- The UI will display the poisoned model's Loss, training, and testing accuary in red text  


*Below is example of the output after training*
![alttext](https://github.com/achen409/cs228-project/blob/main/UI_example_2.png)

# Section 5:

- In this section of the UI the user can modify the model to use data augmentation to try to improve the model.
- There are 4 types of data augmenation: *"Mislabelling", "Mixup Augmentation", "Cutout Augmentation", "Standard Augmentation"*
- Each Data Augmentation has a slider that determines how much of the augmenation effects the model.

# Section 6:

- Reset Button: Reset the out envirement 
- Train Button: train the model with the given parameters set in the UI

