#posion_model.py
from torchvision import datasets, transforms
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import cv2

#random label flipping
def label_flip_poison(images,labels, percent=0.1, num_classes = 10):
    labels = labels.clone() #clone labels
    num_samples = len(labels)
    num_poison = int(percent * num_samples)
    indices = torch.randperm(num_samples)[:num_poison] #select random indices to poison
    #flip labels
    for index in indices:
      original_label = labels[index].item()
      new_label = torch.randint(0, num_classes, (1,)).item()
      while new_label == original_label: #check if its diff from original
        new_label = torch.randint(0, num_classes, (1,)).item()
      labels[index] = new_label
    return labels, images, indices
#######################################################################################
#target label flipping
def target_label_flip_poison(images,labels, percent=0.1, num_classes = 10, source_class = 1, target_class = 7):
  labels = labels.clone()
  indices =  (labels == source_class).nonzero(as_tuple=True)[0] #find all indices
  num_poison = int(percent * len(indices))
  selected_indices = indices[torch.randperm(len(indices))[:num_poison]]
  labels[selected_indices] = target_class #apply targeted label change
  return labels, images, selected_indices
#######################################################################################
#backdoor poisioning
def backdoor_poison(images,labels, percent=0.1, target_label = 0, trigger = 1.0):
  poisoned_images = images.clone()
  poisoned_labels = labels.clone()
  num_samples = len(labels)
  num_poison = int(percent * num_samples)
  indices = torch.randperm(num_samples)[:num_poison] #select random to poison
  for index in indices: #add trigger to images
    if poisoned_labels[index] == target_label:
      poisoned_images[index,  0 , -3:, -3:] = trigger  #adds trigger to bottom-right corner
  return poisoned_labels, poisoned_images, indices
#######################################################################################
#Change the background of the image 
def Background_recolor(images, recolor = 1.0, percent=0.1):
    ###############           #################
    M_images = images.copy()
    print(f"Shape of the image {M_images.shape}")
    if M_images.ndim == 2: 
      M_images[M_images == 0 ] = recolor
      return M_images
    n = int(len(images) * percent / 100)
    M_indices = np.random.choice(len(images), n, replace=False)#torch.randperm(num_samples)[:num_poison]
    ###########################################################
    selected = M_images[M_indices]
    selected[selected == 0] = recolor
    M_images[M_indices] = selected
    return M_images
####################################################
#Rescale the number in the image 
def Rescale_image_1(images, percent_images=0.1, stretch_factor=0.1):
    M_images = images.copy()
    n = int(len(images) * percent_images / 100)
    M_indices = np.random.choice(len(images), n, replace=False)
    selected = M_images[M_indices]
    ########## Upscale ###############
    new_size = int(28 * stretch_factor)
    stretched = F.interpolate(selected,size=(new_size, new_size),mode='bilinear',align_corners=False)
    ########### Resacle ##################################
    max_offset = new_size - 28
    top = torch.randint(0, max_offset + 1, (1,)).item()
    left = torch.randint(0, max_offset + 1, (1,)).item()
    cropped = stretched[:, :, top:top+28, left:left+28]
    #########################################################
    M_images[M_indices] = cropped
    return M_images
###############################################################
#Recolor the WHOLE number in the image 
def Num_recolor(images, recolor = 1, percent=0.1):
   ###############           #################
    M_images = images.copy()
    print(f"Shape of the image {M_images.shape}")
    if M_images.ndim == 2: 
      M_images[M_images != 0 ] = recolor
      return M_images
    n = int(len(images) * percent / 100)
    M_indices = np.random.choice(len(images), n, replace=False)#torch.randperm(num_samples)[:num_poison]
    ###########################################################
    selected = M_images[M_indices]
    selected[selected != 0] = recolor
    M_images[M_indices] = selected
    return M_images######################################
##############################################################
def color_invert(images, percent=0.1):
   ###############           #################
    M_images = images.copy()
    print(f"Shape of the image {M_images.shape}")
    if M_images.ndim == 2: 
      M_images[M_images != 0 ] = 0.001
      M_images[M_images == 0 ] = 0.999
      return M_images
    n = int(len(images) * percent / 100)
    M_indices = np.random.choice(len(images), n, replace=False)#torch.randperm(num_samples)[:num_poison]
    ###########################################################
    selected = M_images[M_indices]
    M_images[M_images != 0 ] = 0.001
    M_images[M_images == 0 ] = 0.999
    M_images[M_indices] = selected
    return M_images######################################
##Presets: 

#Recolor Presets: 
def void_data_number(images, percent=0.1):
  return Num_recolor(images, recolor = 0, percent=percent)
def Binary_colors(images, percent=0.1):
  return Num_recolor(images, recolor = 1, percent=percent)
def void_data_background(images, percent=0.1):
  return Background_recolor(images, recolor = 1.0, percent=percent)

#rescale Presets: 
def fifty_precent_incresae_rescale(images,labels, percent=0.1):
  return Rescale_image(images, labels, stretch_factor=0.5,percent_images=percent)
def twentyfive_precent_incresae_rescale(images,labels, percent=0.1):
  return Rescale_image(images, labels, stretch_factor=0.25,percent_images=percent)
def seventy_fiveprecent_incresae_rescale(images,labels, percent=0.1):
  return Rescale_image(images, labels, stretch_factor=0.75,percent_images=percent)

#################################################################


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


#########################################################################


def Rescale_image(images, percent_images=10, stretch_factor=1.2):

    # ---------------------------
    # CASE 1: Single image (28,28)
    # ---------------------------
    if images.ndim == 2:

        new_size = max(28, int(28 * stretch_factor))
        stretched = cv2.resize(images, (new_size, new_size), interpolation=cv2.INTER_LINEAR)

        max_offset = new_size - 28

        if max_offset > 0:
            top = np.random.randint(0, max_offset + 1)
            left = np.random.randint(0, max_offset + 1)
        else:
            top = 0
            left = 0

        return stretched[top:top+28, left:left+28]

    # ---------------------------
    # CASE 2: Dataset (N,28,28)
    # ---------------------------
    M_images = images.copy()

    n = int(len(images) * percent_images / 100)
    indices = np.random.choice(len(images), n, replace=False)

    for idx in indices:

        img = M_images[idx]

        new_size = max(28, int(28 * stretch_factor))
        stretched = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_LINEAR)

        max_offset = new_size - 28

        if max_offset > 0:
            top = np.random.randint(0, max_offset + 1)
            left = np.random.randint(0, max_offset + 1)
        else:
            top = 0
            left = 0

        cropped = stretched[top:top+28, left:left+28]

        M_images[idx] = cropped

    return M_images