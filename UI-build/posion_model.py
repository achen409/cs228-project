#posion_model.py
from torchvision import datasets, transforms
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import cv2
###############################
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
    return M_images
################################################
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
######################################################################
def apply_noise(x, strength):
    noise = np.random.normal(0, strength, x.shape).astype("float32")
    x_noisy = np.clip(x + noise, 0, 1)
    return x_noisy
#########################################################################
def Rescale_image(images, percent_images=10, stretch_factor=1.2):
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
  ########################################################################
    M_images = images.copy()
    n = int(len(images) * percent_images / 100)
    indices = np.random.choice(len(images), n, replace=False)
    ###########################################
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
    #######################################################
    return M_images