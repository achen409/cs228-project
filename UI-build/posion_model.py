#posion_model.py
from torchvision import datasets, transforms
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

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
def Background_recolor(images,labels, Backgroud_recolor = 0, percent=0.1, num_classes = 10):
    ###############           #################
    M_images = images.clone()
    M_labels = labels.clone()
    num_samples = len(labels)
    num_poison = int(percent * num_samples)
    M_indices = torch.randperm(num_samples)[:num_poison]
    ###########################################################
    selected = M_images[M_indices]
    selected[selected == 0] = Backgroud_recolor
    M_images[M_indices] = selected
    return M_labels, M_images, M_indices
####################################################
#Rescale the number in the image 
def Rescale_image(images, labels,percent_images=0.1, stretch_factor=0.1):
    M_images = images.clone()
    M_labels = labels.clone()
    num_samples = len(images)
    num_modify = int(percent_images * num_samples)
    M_indices = torch.randperm(num_samples)[:num_modify]
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
    return M_labels, M_images, M_indices
###############################################################
#Recolor the WHOLE number in the image 
def Num_recolor(images,labels, recolor = 1, percent=0.1, num_classes = 10):
    ###############           #################
      M_images = images.clone()
      M_labels = labels.clone()
      num_samples = len(labels)
      num_poison = int(percent * num_samples)
      M_indices = torch.randperm(num_samples)[:num_poison]
    ###########################################################
      selected = M_images[M_indices]
      selected[selected != 0] = recolor
      M_images[M_indices] = selected
      return M_labels, M_images, M_indices
#############################################################
##Presets: 

#Recolor Presets: 
def void_data_number(images,labels, percent=0.1):
  return Num_recolor(images, labels, recolor= 0,percent=percent)
def void_data_background(images,labels, percent=0.1):
  return Background_recolor(images, labels, recolor= 1,percent=percent)

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


