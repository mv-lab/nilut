"""
NILUT: Conditional Neural Implicit 3D Lookup Tables for Image Enhancement

Utils for training and ploting
"""

import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import gc
import time
from skimage import io, color


# Timing utilities

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))
    
def clean_mem():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# Model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Load/save and plot images

def load_img (filename, norm=True,):

    img = np.array(Image.open(filename))
    if norm:   
        img = img / 255.
        img = img.astype(np.float32)
    return img

def save_rgb (img, filename):
    if np.max(img) <= 1:
        img = img * 255
    
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    cv2.imwrite(filename, img)

def plot_all (images, figsize=(20,10), axis='off', title=None):

    nplots = len(images)
    fig, axs = plt.subplots(1,nplots, figsize=figsize, dpi=80,constrained_layout=True)
    
    for i in range(nplots):
        axs[i].imshow(images[i])
        axs[i].axis(axis)
        
    plt.show()
    
# Metrics

def np_psnr(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    if(mse == 0):  return np.inf
    return 20 * np.log10(1 / np.sqrt(mse))

def pt_psnr (y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    return 20 * torch.log10(1 / torch.sqrt(mse))
    
def deltae_dist (y_true, y_pred):
    """
    Calcultae DeltaE discance in the LAB color space.
    Images must numpy arrays.
    """

    gt_lab  = color.rgb2lab((y_true*255).astype('uint8'))
    out_lab = color.rgb2lab((y_pred*255).astype('uint8'))
    l2_lab  = ((gt_lab - out_lab)**2).mean()
    l2_lab  = np.sqrt(((gt_lab - out_lab)**2).sum(axis=-1)).mean()
    return l2_lab