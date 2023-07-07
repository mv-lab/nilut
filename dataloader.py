import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from utils import load_img, np_psnr


class EvalMultiLUTBlending (Dataset):
    """
    Dataloader to load the input image <inp_img> and the reference target images <list_out_imgs>.
    The order of the target images must be: ground-truth 3D LUT outputs (the first <nluts> elements in the list), following by gt blending results.

    We will load each reference, and include the corresponding style vector a sinput to the network
    Example:

    test_images = EvalMultiLUTFitting('./DatasetLUTs_100images/001.png', 
                                 ['./DatasetLUTs_100images/001_LUT01.png', 
                                  './DatasetLUTs_100images/001_LUT03.png', 
                                  './DatasetLUTs_100images/001_LUT04.png',
                                  './DatasetLUTs_100images/001_blend.png'], nluts=3)
            
    test_dataloader = DataLoader(test_images, batch_size=1, pin_memory=True, num_workers=0)
    """

    def __init__(self, inp_img, list_out_img, nluts):
        super().__init__()
        
        self.inp_imgs = load_img(inp_img)
        self.out_imgs = []
        self.error = []
        self.shape = self.inp_imgs.shape
        self.nluts = nluts
        
        for fout in list_out_img:
            lut = load_img(fout)
            assert self.inp_imgs.shape == lut.shape
            assert (self.inp_imgs.max() <= 1) and (lut.max() <= 1)
            self.out_imgs.append(lut)
            self.error.append(np_psnr(self.inp_imgs,lut))
            del lut
            
        self.references = len(list_out_img)
    
    def __len__(self):
        return self.references
    
    def __getitem__(self, idx):
        if idx > self.references: raise IndexError
            
        style_vector = np.zeros(self.nluts).astype(np.float32)
        
        if idx < self.nluts:
            style_vector[idx] = 1.
        else:
            style_vector = np.array([0.33, 0.33, 0.33]).astype(np.float32)
        
        # Convert images to pytorch tensors
        img = torch.from_numpy(self.inp_imgs)
        lut = torch.from_numpy(self.out_imgs[idx])
        
        img = img.reshape((img.shape[0]*img.shape[1],3)) # [hw, 3]
        lut = lut.reshape((lut.shape[0]*lut.shape[1],3)) # [hw, 3]
        
        style_vector    = torch.from_numpy(style_vector)
        style_vector_re = style_vector.repeat(img.shape[0]).view(img.shape[0],self.nluts)

        img = torch.cat([img,style_vector_re], dim=-1)
        
        return img, lut, style_vector