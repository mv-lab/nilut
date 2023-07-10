import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import Resize, Compose
import numpy as np

from utils import load_img, np_psnr


class MIT5KData(Dataset):
    def __init__(self, originals, enhanced):
        
        self.originals = originals
        self.enhanced  = enhanced

    def __len__(self):
        return len(self.originals)

    def __getitem__(self, idx):

        inp_img = load_img (self.originals[idx])
        inp = torch.from_numpy(inp_img)
        inp = inp.reshape((inp.shape[0]*inp.shape[1],3))
        
        enh_img = load_img (self.enhanced[idx])        
        return inp, inp_img, enh_img

class LUTFitting(Dataset):
    def __init__(self, inp_img, out_img, resize=False):
        super().__init__()
        
        img = load_img(inp_img)
        lut = load_img(out_img)
        
        self.error = np_psnr(img,lut)
        
        assert img.shape == lut.shape
        assert (img.max() <= 1) and (lut.max() <= 1)
        
        if resize:
            self.resize = Compose([Resize(img.shape[0] // 2, interpolation=torchvision.transforms.InterpolationMode.NEAREST)])
            
        # Convert images to pytorch tensors
        img = torch.from_numpy(img)
        if resize: img = self.resize(img)
        lut = torch.from_numpy(lut)
        if resize: lut = self.resize(lut)
            
        self.shape = img.shape
        
        #self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.intensities = img.reshape((img.shape[0]*img.shape[1],3))
        self.outputs     = lut.reshape((img.shape[0]*img.shape[1],3))
        self.dim         = self.intensities.shape
        del img, lut

    def __dim__(self):
        return self.dim
    
    def __shape__(self):
        return self.shape
    
    def __len__(self):
        return 1
    
    def __error__(self):
        return self.error

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.intensities, self.outputs
    

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