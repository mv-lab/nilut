"""
NILUT: Conditional Neural Implicit 3D Lookup Tables for Image Enhancement
https://github.com/mv-lab/nilut

Fit a complete 3D LUT into a simple NN.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import gc
from collections import defaultdict
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_amp = True
scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
start_time = None


# Import NILUT utils
from utils import start_timer, end_timer_and_print, clean_mem
from utils import load_img, save_rgb, plot_all, pt_psnr, np_psnr, deltae_dist, count_parameters
from dataloader import LUTFitting, MIT5KData
from models.archs import SIREN, NILUT


class NILUT(nn.Module):
    """
    Simple residual coordinate-based neural network for fitting 3D LUTs
    Official code: https://github.com/mv-lab/nilut
    """
    def __init__(self, in_features=3, hidden_features=256, hidden_layers=3, out_features=3, res=True):
        super().__init__()
        
        self.res = res
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU())
        
        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.Tanh())
        
        self.net.append(nn.Linear(hidden_features, out_features))
        if not self.res:
            self.net.append(torch.nn.Sigmoid())
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, intensity):
        output = self.net(intensity)
        if self.res:
            output = output + intensity
            output = torch.clamp(output, 0.,1.)
        
        return output, intensity


def fit(lut_model, total_steps, model_input, ground_truth, img_size, opt, verbose=200):
    """
    Simple training loop.
    """

    start_timer()
    metrics  = defaultdict(list)
    print (f"\n** Start training for {total_steps} iterations\n")

    for step in range(total_steps):
        
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            model_output, _ = lut_model(model_input)
            # loss = torch.mean((model_output - ground_truth)**2)
            loss = torch.mean(torch.abs(model_output - ground_truth)) # more stable than L2
            _psnr = pt_psnr(ground_truth,model_output).item()
        
        metrics['mse'].append(loss.item())
        metrics['psnr'].append(_psnr)
        if (step % verbose)==0:
            print (f">> Step {step} , loss={loss}, psnr={_psnr}")

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()

    plt.plot(metrics['psnr'])
    plt.title("PSNR Evolution")
    plt.show()

    print ("\n**Evaluate and get performance metrics\n")
    eval(model_input, model_output, ground_truth, img_size)

    torch.save(lut_model.state_dict(), f"3dlut.pt")
    clean_mem()


def eval(model_input, model_output, ground_truth, img_size):
    """
    Get performance metrics PSNR and DeltaE for the RGB transformation.
    """

    original_inp = model_input.cpu().view(img_size[0],img_size[1],3).numpy().astype(np.float32)
    np_out       = model_output.cpu().view(img_size[0],img_size[1],3).detach().numpy().astype(np.float32)
    np_gt        = ground_truth.cpu().view(img_size[0],img_size[1],3).detach().numpy().astype(np.float32)
    np_diff      = np.abs(np_gt - np_out)
    
    psnr = np_psnr(np_gt, np_out)
    deltae = deltae_dist(np_gt, np_out)
    
    print(f"Final metrics >> PSNR={psnr}, DeltaE={deltae} --- min error {np.min(np_diff)}, max error {np.max(np_diff)}") 
    plot_all([original_inp, np_out, np_gt, np_diff*10], figsize=(16,8))

    save_rgb(original_inp, f"results/inp.png")
    save_rgb(np_out,       f"results/out.png")
    save_rgb(np_gt ,       f"results/gt.png")


def main(inp_path, out_path, total_steps, lut_size):
    """
    Fit a professional 3D LUT into a simple coordinate-based MLP.
    Complete tutorial at: https://github.com/mv-lab/nilut

    - inp_path: Input RGB map as a hald image
    - out_path: Enhanced RGB map as a hald image, after using the desired 3D LUT

    """

    torch.cuda.empty_cache()
    gc.collect()

    print (f"Start NILUT {lut_size} fitting with")
    print ("Input hald image:", inp_path)
    print ("Target hald image:", out_path)

    # Define the dataloader
    lut_images = LUTFitting(inp_path, out_path)
    dataloader = DataLoader(lut_images, batch_size=1, pin_memory=True, num_workers=0)
    img_size = lut_images.shape
    print ("\nDataloader ready", img_size)
    
    # Define the model
    lut_model = NILUT(in_features=3, out_features=3, hidden_features=lut_size[0], hidden_layers=lut_size[1])
    lut_model.cuda()
    opt = torch.optim.Adam(lr=1e-3, params=lut_model.parameters())

    print (f"\nCreated NILUT model {lut_size} -- params={count_parameters(lut_model)}")
    
    # Load in memory the input and target hald images
    model_input_cpu, ground_truth_cpu = next(iter(dataloader))
    model_input, ground_truth = model_input_cpu.cuda(), ground_truth_cpu.cuda()
    print ("Input/Output shapes", model_input.shape, ground_truth.shape)

    lut_model.train()

    fit(lut_model, total_steps, model_input, ground_truth, img_size, opt)


parser = argparse.ArgumentParser(description='NILUT fitting')
parser.add_argument("--input", help="Input RGB map as a hald image", default="", type=str)
parser.add_argument("--target", help="Enhanced RGB map as a hald image, after using the desired 3D LUT", default="", type=str)
parser.add_argument("--steps", help="Number of optimizaation steps", default=1000, type=int)
parser.add_argument("--units", help="NILUT MLP architecture: number of neurons", default=128, type=int)
parser.add_argument("--layers", help="NILUT MLP architecture: number of layers", default=2, type=int)


if __name__ == "__main__":

    args = parser.parse_args()

    main(inp_path=args.input, 
         out_path=args.target,
         total_steps=args.steps,
         lut_size=(args.units, args.layers))
