"""
NILUT: Conditional Neural Implicit 3D Lookup Tables for Image Enhancement
https://github.com/mv-lab/nilut

Generate hald images, a graphical representation of a 3D LUT in the form of a color table that contains all of the color gradations of the 3D LUT. 
Considering the input RGB space Hald, and the resultant one after applying a 3D LUT, we can use such pairs for training our models.
You can read more about this here: https://3dlutcreator.com/3d-lut-creator---materials-and-luts.html

"""

import numpy as np
import matplotlib.pyplot as plt
from utils import load_img, save_rgb

# Let's save into a list all the possible RGB intensities.

print ("Generating RGB values...")

rgb_map = []

for r in range(0, 256):
    for g in range(0, 256):
        for b in range(0, 256):
            rgb_map.append([r,g,b])

# Double-check that all the values are there.
print ("Number of unique RGB intensities", len(set([str(x) for x in rgb_map])))
rgb_map = np.array(rgb_map)
print ("Dimension of the RGB map", rgb_map.shape, 256**3, "sqrt (to reshape into a squared image)", np.sqrt(16777216))
rgb_img = np.reshape(rgb_map, (4096, 4096, 3))
print ("Reshape is lossless", np.all(rgb_img.reshape(16777216,3) == rgb_map))
print ("Color in [0,0]", rgb_img[0,0,:])
print ("Color in [4095, 4095]", rgb_img[4095, 4095,:])

# The image looks weird but it works!
plt.imshow(rgb_img)
save_rgb(rgb_img, "sample_hald.png")


# Now let's check that indeed all the colours are there. Naive seacrh just to make sure.

rgb_img = load_img("sample_hald.png", norm=False)
colors = []
for i in range(rgb_img.shape[0]):
    for j in range(rgb_img.shape[1]):
        colors.append(rgb_img[i,j])

print ("Number of colors / unique colors", len(colors)) #len(set([str(x) for x in colors])
print ("Done!")
