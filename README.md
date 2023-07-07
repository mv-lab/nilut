# [NILUT: Conditional Neural Implicit 3D Lookup Tables for Image Enhancement](https://arxiv.org/abs/2306.11920)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2306.11920)
[<a href=""><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab demo"></a>]()
[<a href="https://www.kaggle.com/code/jesucristo/super-resolution-demo-swin2sr-official/"><img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/Kaggle_logo.png?20140912155123" alt="kaggle demo" width=50></a>]()


[Marcos V. Conde](https://scholar.google.com/citations?user=NtB1kjYAAAAJ&hl=en), [Javier Vazquez-Corral](https://scholar.google.com/citations?user=gjnuPMoAAAAJ&hl=en), [Michael S. Brown](https://scholar.google.com/citations?hl=en&user=Gv1QGSMAAAAJ), [Radu Timofte](https://scholar.google.com/citations?user=u3MwH5kAAAAJ&hl=en)


**TL;DR** NILUT uses neural representations for controllable photorealistic image enhancement. 🚀 Demo Tutorial and pretrained models available.


<img src="media/nilut-intro.gif" alt="NILUT" width="800"> 

----

3D lookup tables (3D LUTs) are a key component for image enhancement. Modern image signal processors (ISPs) have dedicated support for these as part of the camera rendering pipeline. Cameras typically provide multiple options for picture styles, where each style is usually obtained by applying a unique handcrafted 3D LUT. Current approaches for learning and applying 3D LUTs are notably fast, yet not so memory-efficient, as storing multiple 3D LUTs is required. For this reason and other implementation limitations, their use on mobile devices is less popular. 

In this work, we propose a Neural Implicit LUT (NILUT), an implicitly defined continuous 3D color transformation parameterized by a neural network. We show that NILUTs are capable of accurately emulating real 3D LUTs. Moreover, a NILUT can be extended to incorporate multiple styles into a single network with the ability to blend styles implicitly. Our novel approach is memory-efficient, controllable and can complement previous methods, including learned ISPs.


**Topics** Image Enhancement, Image Editing, Color Manipulation, Tone Mapping, Presets

***Website and repo in progress.*** **See also [AISP](https://github.com/mv-lab/AISP)** for image signal processing code and papers.

----

**Pre-trained** sample models are available at `models/`. We provide `nilutx3style.pt` a NILUT that encodes three 3D LUT styles (1,3,4) with high accuracy.

**Demo Tutorial** in [nilut-multiblend.ipynb](nilut-multiblend.ipynb) we provide a simple tutorial on how to use NILUT for multi-style image enhancement and blending. The corresponding training code will be released soon.

**Dataset** The folder `dataset/` includes 100 images from the Adobe MIT 5K Dataset. The images were processed using professional 3D LUTs on Adobe Lightroom. The structure of the dataset is:

```
dataset/
├── 001_blend.png
├── 001_LUT01.png
├── 001_LUT02.png
├── 001_LUT03.png
├── 001_LUT04.png
├── 001_LUT05.png
├── 001_LUT08.png
├── 001_LUT10.png
└── 001.png
...
```

where `001.png` is the input unprocessed image, `001_LUTXX.png` is the result of applying each corresponding LUT and `001_blend.png` is the example target for evaluating sytle-blending (in the example the blending is between styles 1,3, and 4 with equal weights 0.33). 
The complete dataset includes 100 images `aaa.png` and their enhanced variants for each 3D LUT.


----

Hope you like it 🤗 If you find this interesting/insightful/inspirational or you use it, do not forget to acknowledge our work:

```
@article{conde2023nilut,
  title={NILUT: Conditional Neural Implicit 3D Lookup Tables for Image Enhancement},
  author={Conde, Marcos V and Vazquez-Corral, Javier and Brown, Michael S and Timofte, Radu},
  journal={arXiv preprint arXiv:2306.11920},
  year={2023}
}
```

**Contact** marcos.conde[at]uni-wuerzburg.de

