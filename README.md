# NILUT: Conditional Neural Implicit 3D Lookup Tables for Image Enhancement

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()

[Marcos V. Conde](https://scholar.google.com/citations?user=NtB1kjYAAAAJ&hl=en), [Javier Vazquez-Corral](https://scholar.google.com/citations?user=gjnuPMoAAAAJ&hl=en), [Michael S. Brown](https://scholar.google.com/citations?hl=en&user=Gv1QGSMAAAAJ), [Radu Timofte](https://scholar.google.com/citations?user=u3MwH5kAAAAJ&hl=en)


**TL;DR** NILUT uses neural representations for controllable photorealistic image enhancement.

**Topics** Image Enhancement, Image Editing, Color Manipulation, Tone Mapping, Presets

***Website and repo in progress.*** **See also [AISP](https://github.com/mv-lab/AISP)** for image signal processing code and papers.


<br>

 <img src="nilut-intro.gif" alt="NILUT" width="800"> 

 <br>

----

3D lookup tables (3D LUTs) are a key component for image enhancement. Modern image signal processors (ISPs) have dedicated support for these as part of the camera rendering pipeline. Cameras typically provide multiple options for picture styles, where each style is usually obtained by applying a unique handcrafted 3D LUT. Current approaches for learning and applying 3D LUTs are notably fast, yet not so memory-efficient, as storing multiple 3D LUTs is required. For this reason and other implementation limitations, their use on mobile devices is less popular. 

In this work, we propose a Neural Implicit LUT (NILUT), an implicitly defined continuous 3D color transformation parameterized by a neural network. We show that NILUTs are capable of accurately emulating real 3D LUTs. Moreover, a NILUT can be extended to incorporate multiple styles into a single network with the ability to blend styles implicitly. Our novel approach is memory-efficient, controllable and can complement previous methods, including learned ISPs.

----

 **Contact** marcos.conde[at]uni-wuerzburg.de
