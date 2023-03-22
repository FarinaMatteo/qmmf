# Quantum Multi Model Fitting
Official repository for the paper "Quantum Multi-Model Fitting", published at CVPR 2023. 
Authors: Matteo Farina, [Luca Magri](https://scholar.google.com/citations?user=dF3qKXMAAAAJ&hl=it&oi=ao), [Willi Menapace](https://scholar.google.com/citations?hl=it&user=31ha1LgAAAAJ), [Elisa Ricci](https://scholar.google.com/citations?hl=it&user=xf1T870AAAAJ), [Vladislav Golyanik](https://scholar.google.com/citations?hl=it&user=we9LnVcAAAAJ) and [Federica Arrigoni](https://scholar.google.com/citations?hl=it&user=bzBtqfQAAAAJ).  

## References  
If you use this code, or you find some of this repository helpful, please cite:
```
@inproceedings{Farina2023quantum, 
title={Quantum Multi-Model Fitting}, 
author={Matteo Farina and Luca Magri and Willi Menapace and Elisa Ricci and Vladislav Golyanik and Federica Arrigoni}, 
booktitle={Computer Vision and Pattern Recognition (CVPR)}, 
year={2023} 
}
```

## Installation
This code has been tested under Ubuntu 18.04, 20.04 and 22.04 through a Miniconda virtual environment.
To install all the dependencies to run the demo script ```demo.py``` with a CPU, run:  
```
conda env create -f deps/environment.yml
```
You can then activate your virtual environment with ```conda activate qmmf```.  
The virtual environment comes with the DWave Ocean Software Development Kit. In order to use a DWave AQC, 
please follow the setup instructions at [this link](https://docs.ocean.dwavesys.com/en/stable/overview/install.html) and [this other link](https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html).

## Content  
This repo contains the source code for **QuMF** and **DeQuMF**, along with their CPU counterparts **QuMF (SA)** and **DeQuMF (SA)**, as described in the paper. You can find them in ```problems/disjoint_set_cover.py```. Have a look at the main function in ```demo.py``` to instantiate each of these algorithms.  

The demo script ```demo.py``` runs a qualitative and quantitative demo of **DeQuMF (SA)** on the AdelaideRMF dataset for fundamental matrix estimation.
Qualitative results are by default stored in the ```demo_output``` folder.  

**IMPORTANT NOTE**: Commented code for **QuMF** is given in order to show how to use this algorithm, too. However, the user should bear in mind the considerations reported in Section 5 of the paper: given the current state of Adiabatic Quantum Computers, **QuMF** cannot support large problems as the ones resulting from sampling the AdelaideRMF dataset. Your ```demo.py``` run will likely crash if you select QuMF as it is impossible to perform minor embedding.

## Reproducibility
The datasets we used for our paper are linked here:  
1. AdelaideRMF for Fundamental Matrix estimation: [link](https://osf.io/gb5yc/);
2. *Traffic2* and *Traffic3* subsets of the Hopkins benchmark: [link](http://www.vision.jhu.edu/data/hopkins155/);
3. York Urban Line Segment Databse: [link](https://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/).  

Details on the parameter settings can be found in the Supplementary Material of our paper, containing information on the number of sampled models and the used inlier thresholds, too.  

## Acknowledgements  
This paper is supported by FAIR (Future Artificial Intelligence Research) project, funded by the NextGenerationEU program within the PNRR-PE-AI scheme (M4C2, Investment 1.3, Line on Artificial Intelligence). This work was partially supported by the PRIN project LEGO-AI (Prot. 2020TA3K9N) and it was carried out in the Vision and Learning joint laboratory of FBK and UNITN.
