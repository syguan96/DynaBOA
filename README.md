# DynaBOA

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/out-of-domain-human-mesh-reconstruction-via/3d-human-pose-estimation-on-3dpw)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3dpw?p=out-of-domain-human-mesh-reconstruction-via)

Code repositoty for the paper:

**Out-of-Domain Human Mesh Reconstruction via Dynamic Bilevel Online Adaptation**

Shanyan Guan, Jingwei Xu, Michelle Z. He, Yunbo Wang, Bingbing Ni, Xiaokang Yang

[[Paper]](https://arxiv.org/abs/2111.04017) [[Project Page]](https://sites.google.com/view/dynaboa)

## Description
We focus on reconstructing human mesh from out-of-domain videos. In our experiments, we train a source model (termed as BaseModel) on Human 3.6M. To produce accurate human mesh on out-of-domain images, we optimize the BaseModel on target images via DynaBOA at test time. Below are the comparison results between BaseModel and the adapted model on the Internet videos with various camera parameters, motion, etc.


<p float="center">
  <img src="https://github.com/syguan96/ImageHost/blob/main/seq10_c01-v2-bigger.gif" width="99%" />
</p>
<p float="center">
  <img src="https://github.com/syguan96/ImageHost/blob/main/seq09_c01.gif" width="99%" />
</p>
<p float="center">
  <img src="https://github.com/syguan96/ImageHost/blob/main/seq07_c01.gif" width="49%" />
  <img src="https://github.com/syguan96/ImageHost/blob/main/seq02_c01.gif" width="49%" /> 
</p>


## Get Started

DynaBOA has been implemented and tested on Ubuntu 18.04 with python = 3.6.

Clone this repo:

```
git clone https://github.com/syguan96/DynaBOA.git
```

Install the requirements using `miniconda`:

```
conda env create -f dynaboa-env.yaml
```

Download required file from [this link](https://drive.google.com/file/d/1_4GhHaiNIu2aidVwMBvbdcdGd2vgy-gR/view?usp=sharing). Then unzip the file and rename it to `data` folder.



## Running on the 3DPW

```bash
bash run_on_3dpw.sh
```

#### Results on 3DPW

| Method                                                       | Protocol | PA-MPJPE |  MPJPE   |   PVE    |
| :----------------------------------------------------------- | :------: | :------: | :------: | :------: |
| [SPIN](https://github.com/nkolot/SPIN)                       |   #PS    |   59.2   |   96.9   |  135.1   |
| [PARE](https://github.com/mkocabas/PARE)                     |   #PS    |   46.4   |   79.1   |   94.2   |
| [Mesh Graphormer](https://github.com/microsoft/MeshGraphormer) |   #PS    |   45.6   |   74.7   |   87.7   |
| DynaBOA (Ours)                                               |   #PS    | **40.4** | **65.5** | **82.0** |

<img src="assets/qualitative_res1.png" alt="qualitative results" style="zoom:50%;" />

## Todo

- [ ] DynaBOA for MPI-INF-3DHP and SURREAL
- [ ] DynaBOA for the internet data.



