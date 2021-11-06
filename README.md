# DynaBOA



Code repositoty for the paper:

**Out-of-Domain Human Mesh Reconstruction via Dynamic Bilevel Online Adaptation**

Shanyan Guan, Jingwei Xu, Michelle Z. He, Yunbo Wang, Bingbing Ni, Xiaokang Yang

[[Paper]](https://drive.google.com/file/d/1b6e3rMrVn_xNhM-MitqpLtulARdl4M9F/view?usp=sharing) [Project Page]

## Get Started

DynaBOA has been implemneted and tested on Ubuntu 18.04 with python = 3.6.

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



