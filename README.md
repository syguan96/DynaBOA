# DynaBOA



Code repositoty for the paper:

**Out-of-Domain Human Mesh Reconstruction via Dynamic Bilevel Online Adaptation**

Shanyan Guan, Jingwei Xu, Michelle Z. He, Yunbo Wang, Bingbing Ni, Xiaokang Yang

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

Download required file:

```
unzip xxx.tar.gz -o data
```



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

<img src="asseets/qualitative_res1.png" alt="qualitative results" style="zoom:50%;" />

## Todo

- [ ] DynaBOA for MPI-INF-3DHP and SURREAL
- [ ] DynaBOA for the internet data.



