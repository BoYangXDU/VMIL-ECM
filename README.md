<div align="center">
  <h3 align="center"><strong>Variational Multiple-Instance Learning with Embedding Correlation Modeling <br> for Hyperspectral Target Detection  <br> [IEEE T-NNLS 2024] </strong></h3>
    <p align="center">
    <a>Bo Yang</a><sup>1</sup>&nbsp;&nbsp;
    <a>Changzhe Jiao</a><sup>1</sup>&nbsp;&nbsp;
    <a>Jinjian Wu</a><sup>1</sup>&nbsp;&nbsp;
    <a>Leida Li</a><sup>1</sup>&nbsp;&nbsp;
    <br>
    <sup>1</sup>School of Artificial Intelligence, Xidian University&nbsp;&nbsp;&nbsp;
</div>

This repository contains the code and resources for the paper [Variational Multiple-Instance Learning with Embedding Correlation Modeling for Hyperspectral Target Detection](https://ieeexplore.ieee.org/abstract/document/10803098) published in IEEE Transactions on Neural Networks and Learning Systems (T-NNLS).

## Workflow
![Image](https://github.com/BoYangXDU/VMIL-ECM/blob/main/workflow.png)

## How to use
We propose a variational multiple-instance neural network with embedding correlation modeling (termed VMIL-ECM) for weakly supervised hyperspectral target detection, 
which relaxes the rigid target prior (e.g., target signatures and/or pixel-level annotations) and only region-level labels are required.

1. Get the required hyperspectral data and save it in the `data/` directory.
2. A specialist can box the area containing the target pixels as positive bags by visual observation or with reference to GPS coordinates, without considering the specific location of the targets within bags. The areas without any types of targets can be naturally divided into negative bags.
3. Modify the data path to ensure the program can correctly read the data and labels.
4. Run the [`VMIL_run.py`](https://github.com/BoYangXDU/VMIL-ECM/blob/main/VMIL_run.py) script in the terminal, which will train the VMIL-ECM model.

## Related data
1. [Simulated data](https://github.com/GatorSense/Hyperspectral_Data_Simulation)
2. [MUULF Gulfport](https://github.com/GatorSense/MUUFLGulfport)
3. [Avon](https://www.rit.edu/dirs/spectir-hyperspectral-airborne-2012)

## Paper
Please cite our paper if you find our work and code useful for your research.

```
@ARTICLE{10803098,
  author={Yang, Bo and Jiao, Changzhe and Wu, Jinjian and Li, Leida},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Variational Multiple-Instance Learning With Embedding Correlation Modeling for Hyperspectral Target Detection}, 
  year={2024},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TNNLS.2024.3510385}}
```
```
@ARTICLE{10171432,
  author={Yang, Bo and He, Yi and Jiao, Changzhe and Pan, Xiao and Wang, Guozhen and Wang, Lei and Wu, Jinjian},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Multiple-Instance Metric Learning Network for Hyperspectral Target Detection}, 
  year={2023},
  volume={61},
  number={},
  pages={1-16},
  doi={10.1109/TGRS.2023.3291439}}
```


## Contact
For any questions or feedback, please reach out to the authors or open an issue in the repository.

Email: bond.yang@outlook.com (Bo Yang)

School of Artificial Intelligence, Xidian University

## Acknowledgement
Thanks to the inspirations and codes from [EM-MIL](https://github.com/airmachine/EM-MIL-WeaklyActionDetection).

