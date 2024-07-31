# Variational Multiple-Instance Learning with Embedding Correlation Modeling for Hyperspectral Target Detection

This repository contains the code and resources for the paper "Variational Multiple-Instance Learning with Embedding Correlation Modeling for Hyperspectral Target Detection".

## Worklow
![Image](https://github.com/BoYangXDU/VMIL-ECM/blob/main/workflow.png)

## How to use
We propose a variational multiple-instance neural network with embedding correlation modeling (termed VMIL-ECM) for weakly supervised hyperspectral target detection, 
which relaxes the rigid target prior (e.g., target signatures and/or pixel-level annotations) and only region-level labels are required.

1. Get the required hyperspectral data and save it in the `data/` directory.
2. A specialist can box the area containing the target pixels as positive bags by visual observation or with reference to GPS coordinates, without considering the specific location of the targets within bags. The areas without any types of targets can be naturally divided into negative bags.
3. Modify the data path to ensure the program can correctly read the data and labels.
4. Run the `VMIL_run.py` script in the terminal, which will train the VMIL model.

## Related data
1. [Simulated data](https://github.com/GatorSense/Hyperspectral_Data_Simulation)
2. [MUULF Gulfport](https://github.com/GatorSense/MUUFLGulfport)
3. [Avon](https://www.rit.edu/dirs/spectir-hyperspectral-airborne-2012)

## License
This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

This product is Copyright (c) 2019 Bo Yang All rights reserved.

## Contact
For any questions or feedback, please reach out to the authors or open an issue in the repository.

Email: bond.yang@outlook.com (Bo Yang)

School of Artificial Intelligence, Xidian University
