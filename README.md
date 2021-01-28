# ShenjingCat: An SNN framework with PyTorch
**Authors:** Jun Zhou, Zhanglu Yan

## Installation
The repository includes C++ and CUDA code that has to be compiled and installed before it can be used from Python, download the repository and run the following command to do so:

`python setup.py install`

## Documentation
Please refer to the [wiki](https://github.com/zhoujuncc1/shenjingcat/wiki)

## Training Workflow
CQ Trainig is trained in stages as the figure below. The details of training methods is in [learning_tricks.md](https://github.com/zhoujuncc1/shenjingcat/blob/master/docs/learning_tricks.md).


<img src="https://github.com/zhoujuncc1/shenjingcat/raw/master/workflow_whole.jpg" width=500>

## Publications:
Z. Yan, J. Zhou, and W.F. Wong, "Near Lossless Transfer Learning for Spiking Neural Networks." Accepted by Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21). Virtual conference. Feb 2021.

## Acknowledgement
This simulator is inspired by and makes use of the code from [slayerPytorch](https://github.com/bamsumit/slayerPytorch)

## License & Copyright
Copyright 2020 Jun Zhou, ShenjingCat is free software: you can redistribute it and/or modoify it under the terms of GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

ShenjingCat is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details http://www.gnu.org/licenses/.
