The SDOF code should be run in Mathematica version 14 or later. The results can be obtained by computing each module sequentially from the beginning.

The 3-DOF code should be trained in Mathematica version 14 or later and executed in MATLAB 2025 or later, with support for ONNX beta 24.2 required:

https://www.mathworks.com/matlabcentral/fileexchange/67296-deep-learning-toolbox-converter-for-onnx-model-format

Specifically, the data generation and nonparametric dynamic model training should first be carried out in Mathematica, after which the subsequent computations should be performed on the MATLAB platform. Please make sure that all files are kept in the same folder.
