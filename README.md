# DLFramework

A basic framework for deep learning

## Features

- Colorful logger
- Yaml based configuration
- Extendable and flexible trainer
- Useful tools for visualization

## Concepts

**Model**: A model consists of network structure and hyper-parameters. Create a derived model by changing versions in yaml configuration.
Models with different structures should be placed in different folders under `models/`. Model weights(.pth file) should not be saved under this folder.

**Dataset**: The .py file of a derived pytorch dataset should be placed at `datasets/MyDatasetXXX`. 
Dataset configurations should be loaded from `GLOBAL_CONF['datasets']`. Data and caches should be placed at `data/MyDatasetXXX`.

**Analyzer**: The disentanglement between analyzer and model allows multiple combinations among dataset/model/analyzer.
A derived analyzer can be created by adding new key with modified parameters in `configs/analyzers`.
Saved model parameters should be placed under `saved_model/` folder.