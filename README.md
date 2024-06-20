# Transfer Learning based on atomic feature extraction for the prediction of experimental <sup>13</sup>C chemical shifts


[![DOI](https://zenodo.org/badge/813572681.svg)](https://zenodo.org/doi/10.5281/zenodo.12087119)



## Introduction
Source code for the preprint: "Transfer Learning based on atomic feature extraction for the prediction of experimental <sup>13</sup>C chemical shifts"
Author of the code: Žarko Ivković, EMTCCM Master Student at the University of Barcelona


## Organization of the code
* `main.py` Lightning CLI to the models.

There are three main functions: `fit`, `test`, and `predict`. For more information about each function use `main.py COMMAND -h`.
* `predict.py` Interface for prediction using MACE & Uni-Mol GNN Ensemble. Usage: `predict.py -sdf PATH_TO_SDF_FOR_PREDICTION`
* `models/` the models' configuration files and weights
* `modules/` The main code for all models
* `preprocessing/` Encoders that extract atomic features using pre-trained models
* `low_data/` Results and model weights related to low data regimes
* `raw_data/` train and test datasets in form of SDF files, along with CASCADE datasets
* `visual/` module that contains visualization utilities
