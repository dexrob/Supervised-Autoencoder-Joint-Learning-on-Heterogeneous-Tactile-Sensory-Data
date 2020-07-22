# Supervised-Autoencoder-Joint-Learning-on-Heterogeneous-Tactile-Sensory-Data

This is the supplementary material for the paper Supervised Autoencoder Joint Learning on Heterogeneous Tactile Sensory Data: Improving Material Classification Performance, International Conference on Intelligent Robots and Systems (IROS) 2020.

## Requirements
To run the code, you will need:
* python3
* pytorch v1.4.0
* torchvision (0.5.0)
* sklearn (0.22.1)
* pickle 
* numpy
You can use the following command to install required packages in a seperate virtual environment.
```
conda create -n AE python=3.8.3
conda activate AE
pip install -r requirements.txt
```

## Folders and files
* `code` contains all the code used to run the experiments and analyze the results.
* `data` contains the BioTac data collected during our experiments.
* `docs` contains supplementary informationn.
* `models_and_stats` is an empty folder to contained trained models and saved statistics
