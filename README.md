# Supervised-Autoencoder-Joint-Learning-on-Heterogeneous-Tactile-Sensory-Data

This is the supplementary material for the paper Supervised Autoencoder Joint Learning on Heterogeneous Tactile Sensory Data: Improving Material Classification Performance, International Conference on Intelligent Robots and Systems (IROS) 2020.

## Requirements
To run the code, you can use the following command to install required packages in a seperate virtual environment.
```
conda create -n AE python=3.8.3
conda activate AE
pip install -r requirements.txt
```

## Datasets
In this experiment, we use data of two tactile sensors, iCub and BioTac. <br/>
`gh_download.sh` contains terminal commands to download the preprocessed data from a separate git repo [BioTac_slide_20_50](https://github.com/dexrob/BioTac_slide_20_50) <br/>
usage: open `gh_download`, change DIR to a preferred location, then run
```
chmod 755 gh_download.sh 
./gh_download.sh 
```

After saving the data, you can begin to train a model by simply running the following command under `code` folder
`python T1_BT19_Icub_joint_ae.py -k 0 -c 0 --data_dir /home/ruihan/data`. <br/>
More examples and explanations for parameters are given at the beginning of individual code script.

## Folders and files
* `code` contains all the code used to run the experiments and analyze the results.
    * `models_and_stats` is an empty folder to contained trained models and saved statistics
* `docs` contains supplementary information (empty for now).

## To Tas
To run the experiments T1_BT19_Icub_joint_ae.py (T1), T2_IcubCNN_ae.py (T2), and T4_BT19_ae.py (T4) multiple times, do the following: <br/>
(take T1 as an example)

* In T1 Line387, I have changed the dict_name to include a repetition index (pls help check for other experiments)
* Open T1_repeat.sh, change data_dir in Line8
* Open check_multiple_results.py, check log_dir, change base_name and csv_name for each experiment
* In a terminal, run 
```
chmod +x T1_repeat.sh
./T1_repeat.sh
```
(please comment out some output for model training in the python script if u want)