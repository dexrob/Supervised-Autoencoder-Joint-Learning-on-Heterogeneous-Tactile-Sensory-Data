import numpy as np
import pickle
import csv
import glob

base_name = "BT19Icub_joint_ae_fold0"
log_dir = "/Users/ruihan/Documents/IROS2020/Supervised-Autoencoder-Joint-Learning-on-Heterogeneous-Tactile-Sensory-Data/code/models_and_stats/"
csv_name = log_dir+"T1_joint_results.csv"

with open(csv_name, 'w+', newline='') as f:
    fnames = ['test_acc_B', 'test_acc_I']
    writer = csv.DictWriter(f, fieldnames=fnames) 
    writer.writeheader()

    for name in glob.glob(log_dir+base_name+'_*[0-9]'+".pkl"): 
        print("extract results from " +name.split('/')[-1]) 
        results_dict = pickle.load(open(name, 'rb'))
        writer.writerow({'test_acc_B' : results_dict['acc'][0], 'test_acc_I': results_dict['acc'][0]})

