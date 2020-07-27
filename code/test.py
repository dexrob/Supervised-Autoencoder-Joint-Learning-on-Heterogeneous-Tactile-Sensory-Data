import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--rep", type=int, default=0, help='index of running repetition')
args = parser.parse_args()

results ={"acc": [3*args.rep, 5*args.rep]}
logDir = "/Users/ruihan/Documents/IROS2020/Supervised-Autoencoder-Joint-Learning-on-Heterogeneous-Tactile-Sensory-Data/"
dict_name = "test_results_{}.pkl".format(args.rep)
pickle.dump(results, open(logDir + dict_name, 'wb'))


