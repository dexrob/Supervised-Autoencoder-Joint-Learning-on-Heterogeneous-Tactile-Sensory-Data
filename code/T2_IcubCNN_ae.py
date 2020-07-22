#!/usr/bin/env python
# coding: utf-8

'''
sample command: python T2_IcubCNN_ae.py -k 0 -c 2 -r 1
Individual training for iCub data (full/partial data)
if -r=1, train with full data
if -r=2, train with half data
loss = classification loss + recon loss 
'''

# Import
import os,sys
import pickle
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from vrae.vrae import VRAEC
from vrae.tas_utils_bs import get_trainValLoader, get_testLoader

# Parse argument
parser = argparse.ArgumentParser()
parser.add_argument("-k", "--kfold", type=int, default=0, help="kfold_number for loading data")
parser.add_argument("-r", "--reduction", type=int, default=1, help="data reduction ratio for partial training")
parser.add_argument("-c", "--cuda", default=0, help="index of cuda gpu to use")
args = parser.parse_args()

# dummy class to replace argparser, if running jupyter notebook
# class Args:
#   kfold = 0
#   cuda = '0'
#   reduction = 1

# args=Args()

# Set hyper params
kfold_number = args.kfold
data_reduction_ratio = args.reduction
shuffle = False # set to False for partial training
num_class = 20
sequence_length = 75
number_of_features = 60

hidden_size = 90
hidden_layer_depth = 1
latent_length = 40
batch_size = 32
learning_rate = 0.0005
n_epochs = 2000
dropout_rate = 0.2
cuda = True # options: True, False
header = "CNN"

# loss weightage
w_r = 0.01
w_c = 1

np.random.seed(1)
torch.manual_seed(1)

# Load data
data_dir = '../../new_data_folder/'
logDir = 'models_and_stats/'
if_plot = False

# model_name = 'IcubCNN_ae_{}_rm_{}_wrI_{}_wC_{}_{}'.format(data_reduction_ratio, removal, w_r, w_c, str(kfold_number))
model_name = "test_indiv_I"

if torch.cuda.is_available():
    device = torch.device("cuda:{}".format(args.cuda))
else:
    device = torch.device('cpu')

if args.reduction != 1:
    print("load {} kfold number, reduce data to {} folds, put to device: {}".format(args.kfold, args.reduction, device))
else:
    print("load {} kfold number, train with full data, put to devide: {}".format(args.kfold, device))

train_loader, val_loader, train_dataset, val_dataset = get_trainValLoader(data_dir, k=kfold_number, spike_ready=False, batch_size=batch_size, shuffle=shuffle)
test_loader, test_dataset = get_testLoader(data_dir, spike_ready=False, batch_size=batch_size, shuffle=shuffle)

# Initialize models
model = VRAEC(num_class=num_class,
            sequence_length=sequence_length,
            number_of_features = number_of_features,
            hidden_size = hidden_size, 
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate,
            cuda = cuda,
            model_name=model_name,
            header=header,
            device = device)
model.to(device)

# Initialize training settings
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
cl_loss_fn = nn.NLLLoss()
recon_loss_fn = nn.MSELoss()

training_start=datetime.now()
# create empty lists to fill stats later
epoch_train_loss = []
epoch_train_acc = []
epoch_val_loss = []
epoch_val_acc = []
max_val_acc = 0

for epoch in range(n_epochs):
    
    # TRAIN
    model.train()
    correct = 0
    train_loss = 0
    train_num = 0
    for i, (XI, XB,  y) in enumerate(train_loader):

        if model.header == 'CNN':
            x = XI
        else:
            x = XB
        x, y = x.to(device), y.long().to(device)
        if x.size()[0] != batch_size:
            break

        # reduce data by data_reduction_ratio times
        if i % data_reduction_ratio == 0:
            train_num += x.size(0)
            optimizer.zero_grad()
            x_decoded, latent, output = model(x)

            # construct loss function
            cl_loss = cl_loss_fn(output, y)
            recon_loss = recon_loss_fn(x_decoded, x)
            loss = w_c*cl_loss + w_r *recon_loss

            # compute classification acc
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()
            # accumulator
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

    # if epoch < 20 or epoch%200 == 0:
    #     print("train last batch: recon_loss {:.3f}".format(loss))

    # fill stats
    train_accuracy = correct / train_num
    train_loss /= train_num
    epoch_train_loss.append(train_loss)
    epoch_train_acc.append(train_accuracy) 
    
    # VALIDATION
    model.eval()
    correct = 0
    val_loss = 0
    val_num = 0
    for i, (XI, XB,  y) in enumerate(val_loader):
        if model.header == 'CNN':
            x = XI
        else:
            x = XB
        x, y = x.to(device), y.long().to(device)
        if x.size()[0] != batch_size:
            break
        val_num += x.size(0)
        x_decoded, latent, output = model(x)

        # construct loss function
        cl_loss = cl_loss_fn(output, y)
        recon_loss = recon_loss_fn(x_decoded, x)
        loss = w_c*cl_loss + w_r *recon_loss
    
        # compute classification acc
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()
        
        # accumulator
        val_loss += loss.item()

    # fill stats
    val_accuracy = correct / val_num# / len(val_loader.dataset)
    val_loss /= val_num #len(val_loader.dataset)
    epoch_val_loss.append(val_loss)  # only save the last batch
    epoch_val_acc.append(val_accuracy)

    if epoch < 20 or epoch%200 == 0:
        print("train_num {}, val_num {}".format(train_num, val_num))
        print('Epoch: {} Loss: train {:.3f}, valid {:.3f}. Accuracy: train: {:.3f}, valid {:.3f}'.format(epoch, train_loss, val_loss, train_accuracy, val_accuracy))

    # choose model
    if max_val_acc <= val_accuracy:
        model_dir = logDir + model_name + '.pt'
        print('Saving model at {} epoch to{}'.format(epoch, model_dir))
        max_val_acc = val_accuracy
        torch.save(model.state_dict(), model_dir)

training_end =  datetime.now()
training_time = training_end -training_start 
print("training takes time {}".format(training_time))

model.is_fitted = True
model.eval()

# TEST
correct = 0
test_num = 0
for i, (XI, XB,  y) in enumerate(test_loader):
    if model.header == 'CNN':
        x = XI
    else:
        x = XB
    x, y = x.to(device), y.long().to(device)
    
    if x.size(0) != batch_size:
        print(" test batch {} size {} < {}, skip".format(i, x.size()[0], batch_size))
        break
    test_num += x.size(0)
    x_decoded, latent, output = model(x)

    # compute classification acc
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    correct += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()
    
test_acc = correct / test_num #len(test_loader.dataset)
print('Test accuracy for', str(kfold_number), ' fold : ', test_acc)

# Save stats
results_dict = {"epoch_train_loss": epoch_train_loss,
             "epoch_train_acc": epoch_train_acc,
             "epoch_val_loss": epoch_val_loss,
             "epoch_val_acc": epoch_val_acc,
             "test_acc": test_acc}

dict_name = model_name + '_stats.pkl'
pickle.dump(results_dict, open(logDir + dict_name, 'wb'))
print("dump results dict to {}".format(dict_name))

assert n_epochs == len(epoch_train_acc), "different epoch length {} {}".format(n_epochs, len(epoch_train_acc))
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(np.arange(n_epochs), epoch_train_acc, label="train acc")
ax.set_xlabel('epoch')
ax.set_ylabel('acc')
ax.grid(True)
plt.legend(loc='upper right')
figname = logDir + model_name +"_train_acc.png"
if if_plot:
    plt.show()
