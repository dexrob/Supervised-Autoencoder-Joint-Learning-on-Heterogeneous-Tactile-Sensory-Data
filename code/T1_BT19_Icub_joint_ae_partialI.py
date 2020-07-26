#!/usr/bin/env python
# coding: utf-8

'''
sample command: python T1_BT19_Icub_joint_ae_partialI.py -k 0 -c 0 -r 1
Joint training (partial data)
loss = classification loss + recon loss + mse loss
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

# # dummy class to replace argparser
# class Args:
#   kfold = 0
#   reduction = 1
#   cuda = '0'

# args=Args()

# Set hyper params
args_data_dir = args.data_dir
kfold_number = args.kfold
data_reduction_ratio = args.reduction
shuffle = False # set to False for partial training
num_class = 20
sequence_length_B = 400
sequence_length_I = 75
number_of_features_B = 19
number_of_features_I = 60

hidden_size = 90
hidden_layer_depth = 1
latent_length = 40
batch_size = 32
learning_rate = 0.0005
n_epochs = 2000

dropout_rate = 0.2
cuda = True # options: True, False
print_every=30
clip = True # options: True, False
max_grad_norm=5
header_B = None
header_I = "CNN"

# loss weightage
w_mse = 1 # mse between latent vectors
w_rB = 0.01 # recon for B
w_rI = 0.01 # recon for I
w_cB = 1 # classify for B
w_cI = 1 # classify for I

np.random.seed(1)
torch.manual_seed(1)

# Load data
data_dir = os.path.join(args_data_dir, "compiled_data/")
logDir = 'models_and_stats/'
if_plot = False

# new model
# model_name_B = 'BT19_joint_ae_wrB_{}_wcB_{}_wrI_{}_wcI_{}_wC_{}_reductI_{}_{}'.format(w_rB,w_cB, w_rI, w_cI, w_mse, data_reduction_ratio, str(kfold_number))
# model_name_I = 'IcubCNN_joint_ae_wrB_{}_wcB_{}_wrI_{}_wcI_{}_wC_{}_reductI_{}_{}'.format(w_rB,w_cB, w_rI, w_cI, w_mse, data_reduction_ratio, str(kfold_number))
model_name_B = "test_B_partialI"
model_name_I = "test_I_partialI"

if torch.cuda.is_available():
    device = torch.device("cuda:{}".format(args.cuda))
else:
    device = torch.device('cpu')

if args.reduction != 1:
    print("load {} kfold number, reduce data to {} folds, put to device: {}".format(args.kfold, args.reduction, device))
else:
    print("load {} kfold number, train with full data, put to device: {}".format(args.kfold, device))

train_loader, val_loader, train_dataset, val_dataset = get_trainValLoader(data_dir, k=kfold_number, spike_ready=False, batch_size=batch_size, shuffle=shuffle)
test_loader, test_dataset = get_testLoader(data_dir, spike_ready=False, batch_size=batch_size, shuffle=shuffle)

# Initialize models
model_B = VRAEC(num_class=num_class,
            sequence_length=sequence_length_B,
            number_of_features = number_of_features_B,
            hidden_size = hidden_size, 
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate, 
            cuda = cuda,
            model_name=model_name_B,
            header=header_B,
            device = device)
model_B.to(device)

model_I = VRAEC(num_class=num_class,
            sequence_length=sequence_length_I,
            number_of_features = number_of_features_I,
            hidden_size = hidden_size, 
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate, 
            cuda = cuda,
            model_name=model_name_I,
            header=header_I,
            device = device)
model_I.to(device)

# Initialize training settings
optimB = optim.Adam(model_B.parameters(), lr=learning_rate)
optimI = optim.Adam(model_I.parameters(), lr=learning_rate)
cl_loss_fn = nn.NLLLoss()
recon_loss_fn = nn.MSELoss()

# one stage training: with recon_loss and mse_loss
training_start=datetime.now()

# create empty lists to fill stats later
epoch_train_loss_B = []
epoch_train_acc_B = []
epoch_val_loss_B = []
epoch_val_acc_B = []
max_val_acc_B = 0

epoch_train_loss_I = []
epoch_train_acc_I = []
epoch_val_loss_I = []
epoch_val_acc_I = []
max_val_acc_I = 0

epoch_train_loss_C = []
epoch_val_loss_C = []
epoch_train_tot_loss = []
epoch_val_tot_loss = []

for epoch in range(n_epochs):

    # TRAIN
    model_B.train()
    model_I.train()

    correct_B = 0
    train_loss_B = 0
    correct_I = 0
    train_loss_I = 0
    train_loss_C = 0
    train_loss_tot = 0
    train_num_B = 0
    train_num_I = 0
            
    for i, (XI, XB,  y) in enumerate(train_loader):
        XI, XB, y = XI.to(device), XB.to(device), y.long().to(device)
        
        if XI.size()[0] != batch_size:
            break
    
        train_num_B += XB.size(0)
        # train modelB
        optimB.zero_grad()  
        x_decoded_B, latent_B, output = model_B(XB)

        # construct loss function
        recon_loss_B = recon_loss_fn(x_decoded_B, XB)
        cl_loss_B = cl_loss_fn(output, y)
        loss_B = w_rB*recon_loss_B + w_cB*cl_loss_B

        # compute classification acc
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct_B += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()
        # accumulator
        train_loss_B += loss_B.item()

        
        # train partially on I
        if i % data_reduction_ratio == 0:

            train_num_I += XI.size(0)

            # train model_I
            optimI.zero_grad()  
            x_decoded_I, latent_I, output = model_I(XI)

            # construct loss function
            recon_loss_I = recon_loss_fn(x_decoded_I, XI)
            cl_loss_I = cl_loss_fn(output, y)
            loss_I = w_rB*recon_loss_I + w_cI*cl_loss_I

            # compute classification acc
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct_I += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()
            # accumulator
            train_loss_I += loss_I.item()
        
            loss_C = w_mse*F.mse_loss(latent_B, latent_I)
            loss = loss_B + loss_I + loss_C
            # accumulator
            train_loss_C += loss_C.item()
            train_loss_tot += loss.item()

            if epoch < 20:
                loss_B.backward()
                loss_I.backward()
            else:
                loss.backward()

            optimB.step() 
            optimI.step() 
        
        else:
            # only train model B
            loss = loss_B
            loss.backward()
            optimB.step()
            
        train_loss_tot += loss.item()
        
    # if epoch < 20 or epoch%200 == 0:
    #     print("last batch training: LB: {:.2f}, LI: {:.2f}, LC: {:.2f} \n recon_B {:.2f}, cl_B {:.2f}, recon_I {:.2f}, cl_I {:.2f}"              .format(loss_B, loss_I, loss_C, recon_loss_B, cl_loss_B, recon_loss_I, cl_loss_I))
    
    # fill stats
    train_accuracy_B = correct_B / train_num_B
    train_loss_B /= train_num_B
    epoch_train_loss_B.append(train_loss_B)
    epoch_train_acc_B.append(train_accuracy_B) 
    
    train_accuracy_I = correct_I / train_num_I 
    train_loss_I /= train_num_I
    epoch_train_loss_I.append(train_loss_I)
    epoch_train_acc_I.append(train_accuracy_I) 

    train_loss_C /= train_num_I
    epoch_train_loss_C.append(train_loss_C)
    

    # VALIDATION
    model_B.eval()
    model_I.eval()

    correct_B = 0
    val_loss_B = 0
    correct_I = 0
    val_loss_I = 0
    val_loss_C = 0
    val_loss_tot = 0
    val_num = 0

    for i, (XI, XB,  y) in enumerate(val_loader):
        XI, XB, y = XI.to(device), XB.to(device), y.long().to(device)
        
        if XI.size()[0] != batch_size:
            break

        val_num += XI.size(0)
        
        # eval model_B
        x_decoded_B, latent_B, output = model_B(XB)
        # construct loss function
        recon_loss_B = recon_loss_fn(x_decoded_B, XB)
        cl_loss_B = cl_loss_fn(output, y)
        loss_B = w_rB*recon_loss_B + w_cB*cl_loss_B
       
        # compute classification acc
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct_B += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()
        # accumulator
        val_loss_B += loss_B.item()
        

        # eval modelI 
        x_decoded_I, latent_I, output = model_I(XI)
        # construct loss function
        recon_loss_I = recon_loss_fn(x_decoded_I, XI)
        cl_loss_I = cl_loss_fn(output, y)
        loss_I = w_rI*recon_loss_I + w_cI*cl_loss_I

        # compute classification acc
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct_I += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()
        # accumulator
        val_loss_I += loss_I.item()
        
        loss_C = w_mse*F.mse_loss(latent_B, latent_I)
        loss = loss_B + loss_I + loss_C

        # accumulator
        val_loss_C += loss_C.item()
        val_loss_tot += loss.item()

    # fill stats
    val_accuracy_B = correct_B / val_num
    val_loss_B /= val_num
    epoch_val_loss_B.append(val_loss_B)
    epoch_val_acc_B.append(val_accuracy_B) 
    
    val_accuracy_I = correct_I / val_num
    val_loss_I /= val_num
    epoch_val_loss_I.append(val_loss_I)
    epoch_val_acc_I.append(val_accuracy_I) 

    val_loss_C /= val_num
    epoch_val_loss_C.append(val_loss_C)
    
    if epoch < 20 or epoch%200 == 0:
        print("Epoch {}: Loss: lc {:.3f},  train_B {:.3f}, val_B {:.3f}, train_I {:.3f}, val_I {:.3f}, \n\t\t Acc: train_B {:.3f}, val_B {:.3f}, train_I {:.3f}, val_I {:.3f}"              .format(epoch, loss_C, train_loss_B, val_loss_B, train_loss_I, val_loss_I, train_accuracy_B, val_accuracy_B, train_accuracy_I, val_accuracy_I))
        print("-"*20)

    # choose model
    # TODO: not save at the same time, may have bad common representation
    if max_val_acc_B <= val_accuracy_B:
        model_dir = logDir + model_name_B + '.pt'
        print("Saving model at {} epoch to {}".format(epoch, model_dir))
        max_val_acc_B = val_accuracy_B
        torch.save(model_B.state_dict(), model_dir)

    if max_val_acc_I <= val_accuracy_I:
        model_dir = logDir + model_name_I + '.pt'
        print("Saving model at {} epoch to {}".format(epoch, model_dir))
        max_val_acc_I = val_accuracy_I
        torch.save(model_I.state_dict(), model_dir)

training_end =  datetime.now()
training_time = training_end -training_start 
print("RAE training takes time {}".format(training_time)) 

model_B.is_fitted = True
model_I.is_fitted = True

model_B.eval()
model_I.eval()

# TEST
correct_B = 0
correct_I = 0
test_num = 0

for i, (XI, XB,  y) in enumerate(test_loader):
    XI, XB, y = XI.to(device), XB.to(device), y.long().to(device)

    if XI.size()[0] != batch_size:
        break

    test_num += XI.size(0)

    # test model_B
    x_decoded_B, latent_B, output = model_B(XB)

    # compute classification acc
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    correct_B += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()

    
    # test modelI 
    x_decoded_I, latent_I, output = model_I(XI)

    # compute classification acc
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    correct_I += pred.eq(y.data.view_as(pred)).long().cpu().sum().item()

test_acc_B = correct_B/test_num
test_acc_I = correct_I/test_num
print('Test accuracy for {} fold {} samples: B {}, I {}'.format(str(kfold_number),test_num, test_acc_B, test_acc_I))

# Save stats
results_dict = {"epoch_train_loss_B": epoch_train_loss_B,
                "epoch_train_loss_I": epoch_train_loss_I,
                "epoch_train_loss_C": epoch_train_loss_C,
                "epoch_val_loss_B": epoch_val_loss_B,
                "epoch_val_loss_I": epoch_val_loss_I,
                "epoch_val_loss_C": epoch_val_loss_C,
                "epoch_train_acc_B": epoch_train_acc_B,
                "epoch_train_acc_I": epoch_train_acc_I,
                "epoch_val_acc_B": epoch_val_acc_B,
                "epoch_val_acc_I": epoch_val_acc_I,
                "test_acc": [test_acc_B, test_acc_I]}
dict_name = "BT19Icub_joint_ae_partialI_{}_fold{}.pkl".format(data_reduction_ratio, str(kfold_number))
pickle.dump(results_dict, open(logDir + dict_name, 'wb'))
print("dump results dict to {}".format(dict_name))

# Plot training acc curve
assert n_epochs == len(epoch_train_acc_B), "different epoch length {} {}".format(n_epochs, len(epoch_train_acc_B))
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(np.arange(n_epochs), epoch_train_acc_B, label="train acc B")
ax.set_xlabel('epoch')
ax.set_ylabel('acc')
ax.grid(True)
plt.legend(loc='upper right')
figname = logDir + model_name_B+"_train_acc.png"
plt.savefig(figname)
if if_plot:
    plt.show()

assert n_epochs == len(epoch_train_acc_I), "different epoch length {} {}".format(n_epochs, len(epoch_train_acc_I))
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(np.arange(n_epochs), epoch_train_acc_I, label="train acc I")
ax.set_xlabel('epoch')
ax.set_ylabel('acc')
ax.grid(True)
plt.legend(loc='upper right')
figname = logDir + model_name_I + "_train_acc.png"
plt.savefig(figname)
if if_plot:
    plt.show()
