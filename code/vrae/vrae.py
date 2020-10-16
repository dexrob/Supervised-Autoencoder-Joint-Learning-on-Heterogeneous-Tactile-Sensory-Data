''' Code is modified based on https://github.com/tejaslodaya/timeseries-clustering-vae'''
import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import os, sys
import pickle


class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM

    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout

    """
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        
        self.lstm = nn.LSTM(input_size=self.number_of_features, 
                            hidden_size=self.hidden_size, 
                            num_layers=self.hidden_layer_depth, 
                            batch_first=True,dropout=dropout)

    def forward(self, x):
        """
        Forward propagation of encoder. Given input, outputs the last hidden state of encoder
        
        :param x: input to the encoder, of shape (batch_size, number_of_features, sequence_length)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        lstm: h_n of shape (num_layers, batch, hidden_size)
        
        """
        batch_size, num_features, sequence_size = x.size()
        
        # create embedding
        embed_seq = []
        for t in range(sequence_size):
            out = x[...,t]
            embed_seq.append(out)
        embed_seq = torch.stack(embed_seq, dim=0).transpose_(0, 1)
        
        # forward on LSTM
        self.lstm.flatten_parameters()
        r_out, (h_n, h_c) = self.lstm(embed_seq)
        h_end = h_n[-1, :, :]

        return h_end


class CNN(nn.Module):
    """
    CNN header network for iCub sensor

    :param C: number of channels of the taxel image
    :param H: height of the taxel image
    :param W: width of the taxel image
    :param cnn_number_of_features: number of CNN output features, also equivalent to number of input featurs to CNNEncoder

    """
    def __init__(self, C=1, H=6, W=10, cnn_number_of_features=18):
        super(CNN, self).__init__()
        self.cnn_number_of_features = cnn_number_of_features
        self.C = C
        self.H = H 
        self.W = W

        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=self.C, out_channels=3, kernel_size=(3, 5)),
            nn.MaxPool2d(2, return_indices=True),
        )

    def forward(self, x):
        """
        Forward propagation of CNN. Given input, outputs the CNN feature and mapping indices
        
        :param x: input to the CNN, of shape (batch_size, channel, height, width)
        :return cnn_out: cnn output feature
        :return mp_indices: mapping indices for convolution
        
        """
        # check input size 
        batch_size, C, H, W = x.size()
        assert C==self.C and H==self.H and W==self.W, "wrong size for CNN input, x {}, \
            should be (batch_size,{},{},{})".format(x.size(), self.C, self.H, self.W)
        cnn_out, mp_indices = self.seq(x)
        cnn_out = cnn_out.view(-1, self.cnn_number_of_features)
        return cnn_out, mp_indices


class CnnEncoder(nn.Module):
    """
    Encoder network containing enrolled LSTM

    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: length of the latent vector
    :param dropout: percentage of nodes to dropout

    """
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, cnn_number_of_features=None):

        super(CnnEncoder, self).__init__()
        # overwrite number_of_features, since data -> CNN -> LSTM
        if cnn_number_of_features is not None:
            self.number_of_features = cnn_number_of_features
        else:
            self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.cnn = CNN(cnn_number_of_features=cnn_number_of_features)
        self.lstm = nn.LSTM(input_size=self.number_of_features, 
                    hidden_size=self.hidden_size, 
                    num_layers=self.hidden_layer_depth, 
                    batch_first=True,dropout=dropout)


    def forward(self, x):
        """
        Forward propagation of encoder. Given input, outputs the last hidden state of encoder
        
        :param x: input to the encoder, of shape (sequence_length, batch_size, H, W, sequence_size)
        :return h_end: last hidden state of encoder, of shape (batch_size, hidden_size)
        :return mp_indices: keep mapping indices for reshaping later in decoder
        lstm: h_n of shape (num_layers, batch, hidden_size)
        
        """
        x = x.unsqueeze(1)
        batch_size, C, H, W, sequence_size = x.size()
        
        # create CNN embedding
        cnn_embed_seq = []
        for t in range(sequence_size):
            cnn_out, mp_indices = self.cnn(x[...,t])
            cnn_embed_seq.append(cnn_out)   
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        # forward on LSTM
        self.lstm.flatten_parameters()
        r_out, (h_n, h_c) = self.lstm(cnn_embed_seq)
        h_end = h_n[-1, :, :]

        return h_end, mp_indices


class Lambda(nn.Module):
    """
    Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: length of the latent vector

    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()
 
        self.hidden_size = hidden_size
        self.latent_length = latent_length
        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        nn.init.xavier_uniform_(self.hidden_to_mean.weight)

    def forward(self, cell_output):
        """
        Given last hidden state of encoder, passes through a linear layer, and finds its mean value

        :param cell_output: last hidden state of encoder
        :return: latent vector

        """

        return self.hidden_to_mean(cell_output)
            

class Decoder(nn.Module):
    """
    Converts latent vector into output

    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: length of the latent vector
    :param output_size: output size of the mean vector
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    :param device: Depending on cuda enabled/disabled
    """
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype, device):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype
        self.device = device

        
        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)
        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.hidden_layer_depth, batch_first=True)



    def forward(self, latent):
        """
        Converts latent to hidden to output

        :param latent: latent vector
        :return: output consisting of mean vector

        """

        # update the implementation of decoder in Oct 2020
        latent_input = self.latent_to_hidden(latent)
        decoder_input = latent_input.repeat(self.sequence_length, 1, 1).transpose_(0, 1) # [8, 400, 90]
        decoder_output, (h_n, c_n) = self.lstm(decoder_input)

        out = self.hidden_to_output(decoder_output)
        out = out.permute(0, 2, 1)
        return out


class CnnDecoder(nn.Module):
    """
    Converts latent vector into output

    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: output size 
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    :param device: Depending on cuda enabled/disabled
    :param cnn_number_of_feratures: number of features of cnn output, equivalent to lstm input
    """
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype, device, cnn_number_of_features=None):

        super(CnnDecoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length

        # manually specify the C, H, W, consistent with CNNEncoder
        self.C = 1
        self.H = 6
        self.W = 10

        if cnn_number_of_features is None:
            self.output_size = output_size
        else: 
            self.output_size = cnn_number_of_features
        self.device = device

        # mirror CNN
        self.unpool = nn.MaxUnpool2d(2)
        self.dcnn = nn.Sequential(
            nn.ConvTranspose2d(3, 1, kernel_size=(3, 5)),
            nn.ReLU()
        )

        
        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)
        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)
        
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.hidden_layer_depth, batch_first=True)



    def forward(self, latent, mp_indices):
        """
        Converts latent to hidden to output

        :param latent: latent vector, mp_indices to reverse maxpooling correctly
        :param mp_indices: mapping indices for reshaping in decoder 
        :return: output consisting of mean vector

        """

        latent_input = self.latent_to_hidden(latent)
        decoder_input = latent_input.repeat(self.sequence_length, 1, 1).transpose_(0, 1) # [8, 400, 90]
        decoder_output, (h_n, c_n) = self.lstm(decoder_input)

        
        out = self.hidden_to_output(decoder_output)
        batch_size, sequence_size, number_of_features = out.size() # [32, 75, 18]
        out = out.permute(0, 2, 1)
        # mirror CNN embedding
        dcnn_seq = []
        for t in range(sequence_size):
            x = out[..., t].view(batch_size,3,2,3)
            x = self.unpool(x, mp_indices) 
            x = self.dcnn(x)
            dcnn_seq.append(x)
            
        dcnn_seq = torch.stack(dcnn_seq, dim=0).transpose_(0, 1)
    
        dcnn_seq = dcnn_seq.reshape(batch_size, self.H, self.W, sequence_size)

        return dcnn_seq

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

class VRAEC(BaseEstimator, nn.Module):
    """
    Variational recurrent auto-encoder with classifier.
    This module is used for dimensionality reduction of timeseries and perform classification using hidden representation.

    :param num_class: number of class labels
    :param sequence_length: length of the input sequence
    :param number_of_features: number of input features
    :param hidden_size:  hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param batch_size: number of timeseries in a single batch
    :param learning_rate: the learning rate of the module
    :param n_epochs: Number of iterations/epochs
    :param dropout_rate: The probability of a node being dropped-out
    :param boolean cuda: to be run on GPU or not
    :param dload: Download directory where models are to be dumped. Currently saving model outside.
    :param model_name name of state dict to be stored under dload directory, without post
    :param header: "CNN" or "None", hearder implemented before encoder and after decoder
    """
    def __init__(self, num_class, sequence_length, number_of_features, hidden_size=90, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0.005, n_epochs=5, dropout_rate=0., cuda=False,
                 dload='.', model_name='model', header=None, device='cpu'):

        super(VRAEC, self).__init__()

        self.dtype = torch.FloatTensor
        self.ydtype = torch.LongTensor
        self.use_cuda = cuda
        self.header = header
        self.device = device
        self.epoch_train_acc = []
        
        if not torch.cuda.is_available() and self.use_cuda:
            self.use_cuda = False
        if self.use_cuda:
            self.dtype = torch.cuda.FloatTensor
            self.ydtype = torch.cuda.LongTensor

        if self.header is None:
            self.encoder = Encoder(number_of_features = number_of_features,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                dropout=dropout_rate)

            self.decoder = Decoder(sequence_length=sequence_length,
                                batch_size = batch_size,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                output_size=number_of_features,
                                dtype=self.dtype, device=self.device)
        
        elif self.header == "CNN":
            self.encoder = CnnEncoder(number_of_features = number_of_features,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                dropout=dropout_rate,
                                cnn_number_of_features=18)
            
            self.decoder = CnnDecoder(sequence_length=sequence_length,
                                batch_size = batch_size,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                output_size=number_of_features,
                                dtype=self.dtype,
                                device=self.device,
                                cnn_number_of_features=18)
        
        else:
            raise NotImplementedError
            

        self.lmbd = Lambda(hidden_size=hidden_size,
                           latent_length=latent_length)

        self.classifier = nn.Sequential(
            nn.Linear(latent_length, num_class),
            # nn.Dropout(0.2),
            nn.LogSoftmax(dim=1)
        )

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.is_fitted = False
        self.dload = dload
        self.model_name = model_name

        if self.use_cuda:
            self.cuda()

    def __repr__(self):
        return """VRAE(n_epochs={n_epochs},batch_size={batch_size},cuda={cuda})""".format(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                cuda=self.use_cuda)

    def forward(self, x):
        """
        Forward propagation which involves one pass from inputs to encoder to lambda to decoder

        :param x:input tensor
        :return: the decoded output, latent vector
        """
        if self.header is None:
            cell_output = self.encoder(x)
            latent = self.lmbd(cell_output)
            x_decoded = self.decoder(latent)
        elif self.header == "CNN":
            cell_output, mp_indices = self.encoder(x)
            latent = self.lmbd(cell_output)
            x_decoded = self.decoder(latent, mp_indices)
        else:
            raise NotImplementedError
        output = self.classifier(latent)

        return x_decoded, latent, output
