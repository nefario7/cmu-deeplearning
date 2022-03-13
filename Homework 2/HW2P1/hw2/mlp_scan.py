# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append("mytorch")
from loss import *
from activation import *
from linear import *
from conv import *


class CNN_SimpleScanningMLP:
    def __init__(self):
        ## Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(24, 8, 8, 4)
        self.conv2 = Conv1d(8, 16, 1, 1)
        self.conv3 = Conv1d(16, 4, 1, 1)
        self.relu = ReLU()
        self.flat = Flatten()
        self.layers = [self.conv1, self.relu, self.conv2, self.relu, self.conv3, self.flat]

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        self.conv1.conv1d_stride1.W = w1.reshape(8, 24, 8).transpose([2, 1, 0])  # 192 x 8
        self.conv2.conv1d_stride1.W = w2.reshape(1, 8, 16).transpose([2, 1, 0])  # 8 x 16
        self.conv3.conv1d_stride1.W = w3.reshape(1, 16, 4).transpose([2, 1, 0])  # 16 x 4

        # self.conv1.conv1d_stride1.W = w1.reshape(24, 8, 8).transpose([2, 0, 1])  # 192 x 8
        # self.conv2.conv1d_stride1.W = w2.reshape(8, 1, 16).transpose([2, 0, 1])  # 8 x 16
        # self.conv3.conv1d_stride1.W = w3.reshape(16, 1, 4).transpose([2, 0, 1])  # 16 x 4

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP:
    def __init__(self):
        ## Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(24, 2, 2, 2)
        self.conv2 = Conv1d(2, 8, 2, 2)
        self.conv3 = Conv1d(8, 4, 2, 1)
        self.relu = ReLU()
        self.flat = Flatten()
        self.layers = [self.conv1, self.relu, self.conv2, self.relu, self.conv3, self.flat]

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        self.conv1.conv1d_stride1.W = w1.T[:2, :].reshape(2, 8, 24)[:, :2, :].transpose([0, 2, 1])  # 192 x 8
        self.conv2.conv1d_stride1.W = w2.T[:8, :].reshape(8, 4, 2)[:, :2, :].transpose([0, 2, 1])  # 8 x 16
        self.conv3.conv1d_stride1.W = w3.T.reshape(4, 2, 8).transpose([0, 2, 1])  # 16 x 4

        # self.conv1.conv1d_stride1.W = np.random.randn(2, 24, 2)
        # self.conv2.conv1d_stride1.W = np.random.randn(8, 2, 2)
        # self.conv3.conv1d_stride1.W = np.random.randn(4, 8, 2)

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
