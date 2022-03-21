# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1:
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        self.batch = A.shape[0]
        self.input_size = A.shape[2]
        self.output_size = (self.input_size - self.kernel_size) + 1
        Z = np.zeros((self.batch, self.out_channels, self.output_size), dtype=A.dtype)
        W = np.repeat(self.W[np.newaxis, :, :, :], self.batch, axis=0)
        B = np.repeat(self.b[np.newaxis, :], self.batch, axis=0)
        for l in range(self.out_channels):
            for i in range(self.input_size - self.kernel_size + 1):
                # print(A[:, :, i : i + self.kernel_size].shape)
                # print(W[:, l, :, :].shape)
                Z[:, l, i] = np.sum(A[:, :, i : i + self.kernel_size] * W[:, l, :, :], axis=(1, 2)) + B[:, l]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        dZ_map = np.repeat(dLdZ[:, np.newaxis, :, :], self.in_channels, axis=1)

        self.dLdW = np.zeros_like(self.W, dtype=np.float)  # TODO
        for l in range(self.out_channels):
            for i in range(self.kernel_size):
                # print(self.A[:, :, i : i + self.kernel_size].shape)
                # print(dZ_map[:, :, l, :].shape)
                # print(self.dLdW[l, :, i].shape)
                self.dLdW[l, :, i] = np.sum(self.A[:, :, i : i + self.output_size] * dZ_map[:, :, l, :], axis=(0, 2))

        self.dLdb = np.sum(dLdZ, axis=(0, 2))  # TODO

        dLdA = np.zeros_like(self.A, dtype=np.float)  # TODO
        dZ_map_pad = np.pad(dZ_map, ((0, 0), (0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1)), mode="constant")
        W = np.flip(self.W, axis=(2))
        W = np.repeat(W[np.newaxis, :, :, :], self.batch, axis=0)
        for l in range(self.out_channels):
            for i in range(self.input_size):
                # print(W[:, l, :, :].shape)
                # print(dZ_map[:, :, l, i : i + self.kernel_size].shape)
                # print(dLdA[:, :, i].shape)
                dLdA[:, :, i] += np.sum(dZ_map_pad[:, :, l, i : i + self.kernel_size] * W[:, l, :, :], axis=(2))

        return dLdA


class Conv1d:
    def __init__(self, in_channels, out_channels, kernel_size, stride, weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample1d = Downsample1d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        Z1 = self.conv1d_stride1.forward(A)
        # downsample
        Z = self.downsample1d.forward(Z1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdA1 = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdA1)

        return dLdA


class Conv2d_stride1:
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        self.batch = A.shape[0]
        self.input_w, self.input_h = A.shape[2:]

        self.output_w = (self.input_w - self.kernel_size) + 1
        self.output_h = (self.input_h - self.kernel_size) + 1

        Z = np.zeros((self.batch, self.out_channels, self.output_w, self.output_h), dtype=A.dtype)
        W = np.repeat(self.W[np.newaxis, :, :, :, :], self.batch, axis=0)
        B = np.repeat(self.b[np.newaxis, :], self.batch, axis=0)
        for l in range(self.out_channels):
            for i in range(self.input_w - self.kernel_size + 1):
                for j in range(self.input_h - self.kernel_size + 1):
                    Z[:, l, i, j] = (
                        np.sum(A[:, :, i : i + self.kernel_size, j : j + self.kernel_size] * W[:, l, :, :, :], axis=(1, 2, 3)) + B[:, l]
                    )

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dZ_map = np.repeat(dLdZ[:, np.newaxis, :, :, :], self.in_channels, axis=1)

        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))  # TODO

        self.dLdW = np.zeros_like(self.W, dtype=np.float)  # TODO
        for l in range(self.out_channels):
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    self.dLdW[l, :, i, j] = np.sum(
                        self.A[:, :, i : i + self.output_w, j : j + self.output_h] * dZ_map[:, :, l, :, :], axis=(0, 2, 3)
                    )

        dLdA = np.zeros_like(self.A, dtype=np.float)  # TODO
        dZ_map_pad = np.pad(
            dZ_map,
            ((0, 0), (0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1), (self.kernel_size - 1, self.kernel_size - 1)),
            mode="constant",
        )
        W = np.flip(self.W, axis=(2, 3))
        W = np.repeat(W[np.newaxis, :, :, :, :], self.batch, axis=0)
        for l in range(self.out_channels):
            for i in range(self.input_w):
                for j in range(self.input_h):
                    dLdA[:, :, i, j] += np.sum(
                        dZ_map_pad[:, :, l, i : i + self.kernel_size, j : j + self.kernel_size] * W[:, l, :, :, :], axis=(2, 3)
                    )

        return dLdA


class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride, weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        Z1 = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(Z1)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample1d backward
        dLdA1 = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdA1)  # TODO

        return dLdA


class ConvTranspose1d:
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d isntance
        # TODO
        self.upsample1d = Upsample1d(self.upsampling_factor)  # TODO
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # TODO
        # upsample
        A_upsampled = self.upsample1d.forward(A)  # TODO

        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(A_upsampled)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # TODO

        # Call backward in the correct order
        delta_out = self.conv1d_stride1.backward(dLdZ)  # TODO

        dLdA = self.upsample1d.backward(delta_out)  # TODO

        return dLdA


class ConvTranspose2d:
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.upsample2d = Upsample2d(self.upsampling_factor)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample2d.forward(A)  # TODO

        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(A_upsampled)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call backward in correct order
        delta_out = self.conv2d_stride1.backward(dLdZ)  # TODO

        dLdA = self.upsample2d.backward(delta_out)  # TODO

        return dLdA


class Flatten:
    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.original_shape = A.shape
        Z = A.reshape(1, -1)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        dLdA = dLdZ.reshape(self.original_shape)  # TODO

        return dLdA
