import numpy as np
from resampling import *


class MaxPool2d_stride1:
    def __init__(self, kernel):
        self.kernel_size = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        self.batch, self.in_channels, self.input_w, self.input_h = A.shape
        self.out_channels = self.in_channels

        self.output_w = (self.input_w - self.kernel_size) + 1
        self.output_h = (self.input_h - self.kernel_size) + 1

        Z = np.zeros((self.batch, self.out_channels, self.output_w, self.output_h))
        self.max_indexes = np.zeros(np.append(Z.shape, 2), dtype=np.int)

        for b in range(self.batch):
            for l in range(self.out_channels):
                for i in range(self.output_w):
                    for j in range(self.output_h):
                        window = A[b, l, i : i + self.kernel_size, j : j + self.kernel_size]
                        x = np.unravel_index(np.argmax(window), window.shape)[0] + i
                        y = np.unravel_index(np.argmax(window), window.shape)[1] + j
                        self.max_indexes[b, l, i, j, :] = [x, y]
                        Z[b, l, i, j] = A[b, l, x, y]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros_like(self.A, dtype=np.float)  # TODO
        for b in range(self.batch):
            for l in range(self.out_channels):
                for i in range(self.output_w):
                    for j in range(self.output_h):
                        x, y = self.max_indexes[b, l, i, j]
                        dLdA[b, l, x, y] += dLdZ[b, l, i, j]

        return dLdA


class MeanPool2d_stride1:
    def __init__(self, kernel):
        self.kernel_size = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        self.batch, self.in_channels, self.input_w, self.input_h = A.shape
        self.out_channels = self.in_channels

        self.W = np.ones((self.out_channels, self.kernel_size, self.kernel_size)) / self.kernel_size
        self.W = np.repeat(self.W[np.newaxis, :, :, :], self.batch, axis=0)

        self.output_w = (self.input_w - self.kernel_size) + 1
        self.output_h = (self.input_h - self.kernel_size) + 1

        Z = np.zeros((self.batch, self.out_channels, self.output_w, self.output_h))

        for b in range(self.batch):
            for l in range(self.out_channels):
                for i in range(self.input_w - self.kernel_size + 1):
                    for j in range(self.input_h - self.kernel_size + 1):
                        Z[b, l, i, j] = (
                            np.sum(A[b, l, i : i + self.kernel_size, j : j + self.kernel_size] * self.W[b, l, :, :]) / self.kernel_size
                        )

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros_like(self.A, dtype=np.float)  # TODO
        dZ_map_pad = np.pad(
            dLdZ,
            ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1), (self.kernel_size - 1, self.kernel_size - 1)),
            mode="constant",
        )

        for b in range(self.batch):
            for l in range(self.out_channels):
                for i in range(self.input_w):
                    for j in range(self.input_h):
                        dLdA[b, l, i, j] += (
                            np.sum(dZ_map_pad[b, l, i : i + self.kernel_size, j : j + self.kernel_size] * self.W[b, l, :, :])
                            / self.kernel_size
                        )

        return dLdA


class MaxPool2d:
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z1 = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA1 = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdA1)
        return dLdA


class MeanPool2d:
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z1 = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA1 = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdA1)
        return dLdA
