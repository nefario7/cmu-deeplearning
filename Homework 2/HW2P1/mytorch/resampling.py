import numpy as np


class Upsample1d:
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        b, c, w = A.shape
        w_out = (w * self.upsampling_factor) - (self.upsampling_factor - 1)
        Z = np.zeros((b, c, w_out), dtype=A.dtype)  # TODO
        Z[:, :, :: self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        dLdA = dLdZ[:, :, :: self.upsampling_factor]  # TODO

        return dLdA


class Downsample1d:
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        Z = A[:, :, :: self.downsampling_factor]  # TODO
        self.original_width = A.shape[2]

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        b, c, w = dLdZ.shape
        dLdA = np.zeros((b, c, self.original_width), dtype=dLdZ.dtype)  # TODO
        dLdA[:, :, :: self.downsampling_factor] = dLdZ

        return dLdA


class Upsample2d:
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        b, c, w, h = A.shape
        w_out = (w * self.upsampling_factor) - (self.upsampling_factor - 1)
        h_out = (h * self.upsampling_factor) - (self.upsampling_factor - 1)
        Z = np.zeros((b, c, w_out, h_out), dtype=A.dtype)  # TODO
        Z[:, :, :: self.upsampling_factor, :: self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = dLdZ[:, :, :: self.upsampling_factor, :: self.upsampling_factor]  # TODO

        return dLdA


class Downsample2d:
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        Z = A[:, :, :: self.downsampling_factor, :: self.downsampling_factor]  # TODO
        self.original_shape = [A.shape[2], A.shape[3]]
        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        b, c, w, h = dLdZ.shape
        w_out = self.original_shape[0]
        h_out = self.original_shape[1]
        dLdA = np.zeros((b, c, w_out, h_out), dtype=dLdZ.dtype)  # TODO
        dLdA[:, :, :: self.downsampling_factor, :: self.downsampling_factor] = dLdZ

        return dLdA
