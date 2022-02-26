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

        print("A = ", A.shape)
        print("s = ", self.downsampling_factor)
        Z = A[:, :, :: self.downsampling_factor]  # TODO
        print("Z = ", Z.shape)

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        b, c, w = dLdZ.shape
        w_out = (w * self.downsampling_factor) - (self.downsampling_factor - 1)
        dLdA = np.zeros((b, c, w_out), dtype=dLdZ.dtype)  # TODO
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

        Z = None  # TODO

        return NotImplemented

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = None  # TODO

        return NotImplemented


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

        Z = None  # TODO

        return NotImplemented

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = None  # TODO

        return NotImplemented
