# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Dropout2d(object):
    def __init__(self, p=0.5):
        # Dropout probability
        self.p = p

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, eval=False):
        """
        Arguments:
          x (np.array): (batch_size, in_channel, input_width, input_height)
          eval (boolean): whether the model is in evaluation mode
        Return:
          np.array of same shape as input x
        """
        # 1) Get and apply a per-channel mask generated from np.random.binomial
        # 2) Scale your output accordingly
        # 3) During test time, you should not apply any mask or scaling.
        if not eval:
            batch_size, in_channel, input_width, input_height = x.shape
            mask = np.random.binomial(1, self.p, size=in_channel)
            mask = mask[np.newaxis, :, np.newaxis, np.newaxis]
            self.mask = np.tile(mask, (batch_size, 1, input_width, input_height))
            print(self.mask)
            # self.mask = self.mask < self.p
            x = np.multiply(x, self.mask)
            x = x / (1 - self.p)

        return x

    def backward(self, delta):
        """
        Arguments:
          delta (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
          np.array of same shape as input delta
        """
        # 1) This method is only called during training.
        # 2) You should scale the result by chain rule

        return np.multiply(delta, self.mask)
