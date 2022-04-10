import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        # Weights
        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        # Gradients
        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here
        self.r = 0
        self.z = 0
        self.n = 0

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h):
        return self.forward(x, h)

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx

        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh

        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx

        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        self.rx = self.Wrx @ x + self.brx
        self.rh = self.Wrh @ h + self.brh
        self.rt_prime = self.rx + self.rh
        rt = self.r_act(self.rt_prime)

        self.zx = self.Wzx @ x + self.bzx
        self.zh = self.Wzh @ h + self.bzh
        self.zt_prime = self.zx + self.zh
        zt = self.z_act(self.zt_prime)

        self.nx = self.Wnx @ x + self.bnx
        self.nh = self.Wnh @ h + self.bnh
        self.nt_prime = self.nx + rt * self.nh
        nt = self.h_act(self.nt_prime)

        h_t = (1 - zt) * nt + zt * h

        self.r = rt
        self.z = zt
        self.n = nt

        # This code should not take more than 10 lines.
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)  # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs match the
        # initalized shapes accordingly

        #! Bleh

        delta = delta.reshape(self.h, 1)
        self.x = self.x.reshape(self.d, 1)  # xt
        self.hidden = self.hidden.reshape(self.h, 1)  # ht-1
        self.r = self.r.reshape(self.h, 1)  # rt
        self.n = self.n.reshape(self.h, 1)  # nt
        self.z = self.z.reshape(self.h, 1)  # zt

        dz_t = delta * (-self.n + self.hidden)
        dn_t = delta * (1 - self.z)
        dr_t = dn_t * self.h_act.derivative(self.n) * (self.Wnh @ self.hidden + self.bnh.reshape(self.h, 1))
        dz = dz_t
        dn = dn_t
        dr = dr_t

        dn_t = dn_t * np.expand_dims(self.h_act.derivative(), 1)  #! or nt_prime
        self.dWnh += (dn_t * self.r) @ self.hidden.T
        self.dbnh += (dn_t * self.r).reshape(
            self.h,
        )
        self.dWnx += dn_t @ self.x.T
        self.dbnx += dn_t.reshape(
            self.h,
        )

        dz_t = dz_t * np.expand_dims(self.z_act.derivative(), 1)  #! or zt_prime
        self.dWzh += dz_t @ self.hidden.T
        self.dbzh += dz_t.reshape(
            self.h,
        )
        self.dWzx += dz_t @ self.x.T
        self.dbzx += dz_t.reshape(
            self.h,
        )

        dr_t = dr_t * np.expand_dims(self.r_act.derivative(), 1)  #! or rt_prime
        self.dWrh += dr_t @ self.hidden.T
        self.dbrh += dr_t.reshape(
            self.h,
        )
        self.dWrx += dr_t @ self.x.T
        self.dbrx += dr_t.reshape(
            self.h,
        )

        dz = dz_t
        dn = dn_t
        dr = dr_t

        dx = dn_t.T @ self.Wnx + dz_t.T @ self.Wzx + dr_t.T @ self.Wrx
        dh = (delta * self.z).T + (dn_t * self.r).T @ self.Wnh + dr_t.T @ self.Wrh + dz_t.T @ self.Wzh

        # This code should not take more than 25 lines.

        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
