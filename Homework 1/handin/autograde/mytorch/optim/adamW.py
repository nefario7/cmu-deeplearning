# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class AdamW:
    def __init__(self, model, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.l = model.layers
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.t = 0
        self.weight_decay = weight_decay

        self.m_W = [np.zeros(l.W.shape, dtype="f") for l in model.layers]
        self.v_W = [np.zeros(l.W.shape, dtype="f") for l in model.layers]

        self.m_b = [np.zeros(l.b.shape, dtype="f") for l in model.layers]
        self.v_b = [np.zeros(l.b.shape, dtype="f") for l in model.layers]

    def step(self):
        self.t += 1
        for layer_id, layer in enumerate(self.l):
            g_W = np.array(layer.dLdW, dtype=np.float)
            g_b = np.array(layer.dLdb, dtype=np.float)

            # TODO: Calculate updates for Weight
            self.m_W[layer_id] = self.beta1 * self.m_W[layer_id] + (1 - self.beta1) * g_W
            self.v_W[layer_id] = self.beta2 * self.v_W[layer_id] + (1 - self.beta2) * (g_W ** 2)

            # TODO: calculate updates for bias
            self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1 - self.beta1) * g_b
            self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1 - self.beta2) * (g_b ** 2)

            # # TODO: Perform weight and bias updates
            m_w = self.m_W[layer_id] / (1 - self.beta1 ** self.t)
            v_w = self.v_W[layer_id] / (1 - self.beta2 ** self.t)

            m_b = self.m_b[layer_id] / (1 - self.beta1 ** self.t)
            v_b = self.v_b[layer_id] / (1 - self.beta2 ** self.t)

            W_assign = layer.W - self.lr * m_w / np.sqrt(v_w + self.eps) - layer.W * self.lr * self.weight_decay
            b_assign = layer.b - self.lr * m_b / np.sqrt(v_b + self.eps) - layer.b * self.lr * self.weight_decay

            layer.W = W_assign
            layer.b = b_assign
            # print(layer.W)
