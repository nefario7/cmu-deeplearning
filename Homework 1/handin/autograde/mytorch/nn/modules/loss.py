import numpy as np

class MSELoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        error  = None # TODO
        L      = np.sum(error) / N
        
        return NotImplemented
    
    def backward(self):
    
        dLdA = None
        
        return NotImplemented

class CrossEntropyLoss:
    
    def forward(self, A, Y):
    
        self.A = A
        self.Y = Y
        N      = A.shape[0]
        C      = A.shape[1]
        Ones   = np.ones((C, 1), dtype="f")

        self.softmax = None # TODO
        crossentropy = None # TODO
        L = np.sum(crossentropy) / N
        
        return NotImplemented
    
    def backward(self):
    
        dLdA = None # TODO
        
        return NotImplemented
