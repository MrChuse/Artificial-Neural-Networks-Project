import numpy as np

class Softmax:
        
    def forward(self, x):
        self.x = np.exp(x)
        self.x = self.x/np.sum(self.x)
        return self.x
    
    def backward(self, loss):
        return loss * self.x * (1 - np.sum(self.x))