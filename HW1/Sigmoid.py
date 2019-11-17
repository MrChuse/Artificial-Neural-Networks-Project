import numpy as np

class Sigmoid:
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
        
    def forward(self, data, prnt=False):
        self.x = data.copy()
        t = self.sigmoid(self.x)
        if prnt:
            print(t, 'sigmoid')
        return t
    
    def backward(self, loss):
        return loss * (self.sigmoid(self.x) * (1 - self.sigmoid(self.x)))