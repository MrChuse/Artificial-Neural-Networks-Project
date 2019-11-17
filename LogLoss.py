import numpy as np

class LogLoss:
    def forward(self, data, label):
        self.x = data.copy()
        self.label = label
        return -np.sum(label*np.log(np.clip(label * self.x, 1e-8, 1)))
        #return -np.sum(label*np.log(self.x) + (1 - label)*np.log(1 - self.x))
    
    def backward(self, epsilon=1e-10):
        return -(self.label / (self.label * self.x + epsilon))
        #return -(self.label / (self.x + epsilon) + (1 - self.label)/(1 - self.x + epsilon))