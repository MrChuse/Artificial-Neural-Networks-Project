import numpy as np

class Dense:
    x = None
    
    def __init__(self, num1, num2, learning_rate=0.01, bias=True):
        self.mt = 2/num1 * np.random.randn(num1, num2) # new weights according to Andrew Ng
        self.bias = 2/num1 * np.random.randn(1, num2)
        self.lr = learning_rate

    def forward(self, data, bias=True, prnt=False):
        self.x = data.copy()
        t = np.matmul(self.x, self.mt) + (self.bias if bias else 0)
        if prnt:
            print(t, 'dense')
        return t

    def backward(self, loss, bias=True):
        #print(loss, 'dense bw')
        l = np.matmul(self.x.reshape(-1, 1), loss.reshape(1, -1))
        self.mt -= l * self.lr
        if bias:
            self.bias -= loss * self.lr
        return np.sum(l, axis=1)

if __name__ == '__main__':
    b = Dense(3, 5)
    x = b.forward(np.array([0, 0, 1]))
    print(x)
