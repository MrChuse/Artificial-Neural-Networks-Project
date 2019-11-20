import numpy as np

class Conv:
    def __init__(self, kernel_size=(3, 3, 1), bias=True, padding=(0,0), stride=(1,1), learning_rate=0.01):
        self.kernel = np.random.standard_normal(kernel_size)
        self.padding = padding
        self.stride = stride
        if bias:
            self.bias = 0

    def multiply(self, i, j, data, kernel):
        part = data[i:i+kernel.shape[0], j:j+kernel.shape[1]]
        return np.inner(part.flatten(), kernel.flatten())

    def forward(self, data, bias=True):
        data = np.concatenate((np.zeros((self.padding[0], data.shape[1], data.shape[2])), #
                               data,
                               np.zeros((self.padding[0], data.shape[1], data.shape[2]))),#
                              axis=0)
        data = np.concatenate((np.zeros((data.shape[0], self.padding[1], data.shape[2])),# padding
                               data,
                               np.zeros((data.shape[0], self.padding[1], data.shape[2]))),#
                              axis=1)

        ans = []
        for index_answer, index_data_i in enumerate(range(0, data.shape[0] - self.kernel.shape[0] + 1, self.stride[0])):
            ans.append([])
            for index_data_j in range(0, data.shape[1] - self.kernel.shape[1] + 1, self.stride[1]):
                ans[index_answer].append(self.multiply(index_data_i, index_data_j, data, self.kernel)) #going from left to right, top to bottom, calculating sums
        return np.array(ans)
