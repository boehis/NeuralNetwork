import numpy


class TwoLayerdNeuralNetwork:

    def __init__(self, inputnodes, outputnodes):
        self.inodes = inputnodes
        self.onodes = outputnodes

        self.w = numpy.zeros((self.inodes, self.onodes))
        self.t = numpy.zeros(self.onodes)

        self.activation_function = numpy.vectorize(lambda x: 0 if x < 0 else 1)

    def train(self, input_list, target_list):
        for idx, input in enumerate(input_list):
            for j in range(self.onodes):
                res = 0
                for i in range(self.inodes):
                    res += self.w[i][j] * input[i]
                res = self.activation_function(res - self.t[j])
                for i in range(self.inodes):
                    self.w[i][j] += (target_list[idx][j] - res) * input[i]
                self.t[j] -= (target_list[idx][j] - res)

    def query(self, input_vals):
        return self.activation_function(numpy.dot(input_vals, self.w) - self.t)
