import numpy


class TwoLayerdNeuralNetwork:

    def __init__(self, inputnodes, outputnodes):
        self.inodes = inputnodes
        self.onodes = outputnodes

        self.w = numpy.zeros((self.inodes, self.onodes))
        self.t = numpy.zeros(self.onodes)

        self.activation_function = numpy.vectorize(lambda x: 0 if x < 0 else 1)
        pass

    def train(self, input_list, target_list):
        for idx, input in enumerate(input_list):
            for j in range(self.onodes):
                res = 0
                for i in range(self.inodes):
                    res += self.w[i][j] * input[i]
                    pass
                res = self.activation_function(res - self.t[j])
                for i in range(self.inodes):
                    self.w[i][j] += (target_list[idx][j] - res) * input[i]
                    pass
                self.t[j] -= (target_list[idx][j] - res)
                pass
        pass

    def query(self, input_vals):
        return self.activation_function(numpy.dot(input_vals, self.w) - self.t)


def simple_example():
    n = TwoLayerdNeuralNetwork(2, 1)

    input_values = [[0, 0], [1, 0], [0, 1]]
    target_values = [[0], [1], [1]]

    while True:
        n.train(input_values, target_values)
        res = list(map(n.query, input_values))
        if res == target_values:
            break

    print("fully trained!")
    res = list(map(n.query, input_values))
    for idx, v in enumerate(res):
        print("input:", input_values[idx], "result:", v)


def bigger_example():
    n = TwoLayerdNeuralNetwork(2, 3)

    input_values = [[-2, 1], [-2, -2], [-1, -1], [2, -1], [3, 2]]
    target_values = [[0, 1, 0], [0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 1]]

    while True:
        n.train(input_values, target_values)
        res = list(map(n.query, input_values))
        if (numpy.array(res) == numpy.array(target_values)).all():
            break

    print("fully trained!")
    res = list(map(n.query, input_values))
    for idx, v in enumerate(res):
        print("input:", input_values[idx], "result:", v)


simple_example()
#bigger_example()
