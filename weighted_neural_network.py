import numpy
import scipy.special


class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.t = numpy.random.normal(0.0, pow(self.hnodes, -0.5), self.hnodes)

        self.lr = learningrate

        self.activation_function = lambda x: scipy.special.expit(x)
        self.activation_function_d = lambda x: scipy.special.expit(x) * (1 - scipy.special.expit(x))
        pass

    def train(self, input_list, target_list):
        for idx, input in enumerate(input_list):
            target = target_list[idx]
            actual = self.query(input)

            for j in range(self.onodes):
                for p in range(self.hnodes):
                    hidden_in_p = 0
                    for i in range(self.inodes):
                        hidden_in_p += self.wih[p][i] * input[i]
                        pass
                    hidden_out_p = self.activation_function(hidden_in_p - self.t[p])

                    self.who[j][p] += self.lr * 2 * (target[j] - actual[j]) * hidden_out_p
                    pass
                pass

            for p in range(self.hnodes):
                sum1 = 0
                for j in range(self.onodes):
                    sum1 += (target[j] - actual[j]) * self.who[j][p]
                    pass

                sum2 = 0
                for i in range(self.inodes):
                    sum2 += self.wih[p][i] * input[i]
                    pass
                change = self.lr * 2 * sum1 * self.activation_function_d(sum2 - self.t[p])

                for i in range(self.inodes):
                    self.wih[p][i] += change * input[i]
                    pass

                self.t[p] -= change
                pass

            pass
        pass

    def query(self, inputs):
        hidden_out = []
        for p in range(self.hnodes):
            hidden_in = 0
            for i in range(self.inodes):
                hidden_in += self.wih[p][i] * inputs[i]
            hidden_out.append(self.activation_function(hidden_in - self.t[p]))

        output_out = []
        for j in range(self.onodes):
            output_in = 0
            for p in range(self.hnodes):
                output_in += self.who[j][p] * hidden_out[p]
            output_out.append(output_in)

        return output_out


def xor_test():
    n = NeuralNetwork(2, 2, 1, 0.2)

    input_values = [[0, 0], [1, 0], [0, 1], [1, 1]]
    target_values = [[0.01], [0.99], [0.99], [0.01]]

    epochs = 20
    for e in range(epochs):
        n.train(input_values * 1000, target_values * 1000)

    res = list(map(n.query, input_values))

    for r in res:
        print(r)
