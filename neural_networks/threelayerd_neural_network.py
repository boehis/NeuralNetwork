import math

import numpy


class ThreeLayerdNeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.t = numpy.random.normal(0.0, pow(self.hnodes, -0.5), self.hnodes)

        self.lr = learningrate

        self.activation_function = lambda x: (1 / (1 + math.exp(-x)))
        self.activation_function_d = lambda x: (1 / (1 + math.exp(-x))) * (1 - (1 / (1 + math.exp(-x))))

    def train(self, input_val, target_val):
        hidden_out = []
        hidden_out_d = []
        for p in range(self.hnodes):
            hidden_in = 0
            for i in range(self.inodes):
                hidden_in += self.wih[p][i] * input_val[i]
            hidden_out.append(self.activation_function(hidden_in - self.t[p]))
            hidden_out_d.append(self.activation_function_d(hidden_in - self.t[p]))
        output_out = []
        for j in range(self.onodes):
            output_in = 0
            for p in range(self.hnodes):
                output_in += self.who[j][p] * hidden_out[p]
            output_out.append(self.activation_function(output_in))

        weighted_errors = []
        for p in range(self.hnodes):
            weighted_error_p = 0
            for j in range(self.onodes):
                weighted_error_p += (target_val[j] - output_out[j]) * self.who[j][p]
            weighted_errors.append(weighted_error_p)

        for j in range(self.onodes):
            for p in range(self.hnodes):
                self.who[j][p] += self.lr * (target_val[j] - output_out[j]) * hidden_out[p]
        for p in range(self.hnodes):
            for i in range(self.inodes):
                self.wih[p][i] += self.lr * weighted_errors[p] * hidden_out_d[p] * input_val[i]
            self.t[p] -= self.lr * weighted_errors[p] * hidden_out_d[p]

    def query(self, input_val):
        hidden_out = []
        for p in range(self.hnodes):
            hidden_in = 0
            for i in range(self.inodes):
                hidden_in += self.wih[p][i] * input_val[i]
            hidden_out.append(self.activation_function(hidden_in - self.t[p]))
        output_out = []
        for j in range(self.onodes):
            output_in = 0
            for p in range(self.hnodes):
                output_in += self.who[j][p] * hidden_out[p]
            output_out.append(self.activation_function(output_in))
        return output_out


