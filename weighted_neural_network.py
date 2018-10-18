import math

import numpy


class NeuralNetwork:

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
        pass

    def train(self, input, target):
        hidden_out = []
        hidden_out_d = []
        for p in range(self.hnodes):
            hidden_in = 0
            for i in range(self.inodes):
                hidden_in += self.wih[p][i] * input[i]
                pass
            hidden_out.append(self.activation_function(hidden_in - self.t[p]))
            hidden_out_d.append(self.activation_function_d(hidden_in - self.t[p]))
            pass
        output_out = []
        for j in range(self.onodes):
            output_in = 0
            for p in range(self.hnodes):
                output_in += self.who[j][p] * hidden_out[p]
                pass
            output_out.append(self.activation_function(output_in))
            pass

        weighted_errors = []
        for p in range(self.hnodes):
            weighted_error_p = 0
            for j in range(self.onodes):
                weighted_error_p += (target[j] - output_out[j]) * self.who[j][p]
                pass
            weighted_errors.append(weighted_error_p)
            pass

        for j in range(self.onodes):
            for p in range(self.hnodes):
                self.who[j][p] += self.lr * (target[j] - output_out[j]) * hidden_out[p]
                pass
            pass
        for p in range(self.hnodes):
            for i in range(self.inodes):
                self.wih[p][i] += self.lr * weighted_errors[p] * hidden_out_d[p] * input[i]
                pass
            self.t[p] -= self.lr * weighted_errors[p] * hidden_out_d[p]
            pass
        pass

    def query(self, input):
        hidden_out = []
        for p in range(self.hnodes):
            hidden_in = 0
            for i in range(self.inodes):
                hidden_in += self.wih[p][i] * input[i]
                pass
            hidden_out.append(self.activation_function(hidden_in - self.t[p]))
            pass
        output_out = []
        for j in range(self.onodes):
            output_in = 0
            for p in range(self.hnodes):
                output_in += self.who[j][p] * hidden_out[p]
                pass
            output_out.append(output_in)
            pass
        return output_out


def xor_test():
    n = NeuralNetwork(2, 2, 1, 0.5)

    # n.wih = numpy.zeros((n.hnodes, n.inodes))
    # n.who = numpy.zeros((n.onodes, n.hnodes))
    # n.t = numpy.zeros(n.hnodes)

    input_values = [[0, 0], [1, 0], [0, 1], [1, 1]]
    target_values = [[0.01], [0.99], [0.99], [0.01]]

    epochs = 5
    for e in range(epochs):
        for input, target in zip(input_values * 100, target_values * 100):
            n.train(input, target)

    res = list(map(n.query, input_values))

    for r in res:
        print(r)


def mnist_test():
    n = NeuralNetwork(784, 200, 10, 0.2)

    training_data_file = open("mnist_dataset/mnist_test_100.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    epochs = 3
    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(10) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
            pass
        pass

    test_data_file = open("mnist_dataset/mnist_test_10.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    scorecard = []

    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        print(correct_label, "correct label")
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        print(label, "network's answer")
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass
        pass
    scorecard_array = numpy.asarray(scorecard)
    print("performance=", scorecard_array.sum() / scorecard_array.size)


# xor_test()
mnist_test()
