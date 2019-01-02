import numpy

from neural_networks import fast_neural_network


def get_trained_network_fast(input_nodes, hidden_nodes, output_nodes, learning_rate):
    return get_trained_network(input_nodes, hidden_nodes, output_nodes, learning_rate, 3, "mnist_train_100.csv")


def get_trained_network_full(input_nodes, hidden_nodes, output_nodes, learning_rate):
    return get_trained_network(input_nodes, hidden_nodes, output_nodes, learning_rate, 1, "mnist_train.csv")


def get_trained_network(input_nodes, hidden_nodes, output_nodes, learning_rate, epochs, file):
    n = fast_neural_network.FastNeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    training_data_file = open("../datasets/mnist_dataset/" + file, 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
    return n
