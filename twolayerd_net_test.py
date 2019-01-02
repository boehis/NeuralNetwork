import numpy

from neural_networks.twolayerd_neural_network import TwoLayerdNeuralNetwork


# method to test two layerd neural network with simple association
def simple_example():
    print("simple example")
    n = TwoLayerdNeuralNetwork(2, 1)

    input_values = [[0, 0], [1, 0], [0, 1]]
    target_values = [[0], [1], [1]]

    # training continues until the network is fully trained
    while True:
        n.train(input_values, target_values)
        # apply the query function to all input values
        res = list(map(n.query, input_values))
        if numpy.array_equal(res, target_values):
            break

    print("fully trained!")
    res = list(map(n.query, input_values))
    for idx, v in enumerate(res):
        print("input:", input_values[idx], "result:", v)


def bigger_example():
    print("bigger example")
    n = TwoLayerdNeuralNetwork(2, 3)

    input_values = [[-2, 1], [-2, -2], [-1, -1], [2, -1], [3, 2]]
    target_values = [[0, 1, 0], [0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 1]]

    # training continues until the network is fully trained
    while True:
        n.train(input_values, target_values)
        # apply the query function to all input values
        res = list(map(n.query, input_values))
        if numpy.array_equal(res, target_values):
            break

    print("fully trained!")
    res = list(map(n.query, input_values))
    for idx, v in enumerate(res):
        print("input:", input_values[idx], "result:", v)


simple_example()
bigger_example()
