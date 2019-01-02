import numpy

from neural_networks.threelayerd_neural_network import ThreeLayerdNeuralNetwork


def xor_test():
    print("xor test")
    n = ThreeLayerdNeuralNetwork(2, 8, 1, 0.2)

    input_values = [[0.01, 0.01], [0.99, 0.01], [0.01, 0.99], [0.99, 0.99]]
    target_values = [[0.01], [0.99], [0.99], [0.01]]

    epochs = 2000
    for e in range(epochs):
        for input_val, target_val in zip(input_values, target_values):
            n.train(input_val, target_val)

    res = list(map(n.query, input_values))

    # result should be similar to target values
    for r in res:
        print(r)


def mnist_test():
    print("mnist test")
    n = ThreeLayerdNeuralNetwork(784, 200, 10, 0.2)

    training_data_file = open("datasets/mnist_dataset/mnist_train_100.csv", 'r')
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

    test_data_file = open("datasets/mnist_dataset/mnist_test_10.csv", 'r')
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


xor_test()
mnist_test()
