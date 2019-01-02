import random

import numpy

from neural_networks import fast_neural_network

scorecards = []
for testrun in range(10):
    n = fast_neural_network.FastNeuralNetwork(3, 2, 1, 0.2)

    training_data_file = open("../datasets/skintone/skintone_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    epochs = 3
    for e in range(epochs):
        for record in random.sample(training_data_list, 30000):
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[:3]) / 255.0 * 0.99) + 0.01
            target = int(all_values[-1:][0]) - 1
            target *= 0.98
            target += 0.01
            n.train(inputs, target)

    test_data_file = open("../datasets/skintone/skintone_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    scorecard = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[-1:][0]) - 1
        # print(correct_label, "correct label")
        inputs = (numpy.asfarray(all_values[:3]) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = int(round(outputs[0]))
        # print(label, "network's answer")
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass
        pass
    scorecard_array = numpy.asarray(scorecard)
    # print("performance=", scorecard_array.sum() / scorecard_array.size)
    scorecards.append(scorecard_array.sum() / scorecard_array.size)

scorecards_array = numpy.asarray(scorecards)
print("overall performance=", scorecards_array.sum() / scorecards_array.size)
