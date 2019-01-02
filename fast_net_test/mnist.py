import numpy

from fast_net_test.mnist_helper import get_trained_network_full


def test_network():
    n = get_trained_network_full(784, 200, 10, 0.2)
    test_data_file = open("../datasets/mnist_dataset/mnist_test.csv", 'r')
    # n = get_trained_network_fast(784, 200, 10, 0.2)
    # test_data_file = open("../datasets/mnist_dataset/mnist_test_10.csv", 'r')

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


test_network()
