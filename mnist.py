import glob

import imageio
import numpy

import neural_net


def get_trained_network_fast(input_nodes, hidden_nodes, output_nodes, learning_rate):
    return get_trained_network(input_nodes, hidden_nodes, output_nodes, learning_rate, 3, "mnist_test_100.csv")


def get_trained_network_full(input_nodes, hidden_nodes, output_nodes, learning_rate):
    return get_trained_network(input_nodes, hidden_nodes, output_nodes, learning_rate, 1, "mnist_train.csv")


def get_trained_network(input_nodes, hidden_nodes, output_nodes, learning_rate, epochs, file):
    n = neural_net.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    training_data_file = open("mnist_dataset/" + file, 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
            pass
        pass

    return n


def test_network():
    n = get_trained_network_full(784, 200, 10, 0.2)

    test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
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


def datasets_from_inputimages():
    dataset = []
    for image_file_name in glob.glob('inputimages/?.png'):
        print("loading ... ", image_file_name)
        # use the filename to set the correct label
        label = int(image_file_name[-5:-4])
        # load image data from png files into an array
        img_array = imageio.imread(image_file_name, as_gray=True)
        # reshape from 28x28 to list of 784 values, invert values
        img_data = 255.0 - img_array.reshape(784)
        # then scale data to range from 0.01 to 1.0
        img_data = (img_data / 255.0 * 0.99) + 0.01
        print(numpy.min(img_data))
        print(numpy.max(img_data))
        # append label and image data  to test data set
        record = numpy.append(label, img_data)
        print(record)
        dataset.append(record)
        pass
    return dataset


def test_own_images():
    n = get_trained_network_fast(784, 200, 10, 0.2)

    for dataset in datasets_from_inputimages():
        correct_label = dataset[0]
        inputs = dataset[1:]
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        print("network says ", label)
        if label == correct_label:
            print("match!")
        else:
            print("no match! -> ", correct_label)
            pass


# test_network()
#test_own_images()
