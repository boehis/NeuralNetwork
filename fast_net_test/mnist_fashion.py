import numpy

from neural_networks import fast_neural_network
import utils.mnist_reader

X_train, y_train = utils.mnist_reader.load_mnist('../datasets/mnist_fashion_dataset', kind='train')
X_test, y_test = utils.mnist_reader.load_mnist('../datasets/mnist_fashion_dataset', kind='t10k')

# image_array = numpy.asfarray(X_test[0]).reshape(28, 28)
# matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
# matplotlib.pyplot.show()


n = fast_neural_network.FastNeuralNetwork(784, 400, 10, 0.2)

epochs = 3
for e in range(epochs):
    for idx, record in enumerate(X_train):
        inputs = (numpy.asfarray(record) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(10) + 0.01
        targets[int(y_train[idx])] = 0.99
        n.train(inputs, targets)
        pass
    pass

scorecard = []
for idx, record in enumerate(X_test):
    correct_label = int(y_test[idx])
    print(correct_label, "correct label")
    inputs = (numpy.asfarray(record) / 255.0 * 0.99) + 0.01
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
