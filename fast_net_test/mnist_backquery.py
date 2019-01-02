import matplotlib.pyplot
import numpy

from fast_net_test.mnist_helper import get_trained_network_fast

n = get_trained_network_fast(784, 200, 10, 0.2)

for i in range(10):
    targets = numpy.zeros(10) + 0.01
    targets[i] = 0.99
    print(targets)

    image_data = n.backquery(targets)

    matplotlib.pyplot.imshow(image_data.reshape(28, 28), cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()

input("Press Enter to continue...")
