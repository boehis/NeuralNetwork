import matplotlib.pyplot
import numpy

import mnist

n = mnist.get_trained_network_fast(784, 200, 10, 0.2)

label = 4
targets = numpy.zeros(10) + 0.01
targets[label] = 0.99
print(targets)

image_data = n.backquery(targets)

matplotlib.pyplot.imshow(image_data.reshape(28, 28), cmap='Greys', interpolation='None')
matplotlib.pyplot.show()
