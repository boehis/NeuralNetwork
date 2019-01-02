import glob

import imageio
import numpy

from fast_net_test.mnist_helper import get_trained_network_full


def datasets_from_inputimages():
    dataset = []
    for image_file_name in glob.glob('../datasets/inputimages/?.png'):
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
    n = get_trained_network_full(784, 200, 10, 0.2)

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


test_own_images()
