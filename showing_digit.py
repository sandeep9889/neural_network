import numpy
import matplotlib.pyplot
%matplotlib inline

data_file = open("mnist_dataset/mnist_train_100.csv","r")
data_list = data_file.readlines()
data_file.close
all_values = data_list[0].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array,cmap='Greys', interpolation='none')
