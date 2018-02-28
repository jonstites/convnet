# Convnet

A convolutional neural network for image classification. 

# Results

I've been using the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset because
bigger than MNIST, but not too big to play with on my desktop. 

Currently, the network gets a validation accuracy of 92.2% and a training accuracy of 96.0%. 
This is pretty good, considering that there's an estimated 6.5% label error rate in the dataset.

Some of the tricks that are implemented:
* weight regularization
* batch normalization
* image whitening (subtracting the global mean pixel intensity and dividing by global standard deviation)

# TODO
* Try more network architectures, especially deep residual networks.
* Network tuning, such as different parameters for the Adam optimizer and weight regularization.
* Try on other datasets, e.g. MNIST.
