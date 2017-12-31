# Convnet

A convolutional neural network for image classification. 

# Results

I've been using the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset because
bigger than MNIST, but not too big to play with on my desktop. 

Currently, the network gets a validation accuracy of 88.6% and a training accuracy of 97.8%. 
I won't use the test dataset until I am done tuning the network.

Some of the tricks that are implemented:
* one-hot encoding
* weight regularization
* batch normalization
* residual network (albeit a very small one)
* image whitening (subtracting the global mean pixel intensity and dividing by global standard deviation)

# TODO
* Rewrite with tests. Writing this was great learning experience, but the code is ugly.
* Use tensorboard to track the training and validation batch losses and accuracies.
* Try more network architectures, especially deeper residual networks.
* Performance optimizations, e.g. not passing a feed_dict through tensorflow.
* Network tuning, such as different parameters for the Adam optimizer and weight regularization.
* Try on other datasets, e.g. MNIST.
