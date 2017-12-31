#!/usr/bin/env python3

import argh
import datasets
import numpy as np
import tensorflow as tf

def build_graph():
    g = tf.Graph()
    with g.as_default():

        is_training = tf.placeholder(dtype=bool, shape=None, name="is_training")
        images = tf.placeholder(dtype=np.float32, shape=[None, 28, 28, 1], name="images")
        labels = tf.placeholder(dtype=np.int32, shape=[None], name="labels")
        one_hot_labels = tf.one_hot(labels, 10)

        weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4)

        out_channels = 64
        large_kernel_size = 5
        small_kernel_size = 3

        h = images
        
        h1 = tf.contrib.layers.conv2d(h, num_outputs=out_channels, kernel_size=large_kernel_size, weights_regularizer=weights_regularizer)

        h2 = tf.contrib.layers.batch_norm(h1, is_training=is_training)
        h3 = tf.contrib.layers.conv2d(h2, num_outputs=out_channels, kernel_size=small_kernel_size, weights_regularizer=weights_regularizer)
        h4 = tf.contrib.layers.batch_norm(h3, is_training=is_training)
        h5 = tf.contrib.layers.conv2d(h4, num_outputs=out_channels, kernel_size=small_kernel_size, weights_regularizer=weights_regularizer)
        h6 = tf.concat([h1, h5], axis=-1)
        h = tf.contrib.layers.batch_norm(h6, is_training=is_training)

        h = tf.contrib.layers.flatten(h)
        logits = tf.contrib.layers.fully_connected(h, num_outputs=10, activation_fn=None)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        cross_entropy_loss = tf.identity(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)), name="loss")
        loss = cross_entropy_loss + reg_losses
        
        correct = tf.equal(tf.argmax(one_hot_labels, 1), tf.argmax(logits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss, name="optimizer")        
    return g

def main(train_dir, test_dir):
    not_mnist = datasets.NotMNISTDataset(train_dir, test_dir)
    graph = build_graph()
    batch_size = 128
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        prev_epoch = 0
        while not_mnist.train._epochs_completed < 10:
            images, labels = not_mnist.train.next_batch(batch_size)
            feed = {"images:0": images, "labels:0": labels, "is_training:0": True}
            loss, _ = sess.run(["loss:0", "optimizer"], feed_dict=feed)

            if not_mnist.train._epochs_completed > prev_epoch:
                losses = []
                accuracies = []
                while not_mnist.validate._epochs_completed < not_mnist.train._epochs_completed:
                    images, labels = not_mnist.validate.next_batch(batch_size)
                    feed = {"images:0": images, "labels:0": labels, "is_training:0": False}
                    loss, accuracy = sess.run(["loss:0", "accuracy:0"], feed_dict=feed)
                    losses.append(loss)
                    accuracies.append(accuracy)
                mean_loss = np.mean(losses)
                mean_accuracy = np.mean(accuracies)
                print("finished epoch: ", not_mnist.train._epochs_completed - 1)
                print("mean validate loss:", mean_loss)
                print("mean validate accuracy:", mean_accuracy)
                prev_epoch = not_mnist.train._epochs_completed


        losses = []
        accuracies = []
        while not_mnist.train._epochs_completed < 11:
            images, labels = not_mnist.train.next_batch(batch_size)
            feed = {"images:0": images, "labels:0": labels, "is_training:0": False}
            loss, accuracy = sess.run(["loss:0", "accuracy:0"], feed_dict=feed)
            losses.append(loss)
            accuracies.append(accuracy)
        mean_loss = np.mean(losses)
        mean_accuracy = np.mean(accuracies)
        print("finished epoch: ", not_mnist.train._epochs_completed - 1)
        print("mean train loss:", mean_loss)
        print("mean train accuracy:", mean_accuracy)
                
                
if __name__ == "__main__":
    argh.dispatch_command(main)
