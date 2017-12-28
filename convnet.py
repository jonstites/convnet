#!/usr/bin/env python3

import argh
import datasets
import numpy as np
import tensorflow as tf

def build_graph():
    g = tf.Graph()
    with g.as_default():

        images = tf.placeholder(dtype=np.float32, shape=[None, 28, 28], name="images")
        flat_images = tf.contrib.layers.flatten(images)

        labels = tf.placeholder(dtype=np.int32, shape=[None], name="labels")
        one_hot_labels = tf.one_hot(labels, 10)

        print(images)
        print(labels)
        logits = tf.contrib.layers.fully_connected(flat_images, num_outputs=10)
        print(logits)
        loss = tf.identity(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)), name="loss")
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss, name="optimizer")        
    return g

def main(train_dir, test_dir):
    not_mnist = datasets.NotMNISTDataset(train_dir, test_dir)
    graph = build_graph()
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        images, labels = not_mnist.train.next_batch(32)
        print(images.shape)
        print(labels.shape)
        feed = {"images:0": images, "labels:0": labels}
        loss, _ = sess.run(["loss:0", "optimizer"], feed_dict=feed)
        print("loss: ", loss)

if __name__ == "__main__":
    argh.dispatch_command(main)
