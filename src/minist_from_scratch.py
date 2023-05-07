import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras import optimizers

import math


#
# The main difference between this one and minist_demo is minist_demo has an optimizer
#
#
#


class NaiveDense:
    def __init__(self , input_size , output_size , activation):
        #
        #  build y = W*x + b
        #
        self.activation = activation
        w_shape = (input_size , output_size)  # Build matrix
        w_init_value = tf.random.uniform(w_shape , minval=0 , maxval=1e-1)

        self.W = tf.Variable(w_init_value)
        b_shape = (output_size , )
        b_init_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_init_value)


    def __call__(self , inputs):
        return self.activation(tf.matmul(inputs , self.W) + self.b)

    @property
    def weights(self):
        return [ self.W , self.b]



class NaiveSequential:
    def __init__(self , layers):
        self.layers = layers

    def __call__(self , inputs):
        x = inputs
        for layer in self.layers:
            x =layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights


class BatchGenerator:
    def __init__(self , images , labels , batch_size = 128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.number_of_batches = math.ceil(len(images)/batch_size)

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index: self.index+self.batch_size]
        self.index = self.index + self.batch_size
        return images , labels


def one_training_step(model , images_batch , labels_batch):
    with tf.GradientTape() as tape :
        predictions = model(images_batch)

        per_sample_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch , predictions
        )
        average_loss = tf.reduce_mean(per_sample_loss)
        gradients = tape.gradient(average_loss  , model.weights)
        update_weights(gradients , model.weights)
        return average_loss




optimizer = optimizers.SGD(learning_rate=1e-3)

def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))

#def update_weights(gradients , weights , learning_rate = 1e-3):
#    for g , w in zip(gradients , weights):
#        w.assign_sub(g*learning_rate)


def fit(mode , images , labels , epochs , batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images , labels , batch_size)
        for batch_counter in range(batch_generator.number_of_batches):
            images_batch , labels_batch = batch_generator.next()
            loss = one_training_step(model , images_batch , labels_batch)
            if(batch_counter %100 == 0):
                print(f"loss at batch {batch_counter}:{loss:.2f}")
if __name__ == "__main__":

    model = NaiveSequential(
        [
            NaiveDense(input_size=28 * 28 , output_size=512 , activation=tf.nn.relu),
            NaiveDense(input_size = 512 , output_size=10 , activation=tf.nn.softmax)
        ]
    )


    assert len(model.weights) == 4

    (train_images , train_labels) , (test_images , test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000 , 28*28))
    train_images = train_images.astype("float32")/255

    test_images = test_images.reshape((10000 , 28*28))
    test_images = test_images.astype("float32")/255

    fit(model , train_images , train_labels , epochs=20 , batch_size=256)

    predictions = model(test_images)
    predictions = predictions.numpy()
    predicated_labels = np.argmax(predictions , axis=1)
    matches = predicated_labels  == test_labels
    print(f"accuracy: {matches.mean():.2f}")

