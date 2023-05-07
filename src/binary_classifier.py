import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


import sys

input_dim = 2  # input is two dimension ( x , y point)
output_dim = 1  # output is one dimension ( type a , b ...)
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim, 1)))
def binary_classifier():

    """
        Simple binary classifier to classify two sets of dots

    :return:
    """
    num_samples_per_class = 1000
    negative_samples = np.random.multivariate_normal(
        mean = [0 , 3],
        cov = [[1 , 0.5] , [0.5 , 1]],
        size = num_samples_per_class
    )

    positive_samples = np.random.multivariate_normal(
        mean = [3 ,0],
        cov = [[1 , 0.5] , [0.5 , 1]],
        size = num_samples_per_class
    )

    inputs = np.vstack((negative_samples , positive_samples)).astype(np.float32)

    targets = np.vstack(
        (np.zeros(( num_samples_per_class , 1) , dtype="float32") ,
        np.ones(( num_samples_per_class , 1) , dtype="float32"))
    )





    for step in range(40):
        print(f' training step {step}')
        loss = traning_step(inputs , targets)
        print(f"Loss at step {step} : {loss:.4f}")


    predictions = model(inputs)


    x = np.linspace(-1 , 4 , 100)
    y = -W[0]/W[1] *x + (0.5-b)/W[1]
    plt.plot(x , y[0, :] , "-r")
    plt.scatter(inputs[: , 0] , inputs[: , 1] , c=predictions[: , 0] > 0.5)
    plt.show()


    plt.scatter(inputs[: , 0] , inputs[: , 1] , c = targets[:, 0])
    plt.show()
    sys.stdout.flush()

def model(inputs):
    #
    # prediction = W *input +b


    return tf.matmul(inputs , W) + b
    #return tf.math.sigmoid(tf.matmul(inputs , W) + b)
    #return tf.nn.relu(tf.matmul(inputs , W) + b)


def square_loss(targets , predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)



def traning_step(inputs , targets , learning_rate = 0.1):


    with tf.GradientTape() as tape:
        predictions =  model(inputs)
        loss = square_loss(targets , predictions)
    grad_loss_wrt_W , grad_loss_wrt_b = tape.gradient(loss , [W , b])
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)

    return loss

if "__main__" == __name__ :

    binary_classifier()
