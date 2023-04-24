import tensorflow as tf


def gradient_scalar():
    x = tf.Variable(0.0)
    with tf.GradientTape() as tape:
        y = 4*x +3
        grad_of_y_wrt_x = tape.gradient(y , x)
        print(grad_of_y_wrt_x)


def gradient_matrix():
    x = tf.Variable(tf.zeros((2 , 2)))
    with tf.GradientTape() as tape:
        y = 2*x +3
        grad_of_y_wrt_x = tape.gradient(y , x)
        print(grad_of_y_wrt_x)


def gradient_more():
    W = tf.Variable(tf.random.uniform((2 , 2)))
    b = tf.Variable(tf.zeros(2 , ))
    x = tf.random.uniform((2 , 2))
    with tf.GradientTape() as tape:
        y = tf.matmul(x , W) + b
        grad_of_y_wrt_W_and_b = tape.gradient(y , [W , b])
        print(grad_of_y_wrt_W_and_b)



if __name__ == "__main__":
    #gradient_scalar()
    #gradient_matrix()
    gradient_more()