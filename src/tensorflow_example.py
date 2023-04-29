import tensorflow as tf

# All one
x = tf.ones(shape=(2, 1))
print(x)

# All zeros
x = tf.zeros(shape=(2 , 1))
print(x)

#  random
x = tf.random.normal(shape=(3 , 1) , mean=0. , stddev=1.0)
print('normal distribuion' , x)

x = tf.random.uniform(shape=(3 , 1) , minval=0. , maxval= 1.0)
print('uniform' , x)

# tf tensor is readonly
#x[0,0] = 1.0

v = tf.Variable(initial_value= tf.random.normal(shape=(3 , 1)))
print(v)
v.assign(tf.ones((3 , 1)))
print(v)


input_var = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:
    result = tf.square(input_var)
gradient = tape.gradient(result , input_var)
print('gradient is ' , gradient)


