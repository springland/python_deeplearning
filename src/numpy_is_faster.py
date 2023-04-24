import numpy as np
import time

def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy() # make a copy
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i , j] = max(x[i , j] , 0)

    return x

def naive_add(x , y ):
    assert len(x.shape) == 2
    assert x.shape  == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            x[i , j] += y[i, j]

    return x


def np_is_faster():
    ##
    #  numpy is faster
    #
    x = np.random.random((20 , 100))
    y = np.random.random((20 , 100))

    t0 = time.time()
    for _ in range(1000):
        z = x + y
        z = np.maximum(z , 0)

    print( "np took: {0:.4f} s".format(time.time() - t0))

    t0 = time.time()
    for _ in range(1000):
        z = naive_add(x , y)
        z = naive_relu(z)
    print( "naive took: {0:.4f} s".format(time.time() - t0))


def np_reshape():
    x = np.array(
        [[0.0 , 1.0],
         [2.0 , 3.0],
         [4.0 , 5.0]]
    )

    print(x.shape)
    x = x.reshape((6 , 1))
    print(x)

    x = x.reshape((2 , 3))
    print(x)


def maxtrix_transpose():
    x = np.zeros((300 , 20))
    print('before transpose' , x.shape)
    x = np.transpose(x)
    print('after transpose', x.shape)

if  __name__ == "__main__" :

    #np_is_faster()
    #np_reshape()
    maxtrix_transpose()


