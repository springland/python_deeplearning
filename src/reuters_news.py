
import numpy as np
import keras

import keras.layers as layers
from keras.datasets import reuters
import matplotlib.pyplot as plt

def reuter_news_categorical_crossentropy(num_words=10000):
    (train_data , train_labels) , (test_data , test_labels) = reuters.load_data(num_words=num_words)


    topics_num = 46
    x_train = vectorize(train_data , num_words)
    x_test = vectorize(test_data , num_words)


    y_train = to_one_hot(train_labels)
    y_test = to_one_hot(train_labels)

    reuter_model(x_train , y_train , "categorical_crossentropy" , 5)
def reuter_news_sparse_categorical_crossentropy( num_words = 10000):
    (train_data , train_labels) , (test_data , test_labels) = reuters.load_data(num_words=num_words)


    topics_num = 46
    x_train = vectorize(train_data , num_words)
    x_test = vectorize(test_data , num_words)

    y_train = np.asarray(train_labels ).astype("float32")
    y_test  = np.asarray(test_labels).astype("float32")

    reuter_model(x_train = x_train , y_train = y_train , loss="sparse_categorical_crossentropy" ,epochs=5 )

    pass


def reuter_model(x_train  , y_train  , loss  , epochs):
    model = keras.Sequential(
        [
            layers.Dense(512, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(46, activation="softmax")
        ]

    )

    model.compile(optimizer="rmsprop", loss=loss, metrics=["accuracy"])

    x_val = x_train[: 1000]
    partial_x_train = x_train[1000:]

    y_val = y_train[: 1000]
    partial_y_train = y_train[1000:]

    # seems 5 rounds is the best

    history = model.fit(
        partial_x_train,
        partial_y_train,
        epochs=epochs,
        batch_size=128,
        validation_data=(x_val, y_val)
    )

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1 , len(loss) +1)

    plt.plot(epochs , loss , "bo" , label = "Training loss")
    plt.plot(epochs , val_loss , "b" , label = "Valuation loss")
    plt.title("Training and valuation losses" )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return ( model , history)

def vectorize( sequences , num_words = 10000):

    x_train = np.zeros((len(sequences) , num_words))

    for i , sequence in enumerate(sequences):
        for j in sequence:
            x_train[i , j] = 1

    return x_train

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results



if __name__ == "__main__":
    #reuter_news_categorical_crossentropy()
    reuter_news_sparse_categorical_crossentropy()
    pass