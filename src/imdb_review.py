import keras
from keras.datasets import imdb
from keras import layers
import matplotlib.pyplot as plt

import numpy as np



#
# decode the word index list to sentence
#
def  decode_review(word_list):
    word_index = imdb.get_word_index()
    reverse_word_index = dict (
        [(value , key) for (key , value) in word_index.items()]
    )
    decoded_review = " ".join(
        # the first 3 in reverse_word_index are padding ,  start of sequence and unknown
        # so substract 3
        [ reverse_word_index.get(i - 3 , "?")  for i in word_list]
    )

    return decoded_review


#
# make the sequences the same length
#
def vectorize_sequences(sequences , dimensions=10000):
    results = np.zeros(shape=(len(sequences) , dimensions) )
    for i , sequence in enumerate(sequences):
        for j in sequence:
            results[i , j] = 1
    return results ;




def display_training_result(history):

    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1 , len(loss_values) +1)
    plt.plot(epochs , loss_values , "bo" , label = "Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()

    plt.clf()
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    plt.plot(epochs , acc , "bo" , label = "Training acc")
    plt.plot(epochs , val_acc , "b" , label= "Validation acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()




def imdb_review():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words=10000
    )

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    y_train = np.asarray(train_labels).astype("float32")
    y_test = np.asarray(test_labels).astype("float32")


    model = keras.Sequential(
        [
            layers.Dense(16 , activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1 , activation="sigmoid")
        ]

    )

    model.compile(optimizer="rmsprop", loss="binary_crossentropy" , metrics=["accuracy"])

    x_val = x_train[: 10000]
    partial_x_train = x_train[10000:]

    y_val = y_train[: 10000]
    partial_y_train = y_train[10000:]

    history = model.fit(
        partial_x_train ,
        partial_y_train,
        epochs = 4 ,
        batch_size=512 ,
        validation_data=(x_val , y_val)


    )

    #display_training_result(history)

    results = model.evaluate(x_test , y_test)
    print('results ' ,results)

if __name__ == "__main__":
    imdb_review()