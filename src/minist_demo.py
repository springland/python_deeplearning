import keras
from keras.datasets import mnist
from keras import layers

(train_images , train_labels) ,(test_images , test_labels) = mnist.load_data()

print("train images" , train_images.shape)
print("train labels" , len(train_labels))

print("test images" , test_images.shape)
print("test_labels" , len(test_labels))

model = keras.Sequential([
        layers.Dense(512 , activation="relu"),
        layers.Dense(10 , activation="softmax")

    ]

)

model.compile( optimizer="rmsprop",
               loss = "sparse_categorical_crossentropy" ,
               metrics = ["accuracy"])


train_images = train_images.reshape( (60000 , 28*28))
train_images = train_images.astype("float32")/255

test_images = test_images.reshape((10000 , 28 * 28))
test_images = test_images.astype("float32")/255

model.fit(train_images , train_labels , epochs=5 , batch_size=128)

test_digits = test_images[0:10]
predictions = model.predict(test_images)
print(predictions[0])
print(predictions[0].argmax() , " matches")
print('test lable for 0 is ' , test_labels[0] )


test_loss , test_acct = model.evaluate(test_images , test_labels)
print(f" test_acc: {test_acct}")