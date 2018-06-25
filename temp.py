import random
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plot
from keras.preprocessing import image

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = models.Sequential()
network.add(layers.Dense(500, activation = 'relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation = 'softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

accuracy = network.evaluate(x=test_images, y=test_labels, batch_size=32)
print("Accuracy: ",accuracy[1])

while True:
    test_img = test_images[random.randint(501, 526)]
    test_img = test_img.reshape((1,784))
    img_class = network.predict_classes(test_img)
    prediction01 = img_class[0]
    
    print("Class: ",prediction01)

    test_img = test_img.reshape((28,28))
    plot.imshow(test_img)
    plot.title(prediction01)
    plot.show()
    
    loaded_img = image.load_img(path="6.png",grayscale=True,target_size=(28,28,1))
    loaded_img = image.img_to_array(loaded_img)
    loaded_img = loaded_img.reshape((1,784))
    
    img_class = network.predict_classes(loaded_img)
    prediction02 = img_class[0]
    
    print("Class: ",prediction02)
    loaded_img = loaded_img.reshape((28,28))
    plot.imshow(loaded_img)
    plot.title(prediction02)
    plot.show()
    
    if (prediction01 == prediction02):
        print("The Image loaded matches the random image from dataset.")
        break
    else:
        print("Image loaded doesn't match the random image from dataset.")