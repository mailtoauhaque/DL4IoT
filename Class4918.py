#Github : mailtoauhaque
#Python interpreter: 3.6
#Deep Learning 4 IoT Course


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print (train_images.shape)
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
print (len(train_images))
print (len(test_images))

train_images = train_images / 255.0
test_images = test_images / 255.0
print (len(train_images))
print (len(test_images))


# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


#Modeling
"""
tf.keras.layers.Flatten: transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels.
tf.keras.layers.Dense layers: fully-connected layers. layer 1 has 128 nodes/neurons.
tf.keras.layers.Dense layers: layer 2 is a 10-node softmax layer—this returns an array of 10 probability scores that sum to 1. Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.
"""
#NSG: rcrelu/rcrelu6/crelu
#LSG: crcrelu
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    # keras.layers.Dense(10, activation=tf.nn.crelu),
    # keras.layers.Dense(20, activation=tf.nn.crelu),
    # keras.layers.Dense(30, activation=tf.nn.crelu),
    # keras.layers.Dense(40, activation=tf.nn.crelu),
    # keras.layers.Dense(50, activation=tf.nn.crelu),
    # keras.layers.Dense(60, activation=tf.nn.crelu),
    # keras.layers.Dense(70, activation=tf.nn.crelu),
    # keras.layers.Dense(80, activation=tf.nn.crelu),
    # keras.layers.Dense(90, activation=tf.nn.crelu),
    # keras.layers.Dense(100, activation=tf.nn.crelu),
    # keras.layers.Dense(110, activation=tf.nn.crelu),
    # keras.layers.Dense(120, activation=tf.nn.crelu),
    # keras.layers.Dense(130, activation=tf.nn.crelu),
    # keras.layers.Dense(140, activation=tf.nn.crelu),
    keras.layers.Dense(150, activation=tf.nn.crelu),
    # keras.layers.Dense(160, activation=tf.nn.crelu),
    # keras.layers.Dense(120, activation=tf.nn.selu),
    # keras.layers.Dense(120, activation=tf.nn.selu),
    # keras.layers.Dense(120, activation=tf.nn.selu),
    # keras.layers.Dense(120, e(120, activation=tf.nn.crelu),
    # keras.layers.Dense(130, activation=tf.nn.crelu),
    # keras.layers.Dense(140, activation=tf.nn.crelu),
    # keras.layers.Dense(activation=tf.nn.selu),
    # keras.layers.Dense(120, activation=tf.nn.selu),
    # keras.layers.Dense(120, activation=tf.nn.selu),
    # keras.layers.Dense(120, activation=tf.nn.selu),
    # keras.layers.Dense(120, activation=tf.nn.selu),
    # keras.layers.Dense(120, activation=tf.nn.selu),
    # keras.layers.Dense(120, activation=tf.nn.selu),
    # keras.layers.Dense(120, activation=tf.nn.selu),
    # keras.layers.Dense(120, activation=tf.nn.selu),
    # keras.layers.Dense(120, activation=tf.nn.selu),
    # keras.layers.Dense(120, activation=tf.nn.selu),
    # keras.layers.Dense(120, activation=tf.nn.selu),
    # keras.layers.Dense(120, activation=tf.nn.selu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

"""
Loss function: Measures how accurate the model is during training. Minimize it to "steer" the model in the right direction.
Optimizer: This updates the model based on the data it sees and its loss function.
Metrics: Monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
"""
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""To start training call"""
model.fit(train_images, train_labels, epochs=5)

"""Comparison of Testing with actual"""
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

"""Prediction on the basis of trained model"""
predictions = model.predict(test_images)
print ("predictions", predictions[1])
print ("np.argmax", np.argmax(predictions[1]))
print ("Test_Labels:", test_labels[1])

print('Before Plot_image')

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

#
# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# # plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# # plot_value_array(i, predictions,  test_labels)
# plt.show()

#predict several images with tags
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  #plot_value_array(i, predictions, test_labels)
# plt.show()



