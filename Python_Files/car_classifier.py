# -*- coding: utf-8 -*-

# Installs packages neccessary to create the model
!pip install -U tensorflow_hub
!pip install tf-nightly-gpu
!pip install -q h5py pyyaml

# Allows Colaboratory to link to Google Drive
from google.colab import drive
drive.mount('/content/gdrive')

# Imports Packages needed to create model
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow import keras

import tensorflow_hub as hub
from tensorflow.keras import layers

# Assigns a URL that links to the classifier to the variable classifier_url
classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"

# Assigns a image size to the variable IMAGE_SHAPE
IMAGE_SHAPE = (224,224)


classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])

# Imports more packages needed for the program
import os
import numpy as np
import PIL.Image as Image

# Assigns the root directories for the training data to the variables training_data_root
training_data_root = '../content/gdrive/My Drive/Software Engineering Practice/train'

# Resizes the training images to fit in the model
training_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
training_image_data = training_image_generator.flow_from_directory(str(training_data_root), target_size=IMAGE_SHAPE)

# pairs up image batches with label batches
for image_batch, label_batch in training_image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break

# shows the shape of the result_batch
result_batch = classifier.predict(image_batch)
result_batch.shape

# Gets the labels list from Google Drive and places them in an array
labels_path = '../content/gdrive/My Drive/Software Engineering Practice/training_names.txt'
prediction_labels = np.array(open(labels_path).read().splitlines())
predicted_class_names = prediction_labels[np.argmax(label_batch, axis=-1)]
predicted_class_names

# Shows preliminary predictions for 30 images in the training data
plt.figure(figsize=(25,19))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
  _= plt.suptitle("Preliminary predictions")

# Downloads a feature extractor and initializes the feature extractor layer to accept images of size 224 x 224
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224,224,3))

# Feature extractor returns a 1280 element vector for each image
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)

# Freezes the feature extractor layer of the model to prevent retraining
feature_extractor_layer.trainable = False

# Wraps the extraction layer in a sequential model and shows a summary
model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(training_image_data.num_classes, activation='softmax')
])


model.summary()

predictions = model(image_batch)
predictions.shape



# Compiles the model using the Adam optimizer and taking accuracy as a metric
model.compile(
optimizer=tf.keras.optimizers.Adam(),
loss='categorical_crossentropy',
    metrics=['acc'])

# Creates a class to collect batch stats such as loss and accuracy from each epoch
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []
    
  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

from keras.callbacks import *
from tensorflow.keras.callbacks import *

# Declares how many steps for each epoch to run
steps_per_epoch = np.ceil(training_image_data.samples/training_image_data.batch_size)

# Assigns the class for collecting batch stats to a variable
batch_stats = CollectBatchStats()

# Trains the model on 11 epochs
history = model.fit(training_image_data, epochs=11,
                   steps_per_epoch=steps_per_epoch,
                   callbacks = [batch_stats])

# Save Model to Google Drive Model Folder
model.save('../content/gdrive/My Drive/Software Engineering Practice/Model/car_classifier_model.h5')

#Plots a graph showing how loss of the model has changed over the training steps
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats.batch_losses)

#Plots a graph showing how Accuracy of the model has changed over the training steps
plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats.batch_acc)

# Loads the model from Google Drive Model Folder
model = keras.models.load_model('../content/gdrive/My Drive/Software Engineering Practice/Model/car_classifier_model.h5', custom_objects={'KerasLayer':hub.KerasLayer})
# Displays a summary of the model
model.summary()

# Sorts the label names and displays the array of names
label_names = sorted(training_image_data.class_indices.items(), key=lambda pair:pair[1])
label_names = np.array([key.title() for key, value in label_names])
label_names

# Assigns predicted label id's for the image_batch to variables
result_batch = model.predict(image_batch)
result_id = np.argmax(result_batch, axis=-1)
result_labels_batch = label_names[result_id]

label_id = np.argmax(label_batch, axis=-1)

# Displays a plot of 30 images with the prediction as each images title, the title of the images is green if correct and red if incorrect
plt.figure(figsize=(25,19))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  color = "green" if result_id[n] == label_id[n] else "red"
  plt.title(result_labels_batch[n].title(), color=color)
  plt.axis('off')
  _ = plt.suptitle("Model predictions (Names Will Be Green If Correct And Red If
