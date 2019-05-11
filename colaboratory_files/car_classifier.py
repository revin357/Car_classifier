pip install -U tensorflow_hub
pip install tf-nightly-gpu
pip install -q h5py pyyaml

from google.colab import drive
drive.mount('/content/gdrive')

from __future__ import absolute_import, division, print_function, unicode_literals

import os

import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow import keras
#tf.enable_eager_execution()

import tensorflow_hub as hub
from tensorflow.keras import layers

classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"

IMAGE_SHAPE = (224,224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])

import os
import numpy as np
import PIL.Image as Image

data_root = '../content/gdrive/My Drive/Software Engineering Practice/train'

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)

for image_batch, label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break

result_batch = classifier.predict(image_batch)
result_batch.shape

labels_path = '../content/gdrive/My Drive/Software Engineering Practice/training_names.txt'
imagenet_labels = np.array(open(labels_path).read().splitlines())
predicted_class_names = imagenet_labels[np.argmax(label_batch, axis=-1)]
predicted_class_names

plt.figure(figsize=(25,19))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
  _= plt.suptitle("ImageNet predictions")

feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224,224,3))

feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)

feature_extractor_layer.trainable = False

model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(image_data.num_classes, activation='softmax')
])


model.summary()

predictions = model(image_batch)

predictions.shape

model.compile(
optimizer=tf.keras.optimizers.Adam(),
loss='categorical_crossentropy',
    metrics=['acc'])

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

steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)

batch_stats = CollectBatchStats()

import os 
checkpoint_path = "../gdrive/My Drive/Software Engineering Practice/Checkpoints/training_1/cp.ckpt" 

checkpoint_dir = os.path.dirname(checkpoint_path)

 # Create checkpoint  callback 
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
     monitor='acc',save_best_only=True,save_weights_only=True,verbose=1)

history = model.fit(image_data, epochs=11,
                   steps_per_epoch=steps_per_epoch,
                   callbacks = [batch_stats])

model.save('car_classifier_model.h5')

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

label_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
label_names = np.array([key.title() for key, value in label_names])
label_names

result_batch = model.predict(image_batch)

labels_batch = label_names[np.argmax(result_batch, axis=-1)]
labels_batch

plt.figure(figsize=(25,19))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(labels_batch[n])
  plt.axis('off')
  _ = plt.suptitle("Model predictions")
