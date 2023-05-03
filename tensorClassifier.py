import os
import random
from os import mkdir

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from numpy import mean

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def buildModel(dataDirectory,modelDirectory):
  batch_size = 32
  img_height = 64
  img_width = 64
  data_dir = dataDirectory

  train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

  class_names = train_ds.class_names
  print(class_names)

  AUTOTUNE = tf.data.AUTOTUNE

  train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

  normalization_layer = layers.Rescaling(1./255)
  normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
  image_batch, labels_batch = next(iter(normalized_ds))

  num_classes = len(class_names)

  model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  model.summary()

  epochs=15
  model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
  )

  model.save(modelDirectory)

  # history.save('saved_model/myTensorFlowModel')
  # acc = history.history['accuracy']
  # val_acc = history.history['val_accuracy']
  #
  # loss = history.history['loss']
  # val_loss = history.history['val_loss']
  #
  # epochs_range = range(epochs)
  #
  # plt.figure(figsize=(8, 8))
  # plt.subplot(1, 2, 1)
  # plt.plot(epochs_range, acc, label='Training Accuracy')
  # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  # plt.legend(loc='lower right')
  # plt.title('Training and Validation Accuracy')
  #
  # plt.subplot(1, 2, 2)
  # plt.plot(epochs_range, loss, label='Training Loss')
  # plt.plot(epochs_range, val_loss, label='Validation Loss')
  # plt.legend(loc='upper right')
  # plt.title('Training and Validation Loss')
  # plt.show()




