import numpy as np
import pandas as pd
import tensorflow as tf

"""
Authors:
- Badysiak Pawel - s21166
- Turek Wojciech - s21611
"""

"""
    Banknotes authentication classifying
"""
print("Banknotes authentication")
b_data = pd.read_csv('data_banknote_authentication.csv', names=['variance', 'skewness', 'kurtosis', 'entropy', 'class'])
b_train_data = b_data.copy()
b_train_label = b_data.pop('class')
b_train_data = np.array(b_train_data)

b_model = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu'),
                                      tf.keras.layers.Dense(1)
                                      ])
b_model.compile(optimizer='adam',
                       loss='mean_squared_error',
                       metrics=['accuracy']
                       )
b_model.fit(b_train_data, b_train_label, epochs=7)


"""
Bird, Cat and Deed from CIFAR10 classifying
"""
print("\nBird, Cat and Deer from CIFAR10 classifying")
cifar10_data = tf.keras.datasets.cifar10
(c_train_data, c_train_label), (c_test_data, c_test_label) = cifar10_data.load_data()
c_train_data = c_train_data / 255
c_test_data = c_test_data / 255

bird_index = np.where(c_train_label.reshape(-1) == 2)
bird_data = c_train_data[bird_index]
bird_label = c_train_label[bird_index]

cat_index = np.where(c_train_label.reshape(-1) == 3)
cat_data = c_train_data[cat_index]
cat_label = c_train_label[cat_index]

deer_index = np.where(c_train_label.reshape(-1) == 4)
deer_data = c_train_data[deer_index]
deer_label = c_train_label[deer_index]

animals_train_data = np.concatenate((bird_data, cat_data, deer_data))
animals_train_label = np.concatenate((bird_label, cat_label, deer_label)).reshape(-1, 1)
animals_train_label[animals_train_label == 2] = 0
animals_train_label[animals_train_label == 3] = 1
animals_train_label[animals_train_label == 4] = 2

cifar3_model = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(64, activation='relu'),
     tf.keras.layers.Dense(3, activation='softmax')
     ])
cifar3_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy']
                     )
cifar3_model.fit(animals_train_data, animals_train_label, epochs=11)