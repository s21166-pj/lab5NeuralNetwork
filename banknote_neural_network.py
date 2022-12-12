import matplotlib.pyplot as plt
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
print("\nBird, Cat and Deer from CIFAR10 classifying [5] tests")
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
cifar3_model.fit(animals_train_data, animals_train_label, epochs=5)

print("\nBird, Cat and Deer from CIFAR10 classifying [10] tests (2nd approach)")
cifar10_data = tf.keras.datasets.cifar10
(c2_train_data, c2_train_label), (c2_test_data, c2_test_label) = cifar10_data.load_data()
c2_train_data = c2_train_data / 255
c2_test_data = c2_test_data / 255

bird2_index = np.where(c2_train_label.reshape(-1) == 2)
bird2_data = c2_train_data[bird2_index]
bird2_label = c2_train_label[bird2_index]

cat2_index = np.where(c2_train_label.reshape(-1) == 3)
cat2_data = c2_train_data[cat_index]
cat2_label = c2_train_label[cat_index]

deer2_index = np.where(c2_train_label.reshape(-1) == 4)
deer2_data = c2_train_data[deer2_index]
deer2_label = c2_train_label[deer2_index]

animals2_train_data = np.concatenate((bird2_data, cat2_data, deer2_data))
animals2_train_label = np.concatenate((bird2_label, cat2_label, deer2_label)).reshape(-1, 1)
animals2_train_label[animals2_train_label == 2] = 0
animals2_train_label[animals2_train_label == 3] = 1
animals2_train_label[animals2_train_label == 4] = 2

cifar32_model = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(64, activation='relu'),
     tf.keras.layers.Dense(3, activation='softmax')
     ])
cifar32_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy']
                     )
cifar32_model.fit(animals_train_data, animals_train_label, epochs=10)

"""
10 Types of Clothes classifying.
"""
print("\n10 Types of Clothes classifying")
clothes_data = tf.keras.datasets.fashion_mnist
(clothes_train_data, clothes_train_label), (clothes_test_data, clothes_test_label) = clothes_data.load_data()
clothes_train_data = clothes_train_data / 255.0
clothes_test_data = clothes_test_data / 255.0

clothes_model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                     tf.keras.layers.Dense(128, activation='relu'),
                                     tf.keras.layers.Dense(10, activation='softmax')
                                     ])
clothes_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
                      )
clothes_model.fit(clothes_train_data, clothes_train_label, epochs=10)



"""
    Phoneme authentication classifying
"""
print("Phoneme authentication")
phoneme_data = pd.read_csv('phoneme.csv', names=['var1', 'var2', 'var3', 'var4', 'var5', 'class'])
phoneme_train_data = phoneme_data.copy()
phoneme_train_label = phoneme_data.pop('class')
phoneme_train_data = np.array(phoneme_train_data)

phoneme_model = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu'),
                                      tf.keras.layers.Dense(1)
                                      ])
phoneme_model.compile(optimizer='adam',
                       loss='mean_squared_error',
                       metrics=['accuracy']
                       )
phoneme_model.fit(phoneme_train_data, phoneme_train_label, epochs=3)
