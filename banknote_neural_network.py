import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns

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
Confusion Matrix for CIFAR10 dataset
"""
print("\nConfusion Matrix for CIFAR10 dataset")
cifar10_data = tf.keras.datasets.cifar10
(c_train_data, c_train_label), (c_test_data, c_test_label) = cifar10_data.load_data()
c_train_data = c_train_data / 255
c_test_data = c_test_data / 255


cifar3_model = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(256, activation='relu'),
     tf.keras.layers.Dense(10, activation='softmax')
     ])
cifar3_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy']
                     )
cifar3_model.fit(c_train_data, c_train_label, epochs=5)

class_names = ['airplane','automobile','bird', 'cat', 'deer', 'dog', 'frog', 'horse','ship','truck']

image_classes_prediction=np.argmax(cifar3_model.predict(c_test_data), axis=1)
confusion_matrix = tf.math.confusion_matrix(labels=c_test_label, predictions=image_classes_prediction).numpy()
confusion_matrix_norm = np.around(confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
confusion_matrix_df = pd.DataFrame(confusion_matrix_norm,index = class_names,columns = class_names)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(confusion_matrix_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()

print("\nAnimals from CIFAR10 classifying [5] tests")
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

dog_index = np.where(c_train_label.reshape(-1) == 5)
dog_data = c_train_data[deer_index]
dog_label = c_train_label[deer_index]

frog_index = np.where(c_train_label.reshape(-1) == 6)
frog_data = c_train_data[deer_index]
frog_label = c_train_label[deer_index]

horse_index = np.where(c_train_label.reshape(-1) == 7)
horse_data = c_train_data[deer_index]
horse_label = c_train_label[deer_index]

animals_train_data = np.concatenate((bird_data, cat_data, deer_data, dog_data, frog_data, horse_data))
animals_train_label = np.concatenate((bird_label, cat_label, deer_label, dog_label, frog_label, horse_label)).reshape(-1, 1)
animals_train_label[animals_train_label == 2] = 0
animals_train_label[animals_train_label == 3] = 1
animals_train_label[animals_train_label == 4] = 2
animals_train_label[animals_train_label == 5] = 3
animals_train_label[animals_train_label == 6] = 4
animals_train_label[animals_train_label == 7] = 5


cifar3_model = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(256, activation='relu'),
     tf.keras.layers.Dense(6, activation='softmax')
     ])
cifar3_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy']
                     )
cifar3_model.fit(animals_train_data, animals_train_label, epochs=5)

print("\nAnimals from CIFAR10 classifying [10] tests (2nd approach)")
cifar10_data = tf.keras.datasets.cifar10
(c2_train_data, c2_train_label), (c2_test_data, c2_test_label) = cifar10_data.load_data()
c2_train_data = c2_train_data / 255
c2_test_data = c2_test_data / 255

bird_index = np.where(c_train_label.reshape(-1) == 2)
bird_data = c_train_data[bird_index]
bird_label = c_train_label[bird_index]

cat_index = np.where(c_train_label.reshape(-1) == 3)
cat_data = c_train_data[cat_index]
cat_label = c_train_label[cat_index]

deer_index = np.where(c_train_label.reshape(-1) == 4)
deer_data = c_train_data[deer_index]
deer_label = c_train_label[deer_index]

dog_index = np.where(c_train_label.reshape(-1) == 5)
dog_data = c_train_data[deer_index]
dog_label = c_train_label[deer_index]

frog_index = np.where(c_train_label.reshape(-1) == 6)
frog_data = c_train_data[deer_index]
frog_label = c_train_label[deer_index]

horse_index = np.where(c_train_label.reshape(-1) == 7)
horse_data = c_train_data[deer_index]
horse_label = c_train_label[deer_index]

animals_train_data = np.concatenate((bird_data, cat_data, deer_data, dog_data, frog_data, horse_data))
animals_train_label = np.concatenate((bird_label, cat_label, deer_label, dog_label, frog_label, horse_label)).reshape(-1, 1)
animals_train_label[animals_train_label == 2] = 0
animals_train_label[animals_train_label == 3] = 1
animals_train_label[animals_train_label == 4] = 2
animals_train_label[animals_train_label == 5] = 3
animals_train_label[animals_train_label == 6] = 4
animals_train_label[animals_train_label == 7] = 5

cifar32_model = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(256, activation='relu'),
     tf.keras.layers.Dense(6, activation='softmax')
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
