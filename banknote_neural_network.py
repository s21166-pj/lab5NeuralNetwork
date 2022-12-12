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