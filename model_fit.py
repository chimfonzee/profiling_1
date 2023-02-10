import keras
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

train_df = pd.read_csv('input/fashion-mnist_train.csv', sep=',')
test_df = pd.read_csv('input/fashion-mnist_test.csv', sep=',')
train_data = np.array(train_df, dtype='float32')
test_data = np.array(test_df, dtype='float32')
x_train = train_data[:50, 1:]/255
y_train = train_data[:50, 0]
x_test = test_data[:15, 1:]/255
y_test =test_data[:15, 0] 

x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2, random_state=12345)
image_rows = 28
image_cols = 28
batch_size = 512
image_shape = (image_rows, image_cols, 1)

x_train = x_train.reshape(x_train.shape[0], *image_shape)
x_test = x_test.reshape(x_test.shape[0], *image_shape)
x_validate = x_validate.reshape(x_validate.shape[0], *image_shape)

name='3_layer'
cnn_model_3 = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', 
           input_shape=image_shape, kernel_initializer='he_normal', name='Conv2D-1'),
    MaxPooling2D(pool_size=2, name='MaxPool'),
    Dropout(0.25, name='Dropout-1'),
    Conv2D(64, kernel_size=3, activation='relu', name='Conv2D-2'),
    Dropout(0.25, name='Dropout-2'),
    Conv2D(128, kernel_size=3, activation='relu', name='Conv2D-3'),
    Dropout(0.4, name='Dropout-3'),
    Flatten(name='flatten'),
    Dense(128, activation='relu', name='Dense'),
    Dropout(0.4, name='Dropout'),
    Dense(10, activation='softmax', name='Output')
], name=name)

cnn_model_3.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy']
)

history = cnn_model_3.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=50, verbose=1,
    validation_data=(x_validate, y_validate)
)
