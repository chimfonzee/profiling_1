from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

def to_profile(image_shape):
    Conv2D(32, kernel_size=3, activation='relu', input_shape=image_shape, kernel_initializer='he_normal', name='Conv2D-1')
    MaxPooling2D(pool_size=2, name='MaxPool')
    Dropout(0.25, name='Dropout-1')
    Conv2D(64, kernel_size=3, activation='relu', name='Conv2D-2')
    Dropout(0.25, name='Dropout-2')
    Conv2D(128, kernel_size=3, activation='relu', name='Conv2D-3')
    Dropout(0.4, name='Dropout-3')
    Flatten(name='flatten')
    Dense(128, activation='relu', name='Dense')
    Dropout(0.4, name='Dropout')
    Dense(10, activation='softmax', name='Output')

if __name__ == '__main__':
    to_profile((28, 28, 1))
