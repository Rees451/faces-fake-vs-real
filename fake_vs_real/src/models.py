import tensorflow.keras.layers as layers
import tensorflow.keras as keras

# CNN 1 ---------------------------------------------------------------
cnn1_model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='selu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='selu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='selu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(units=128, activation='selu'),
    layers.Dense(units=1, activation='sigmoid')
])
cnn1_model.compile(optimizer=keras.optimizers.Adam(),
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
cnn1 = {'id': 'c1', 'model': cnn1_model}

# CNN 2 ---------------------------------------------------------------
cnn2 = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='selu', input_shape=img_shape),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='selu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='selu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(units=128, activation='selu'),
    layers.Dense(units=1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam()
cnn2.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])