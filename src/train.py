'''
File designed to be run in a remote environment to train models
'''

# Package imports
import tensorflow as tf
import joblib  # For saving history
import sys
import model_spec as md
import loading as ld
import copy

# Check that the GPU is being used
device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':
    print('Running on CPU')
print('Found GPU at: {}'.format(device_name))

# Input arguments
model_id = sys.argv[1]
model_obj = copy.copy(md.models[model_id]['model_obj'])

# Data load location
data_dir = sys.argv[2]

# Model save location
model_dir = sys.argv[3]

# Augmentation aruguments
train_args = {
    'rotation_range': 0,
    'height_shift_range': 0.05,
    'width_shift_range': 0.05,
    'horizontal_flip': True,
    'brightness_range': [1.1, 0.9],
    'zoom_range': [1.0, 1.0]
}

# Load data resizing to specific size
img_shape = (64, 64, 3)
# train, val = ld.load_data_generator(data_dir + '/' + 'train',
#                                     data_dir + '/' + 'val',
#                                     img_shape=img_shape,
#                                     train_args=train_args)

# Early stopping function
es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                      verbose=1,
                                      patience=5)

# Setup a model checkpoint to save our best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    f'{model_dir}/{model_id}.h5',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
)

#  Fit the tensorflow model
model_fit = model_obj.fit_generator(train,
                                    epochs=20,
                                    validation_data=val,
                                    callbacks=[checkpoint, es])

# Save the model fit history
joblib.dump(model_fit.history, f'{model_dir}/{model_id}_history')


# Supporting functions --------------------------------------------------------
def make_k_layers_trainable(model, k):

    # Number of layers
    n = len(model.layers)

    for layer in model.layers[:(n - k)]:
        layer.trainable = False

    return model
