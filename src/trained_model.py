import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import os


class TrainedModel():
    def __init__(self, model_dir, model_id):
        self.model_id = model_id
        self.model_dir = model_dir
        self.load_model()

    def load_model(self):
        model_id = self.model_id
        model_dir = self.model_dir
        self.model = tf.keras.models.load_model(model_dir + '/' + model_id +
                                                '.h5')
        self.train_history = joblib.load(model_dir + '/' + model_id +
                                         '_history')

        self.img_shape = tuple(self.model.input.shape[1:])

        # Utility function to plot history of training a network
    def plot_training(self):
        train_accuracy = self.train_history['accuracy']
        val_accuracy = self.train_history['val_accuracy']
        epoch = range(1, len(train_accuracy) + 1)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epoch, train_accuracy, label='Training Accuracy')
        ax.plot(epoch, val_accuracy, label='Validation Accuracy')
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.legend()
        plt.legend(framealpha=0)

    # Function to return confusion matrix given a model and data
    def conf_matrix(self, test):
        predict_labels = self.model.predict_classes(self.test[0][0])
        ground_truth = self.test[0][1]

        df = pd.DataFrame(confusion_matrix(ground_truth, predict_labels),
                          columns=['Predicted Fake', ' Predicted Real'],
                          index=['Actual Fake', 'Actual Real'])
        return df

    # Function to return the classification summary
    def classification_report(self, test):
        predict_labels = self.model.predict_classes(self.test[0][0])
        ground_truth = self.test[0][1]
        keys = self.test.class_indices.keys()
        labels = [self.test.class_indices[key] for key in keys]

        print(
            classification_report(ground_truth,
                                  predict_labels,
                                  labels=labels,
                                  target_names=keys))

    # Function to convert predicted probabilities into class predictions
    def prob_to_labels(predict_prob):
        if len(predict_prob[0]) == 1:
            predict_labels = [0 if i > 0.5 else 1 for i in predict_prob]
        else:
            pass
        return predict_labels

    def test_on_difficulty(self, tests=1000):

        fake_paths, real_paths = self._get_paths(test_dir, dif, n=tests)

        fake_imgs = np.zeros((tests, ) + self.img_shape)
        real_imgs = np.zeros((tests, ) + self.img_shape)

        for i, (fake_path, real_path) in enumerate(zip(fake_paths,
                                                       real_paths)):
            fake_imgs[i], _ = self._process_path(fake_path)
            real_imgs[i], _ = self._process_path(real_path)

        fake_imgs = tf.convert_to_tensor(fake_imgs)
        real_imgs = tf.convert_to_tensor(real_imgs)

        print(fake_imgs[0])

        fake_probs = self.model.predict(fake_imgs)
        real_probs = self.model.predict(real_imgs)

        correct = (fake_probs < real_probs) * 1

        return np.mean(correct)

    def _get_paths(self, test_dir, dif='hard', n=1):
        real_ls = os.listdir(f'{test_dir}/real')
        fake_ls = os.listdir(f'{test_dir}/fake')
        if dif == 'all':
            fake_d_ls = [f'fake/{i}' for i in fake_ls]
        else:
            fake_d_ls = [f'fake/{i}' for i in fake_ls if dif in i]
        real_d_ls = [f'real/{i}' for i in real_ls]

        fake_paths = []
        real_paths = []
        for i in range(n):
            fake_path = np.random.choice(fake_d_ls)
            real_path = np.random.choice(real_d_ls)
            fake_paths.append(test_dir + '/' + fake_path)
            real_paths.append(test_dir + '/' + real_path)

        return fake_paths, real_paths

    def _process_path(self, path):
        if 'fake' in path:
            label = 0
        else:
            label = 1
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize_with_pad(img, self.img_shape[0],
                                       self.img_shape[1])
        return img, label

    def spot_from_pair(self, left_paths, right_paths):
        # If not a list convert to a list
        if not isinstance(left_paths, list):
            left_paths = [left_paths]
            right_paths = [right_paths]

        n_pairs = len(left_paths)
        left_imgs = np.zeros((n_pairs, ) + self.img_shape)
        right_imgs = np.zeros((n_pairs, ) + self.img_shape)
        print('left_paths in spot_from_pair', left_paths)
        for i, (left_path,
                right_path) in enumerate(zip(left_paths, right_paths)):

            print('Attempting to read image from:', left_path)
            left_imgs[i], _ = self._process_path(left_path)
            right_imgs[i], _ = self._process_path(right_path)

        left_imgs = tf.convert_to_tensor(left_imgs)
        right_imgs = tf.convert_to_tensor(right_imgs)

        left_probs = self.model.predict(left_imgs)
        right_probs = self.model.predict(right_imgs)
        print(left_probs)
        dif = ((left_probs[:, 1] - right_probs[:, 1]) > 0) * 1

        fake_on = ['left' if i == 1 else 'right' for i in dif]
        if len(fake_on) == 1:
            return fake_on[0]
        else:
            return fake_on
