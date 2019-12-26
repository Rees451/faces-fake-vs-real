'''
Module for loading data using tensorflow data generator
'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator as datagen
import os
import tensorflow as tf
import random


def load_data_generator(train_dir,
                        val_dir,
                        train_batch_size=16,
                        val_batch_size=16,
                        img_shape=(64, 64, 3),
                        train_args={},
                        val_args={}):

    target_size = img_shape[:2]

    train_datagen = datagen(rescale=1. / 255, **train_args)

    val_datagen = datagen(rescale=1. / 255, **val_args)

    # Trainining dataset generator
    train = train_datagen.flow_from_directory(train_dir,
                                              class_mode="binary",
                                              target_size=target_size,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              seed=1)

    # Validation dataset generator
    val = val_datagen.flow_from_directory(val_dir,
                                          class_mode="binary",
                                          target_size=target_size,
                                          batch_size=val_batch_size,
                                          shuffle=True,
                                          seed=1)

    assert train.class_indices == val.class_indices
    print('Class Indicies:', train.class_indices)
    return train, val


def load_test_generator(test_dir, batch_size=-1, img_shape=(64, 64, 3)):

    target_size = img_shape[:2]

    if batch_size == -1:
        batch_size = len([
            i for i in os.listdir(test_dir + '/fake') +
            os.listdir(test_dir + '/real') if '.jpg' in i
        ])

    test_datagen = datagen(rescale=1. / 255)

    # Test dataset generator
    test = test_datagen.flow_from_directory(test_dir,
                                            class_mode="binary",
                                            target_size=target_size,
                                            batch_size=batch_size,
                                            seed=1)

    return test


def get_paths(test_dir, dif='easy', n=1, shuffle=True):

    # List the file paths of all real and fake images in the test diretory
    real_ls = os.listdir(f'{test_dir}/real')
    fake_ls = os.listdir(f'{test_dir}/fake')

    # Sort the lists
    real_ls.sort()
    fake_ls.sort()

    # Get the full file paths
    real_ls = [test_dir + '/real/' + path for path in real_ls]
    fake_ls = [test_dir + '/fake/' + path for path in fake_ls]

    # Subset these lists based on the difficulty
    if dif == 'all':
        fake_d_ls = [i for i in fake_ls]
    else:
        fake_d_ls = [i for i in fake_ls if dif in i]
    real_d_ls = [i for i in real_ls]

    # Shuffle these lists
    random.seed(2)
    random.shuffle(fake_d_ls)
    random.shuffle(real_d_ls)

    # Take the first n of these lists
    fake_paths = fake_d_ls[:n]
    real_paths = real_d_ls[:n]
    return fake_paths, real_paths


class DsLoader():
    def __init__(self,
                 dir_path,
                 image_size=(64, 64),
                 problem='all',
                 full_label=False):
        self.dir_path = dir_path
        self.train_path = dir_path + '/train'
        self.val_path = dir_path + '/val'
        self.test_path = dir_path + '/test'
        self.image_size = image_size
        self.full_label = full_label

    def get_ds(self, split='train', batch_size=32, augment=False, n_repeats=1):
        if split == 'train':
            dir_path = self.train_path
        elif split == 'val':
            dir_path = self.val_path
        elif split == 'test':
            dir_path = self.test_path

        list_ds = tf.data.Dataset.list_files(dir_path + '/*/*.jpg')

        # Set `num_parallel_calls` so multiple images are loaded/processed
        # in parallel.
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        labeled_ds = list_ds.map(
            self.process_path,
            num_parallel_calls=self.AUTOTUNE,
        )

        ds = self.prepare_for_training(labeled_ds, batch_size, augment=augment, n_repeats=n_repeats)
        return ds

    def get_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)

        # second to last part is folder name 'fake' or 'real'
        label = parts[-2]

        # convert to binary 0 for 'real' and 1 for 'fake'
        short_lab = tf.Variable(0, tf.int16)
        if label == 'fake':
            short_lab = 1
        elif label == 'real':
            short_lab = 0

        # read the last 4 digits of the filename if fake
        if short_lab == 1:
            long_lab = tf.strings.split(parts[-1], '.')[-2]
            long_lab = tf.strings.split(long_lab, '_')[-1]
        else:
            long_lab = '0000'

        if self.full_label:
            long_lab = tf.strings.bytes_split(long_lab)
            long_lab = tf.strings.to_number(long_lab, tf.int32)
            return long_lab
        else:
            return short_lab

    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, self.image_size)

    def process_path(self, file_path):
        label = self.get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def prepare_for_training(self,
                             ds,
                             batch_size,
                             cache=True,
                             shuffle_buffer_size=1000,
                             augment=False,
                             n_repeats=0):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets
        # that don't fit in memory.
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Augment the images
        if augment:
            ds = ds.map(self.flip)
            ds = ds.map(self.rotate)

        # Repeat if required
        ds = ds.repeat(n_repeats)

        ds = ds.batch(batch_size)

        # `prefetch` lets the dataset fetch batches in the background while
        # the model is training
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)

        return ds

    def flip(self, x, y):
        """Flip augmentation

        Args:
            x: Image to flip

        Returns:
            Augmented image
        """
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)

        return x, y

    def rotate(self, x, y):
        """Rotation augmentation

        Args:
            x: Image

        Returns:
            Augmented image
        """

        x = tf.image.rot90(
            x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        return x, y
