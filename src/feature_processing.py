'''
A set of functions to help with feature creation
'''

from tqdm import tqdm_notebook as tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from image_processing import convert_to_gray


def create_features(img, feature_params):

    if feature_params['color'] == 'gray':
        make_gray = True
    else:
        make_gray = False

    if 'hog' in feature_params:
        include_hog_features = True
        pixels_per_cell = feature_params['hog']
    else:
        include_hog_features = False

    include_loc_features = 'loc' in feature_params

    if len(img.shape) < 4:
        single_img = True
    else:
        single_img = False

    if make_gray:
        gray_img = convert_to_gray(img)
        pixel_features = gray_img.flatten()
    else:
        # i included to make sure reshape works with multiple images
        i = (not single_img) * 1
        pixel_features = img.reshape(np.prod(img.shape[:2 + i]), 3)

    out = pixel_features.reshape(len(pixel_features), -1)

    if include_loc_features:
        loc_features = create_loc_features(img.shape, single_image=single_img)
        out = np.c_[out, loc_features]

    if include_hog_features:

        hog_features = create_hog_features(img,
                                           pixels_per_cell=pixels_per_cell)
        out = np.c_[out, hog_features]

    return out


def create_hog_features(imgs,
                        orientations=8,
                        pixels_per_cell=(20, 20),
                        gray_input=False,
                        visualize=False,
                        multichannel=False):

    # Check if grayscale input
    multichannel = not gray_input

    # If an array of more than 1 image
    if len(imgs.shape) == 4 + gray_input * 1:
        output = np.zeros(shape=(imgs.shape[0], imgs.shape[1] * imgs.shape[2],
                                 orientations))

        for i, img in enumerate(imgs):
            fd_rr = create_hog_features(img,
                                        orientations=orientations,
                                        pixels_per_cell=pixels_per_cell,
                                        multichannel=multichannel,
                                        visualize=visualize)
            output[i] = fd_rr
        return output.reshape(output.shape[0] * output.shape[1], orientations)

    # Else if only one image
    else:
        img = imgs

        # Check that the image is square
        assert img.shape[0] == img.shape[1]

        # If want to display the output
        if visualize:
            fd, hog_image = hog(
                imgs,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=(1, 1),
                visualize=visualize,
                multichannel=multichannel,
                feature_vector=False,
            )

            plt.imshow(hog_image, cmap='gray')

        # Don't want to display the output
        elif not visualize:
            fd = hog(
                imgs,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=(1, 1),
                visualize=visualize,
                multichannel=multichannel,
                feature_vector=False,
            )

        fd_r = fd.reshape(fd.shape[0] * fd.shape[1], orientations)
        fd_rr = fd_r.repeat(img.shape[0]**2 / fd_r.shape[0], axis=0)
        return fd_rr


def create_loc_features(array_shape, single_image=False):
    if single_image:
        array_shape = (1, ) + array_shape

    x, y = create_pixel_loc(array_shape[1:3])
    x = np.tile(x, array_shape[0]).reshape(-1, 1)
    y = np.tile(y, array_shape[0]).reshape(-1, 1)

    return np.hstack((x, y))


def create_pixel_loc(shape):
    row_ = np.array(range(0, shape[0] * shape[1]))
    y = row_.reshape(shape[0], shape[1]).T.flatten() % shape[1]
    x = row_ % shape[1]
    return x / shape[1], y / shape[1]
