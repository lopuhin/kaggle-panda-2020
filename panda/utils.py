from functools import partial

import cv2
import numpy as np
import scipy as sp
from sklearn import metrics


def crop_white(image: np.ndarray, value: int = 255) -> np.ndarray:
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    ys, = (image.min((1, 2)) < value).nonzero()
    xs, = (image.min(0).min(1) < value).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return image
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping.
    https://stackoverflow.com/a/47248339
    """
    assert image.shape[2] == 3
    assert image.dtype == np.uint8

    height, width = image.shape[:2]  # image shape has 3 dimensions
    # getRotationMatrix2D needs coordinates in reverse order (width, height)
    # compared to shape
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo)
    # and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(
        image, rotation_mat, (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255),
    )
    return rotated_mat


class OptimizedRounder:
    """ Based on
    https://www.kaggle.com/abhishek/optimizer-for-quadratic-weighted-kappa
    """
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.coef_ = None

    def _kappa_loss(self, coef, X, y):
        try:
            X_p = np.digitize(X, coef)
        except ValueError:
            return 1
        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        import time
        t0 = time.time()
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5 + i for i in range(self.n_classes - 1)]
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method='nelder-mead')['x']
        print('fit done in ', time.time() - t0)

    def predict(self, X):
        return np.digitize(X, self.coef_)


def load_weights(model, state):
    weights = state['weights']
    if all(key.startswith('module.') for key in weights):
        for key in list(weights):
            weights[key[len('module.'):]] = weights.pop(key)
    model.load_state_dict(weights)
