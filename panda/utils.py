import cv2
import numpy as np


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
