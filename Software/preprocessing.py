"""
preprocessing.py
----------------
Image loading and preprocessing.
"""

import cv2
import os


def load_image(image_path):

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Failed to read image: {image_path}"
        )

    return image


def resize_image(
    image,
    max_dimension=1200,
):
    """
    Resize while preserving aspect ratio.
    """

    height, width = image.shape[:2]

    largest = max(height, width)

    if largest <= max_dimension:
        return image

    scale = (
        max_dimension /
        float(largest)
    )

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def to_grayscale(image):

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


def equalize_histogram(gray_image):

    return cv2.equalizeHist(
        gray_image
    )


def preprocess(
    image_path,
    max_dimension=1200,
    use_histogram_equalization=True,
):

    original = load_image(
        image_path
    )

    resized = resize_image(
        original,
        max_dimension=max_dimension,
    )

    gray = to_grayscale(
        resized
    )

    if use_histogram_equalization:

        gray = equalize_histogram(
            gray
        )

    return resized, gray