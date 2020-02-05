"""
Implements a descriptor of image files.
"""

import cv2
import numpy


# Keeps doubling the given image size up to the point of having more pixels than <min_image_size>.
# The image aspect ratio is kept.
# Returns the resized image and the number of times it was increased.
def _increase_image_if_necessary(image, min_image_size):
    resize_count = 0

    while image.shape[0] * image.shape[1] < min_image_size:
        image = cv2.resize(image, (0, 0), fx=2, fy=2)
        resize_count = resize_count + 1

    return image, resize_count


# Keeps halving the given image size up to the point of having less pixels than <max_image_size>.
# The image aspect ratio is kept.
# Returns the resized image and the number of times it was downsized.
def _decrease_image_if_necessary(image, max_image_size):
    resize_count = 0

    while image.shape[0] * image.shape[1] > max_image_size:
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        resize_count = resize_count - 1

    return image, resize_count


# Detects SURF keypoints over the given image and describes them with RootSIFT.
# It extracts up to <kp_count> samples, obeying the eventually given image mask.
def surf_detect_rsift_describe(image, mask=None, kp_count=5000, min_image_size=100000, max_image_size=10000000,
                               hessian=1, eps=1e-7):
    # performs image resizing, if necessary
    image, resize_count = _increase_image_if_necessary(image, min_image_size)
    if resize_count == 0:  # there was no image increase, so tries to decrease it
        image, resize_count = _decrease_image_if_necessary(image, max_image_size)

    if mask is not None and resize_count != 0:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # image in grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # SURF detector
    surf_detector = cv2.xfeatures2d.SURF_create(hessian)

    # SIFT descriptor
    sift_detector = cv2.xfeatures2d.SIFT_create()

    # detects SURF keypoints over the given image
    keypoints = surf_detector.detect(image, mask)

    # orders the obtained interest points according to their response, and keeps only the top-<kp_count> ones
    keypoints = sorted(keypoints, key=lambda k: k.response, reverse=True)
    del keypoints[kp_count:]

    # describes the remaining interest points
    keypoints, descriptions = sift_detector.compute(image, keypoints)
    if len(keypoints) == 0:
        return [], []

    descriptions = descriptions / (descriptions.sum(axis=1, keepdims=True) + eps)
    descriptions = numpy.sqrt(descriptions).astype('float32')

    # adjusts the obtained keypoints according to the change of image size
    if resize_count != 0:
        for kp in keypoints:
            kp.pt = (kp.pt[0] / (2.0 ** resize_count), kp.pt[1] / (2.0 ** resize_count))
            kp.size = kp.size / (2.0 ** resize_count)

    # returns the obtained keypoints and their respective descriptions
    return keypoints, descriptions


# Stores the given interest points and their respective descriptions into the given file path.
# Format: x y s a r d+
def store_descriptions(keypoints, descriptions, description_file_path):
    if len(keypoints) != len(descriptions):
        raise Exception('[ERROR] Number of keypoints and descriptions is not the same. k:',
                        str(len(keypoints)) + ', d:', str(len(descriptions)) + '.')

    content = []

    if len(keypoints) > 0:
        content = numpy.zeros((len(keypoints), descriptions.shape[1] + 5), numpy.float32)

        for i in range(len(keypoints)):
            content[i][0] = keypoints[i].pt[0]
            content[i][1] = keypoints[i].pt[1]
            content[i][2] = keypoints[i].size
            content[i][3] = keypoints[i].angle
            content[i][4] = keypoints[i].response
            content[i][5:] = descriptions[i][:]

    numpy.save(description_file_path, content)


# Loads the image descriptions stored in the given file path.
# Returns the loaded interest points and their respective descriptions.
def load_descriptions(description_file_path):
    keypoints = []
    descriptions = []

    content = numpy.load(description_file_path)
    for c in content:
        x = c[0]
        y = c[1]
        size = c[2]
        angle = c[3]
        response = c[4]
        description = c[5:].astype(numpy.float32)

        keypoints.append(cv2.KeyPoint(x, y, size, angle, response))
        if len(descriptions) == 0:
            descriptions.append(description)
            descriptions = numpy.array(descriptions)
        else:
            descriptions = numpy.vstack((descriptions, description))

    return keypoints, descriptions
