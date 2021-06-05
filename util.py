import errno
import os
import signal
from functools import wraps

import numpy as np
from skimage.transform import resize


class TimeoutError(Exception):
    pass


def timeout(seconds=5, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


def diff_x(x, r):
    left = x[r : 2 * r + 1]
    middle = x[2 * r + 1 :] - x[: -2 * r - 1]
    right = x[-1:] - x[-2 * r - 1 : -r - 1]
    return np.concatenate([left, middle, right], axis=0)


def diff_y(x, r):
    left = x[:, r : 2 * r + 1]
    middle = x[:, 2 * r + 1 :] - x[:, : -2 * r - 1]
    right = x[:, -1:] - x[:, -2 * r - 1 : -r - 1]
    return np.concatenate([left, middle, right], axis=1)


def box_filter(x, r):
    return diff_y(diff_x(x.cumsum(axis=0), r).cumsum(axis=1), r)


def average_kernel(r):
    return np.ones((1, r, r, 1)) / r ** 2


def gaussian_kernel(radius, sigma):
    kernel = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 / sigma ** 2 * kernel ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel[:, np.newaxis] * kernel[np.newaxis, :]
    return kernel[np.newaxis, ..., np.newaxis]


def laplacian_kernel():
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    return kernel[np.newaxis, ..., np.newaxis]


def pad(image, padding):
    padding = int(padding)
    if image.ndim == 2:
        h, w = image.shape
        result = np.zeros((h + padding * 2, w + padding * 2))
        result[padding:-padding, padding:-padding] = image
    if image.ndim == 3:
        h, w, c = image.shape
        result = np.zeros((h + padding * 2, w + padding * 2, c))
        result[padding:-padding, padding:-padding] = image
    elif image.ndim == 4:
        b, h, w, c = image.shape
        result = np.zeros((b, h + padding * 2, w + padding * 2, c))
        result[:, padding:-padding, padding:-padding] = image
    return result


def info(x):
    print(x.shape, x.min(), np.median(x), x.max())


######################################################################
############################# GRAVEYARD ##############################
######################################################################
#### naive implementations that were orders of magnitude too slow ####
######################################################################


def slow_convolve(image, kernel):
    assert (
        image.ndim == kernel.ndim == 4
    ), "The image and convolutional kernel must both be 4 dimensional (batch, height, width, channels)"

    _, kh, kw, _ = kernel.shape
    b, h, w, c = image.shape

    image = pad(image, padding=np.floor(kw / 2))

    output = np.zeros((b, h, w, c))

    # TODO get rid of these for loops
    for y in range(h):
        for x in range(w):
            output[:, y, x] = np.sum(kernel * image[:, y : y + kh, x : x + kw], axis=(1, 2))

    return output


def slow_guided_filter(image, guide, radius, eps):
    assert image.ndim == 2, "Input to guided filter must have a single channel"
    assert guide.ndim == 3, "Guide in guided filter must have 3 channels"

    ks = 2 * radius + 1
    ka = ks ** 2
    h, w, c = guide.shape

    image = pad(image, radius)
    guide = pad(guide, radius)
    output = pad(np.zeros((h, w)), radius)

    # TODO get rid of these for loops
    for y in range(h):
        for x in range(w):
            I = guide[y : y + ks, x : x + ks]
            P = image[y : y + ks, x : x + ks, np.newaxis]

            mu = np.mean(I, axis=(0, 1))
            mu_p = np.mean(P)
            cov = np.cov(I.reshape(ks * ks, c).T)

            a = (cov + eps * np.eye(c)) @ (np.sum(I * P, axis=(0, 1)) / ka - mu * mu_p)
            b = mu_p - a.T @ mu

            # TODO I think we need to average a & b, not the output as we're doing now

            output[y : y + ks, x : x + ks] += (I @ a + b) / ka

    return output[radius:-radius, radius:-radius][..., np.newaxis]


def pad2(image, padding):
    padding = int(padding)
    if image.ndim == 2:
        w, h = image.shape
        result = resize(image.T, (h + padding * 2, w + padding * 2)).T
        result[padding:-padding, padding:-padding] = image
    if image.ndim == 3:
        c, w, h = image.shape
        result = resize(image.T, (h + padding * 2, w + padding * 2)).T
        result[:, padding:-padding, padding:-padding] = image
    return result


def slow_decision_map(img1, img2, ks, reduce=np.max):
    ks = int(ks)
    p = int((ks - 1) / 2)
    img1_p = pad2(img1, padding=p)
    img2_p = pad2(img2, padding=p)

    output = np.zeros(img1.shape)

    _, w, h = img1.shape
    for i in range(w):
        for j in range(h):
            w1 = reduce(np.abs(img1_p[:, i : i + ks - 1, j : j + ks - 1]), axis=(1, 2))
            w2 = reduce(np.abs(img2_p[:, i : i + ks - 1, j : j + ks - 1]), axis=(1, 2))
            output[:, i, j] = w1 > w2

    return output


def slow_majority_filter(map, ks):
    ks = int(ks)
    map_p = pad2(map, padding=int((ks - 1) / 2))

    output = np.zeros(map.shape)

    c, w, h = map.shape
    for i in range(w):
        for j in range(h):
            w1 = map_p[:, i : i + ks - 1, j : j + ks - 1]
            output[:, i, j] = w1.sum((1, 2)) > ks ** 2 / 2

    return output
