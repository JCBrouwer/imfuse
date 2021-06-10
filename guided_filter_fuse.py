import argparse
from functools import partial
from time import time
from typing import List

import imageio
import numpy as np
import scipy.ndimage
import scipy.signal
from skimage.transform import resize

from util import average_kernel, box_filter, gaussian_kernel, laplacian_kernel, slow_guided_filter, slow_convolve

convolve = partial(scipy.signal.fftconvolve, mode="same")


def guided_filter(image, guide, radius, eps):
    h, w = guide.shape[:2]

    if image.ndim < guide.ndim:
        image = image[..., np.newaxis]

    ones = np.ones((h, w, 1))
    N = box_filter(ones, radius)

    mu_I = box_filter(guide, radius) / N
    mu_P = box_filter(image, radius) / N
    cov_IP = box_filter(guide * image, radius) / N - mu_I * mu_P
    var_I = box_filter(guide * guide, radius) / N - mu_I * mu_I

    A = cov_IP / (var_I + eps)
    b = mu_P - A * mu_I

    mean_A = box_filter(A, radius) / N
    mean_b = box_filter(b, radius) / N

    output = mean_A * guide + mean_b
    return output


def fuse_images(
    images: List[np.ndarray],
    guide_radius=7,
    guide_epsilon=0.05,
    average_radius=31,
    gaussian_radius=5,
    gaussian_sigma=5.0,
    slow=False,
):
    max_shape = max([im.shape for im in images])
    images = [resize(im, max_shape) for im in images]
    images = np.stack(images)
    if not (images.min() == 0 and images.max() == 1):
        images = images - images.min()
        images = images / images.max()
    if len(images.shape) < 4:
        images = images[..., None]

    images_base = (slow_convolve if slow else convolve)(images, average_kernel(average_radius))
    images_detail = images - images_base

    images_highpass = (slow_convolve if slow else convolve)(images, laplacian_kernel())

    saliency = (slow_convolve if slow else convolve)(
        np.abs(images_highpass), gaussian_kernel(gaussian_radius, gaussian_sigma)
    )
    saliency = np.sum(saliency, axis=-1)

    weight_map = np.argmax(saliency, axis=0)

    base_map, detail_map = [], []
    for i in range(len(images)):
        base_map.append(
            (slow_guided_filter if slow else guided_filter)(
                weight_map == i, images_base[i], radius=guide_radius, eps=guide_epsilon
            )
        )
        detail_map.append(
            (slow_guided_filter if slow else guided_filter)(
                weight_map == i, images_detail[i], radius=guide_radius, eps=guide_epsilon
            )
        )
    base_map, detail_map = np.stack(base_map), np.stack(detail_map)
    base_map /= base_map.sum(0)
    detail_map /= detail_map.sum(0)

    out_base = np.sum(base_map * images_base, axis=0)
    out_detail = np.sum(detail_map * images_detail, axis=0)

    fused = out_base + out_detail
    return fused.clip(0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=str)
    parser.add_argument("-r", "--guide_radius", type=int, default=7)
    parser.add_argument("-e", "--guide_epsilon", type=float, default=0.05)
    parser.add_argument("-ar", "--average_radius", type=int, default=31)
    parser.add_argument("-gr", "--gaussian_radius", type=int, default=5)
    parser.add_argument("-gs", "--gaussian_sigma", type=float, default=5.0)
    args = parser.parse_args()

    images = [imageio.imread(img_file) for img_file in args.images]
    fused = fuse_images(
        images, args.guide_radius, args.guide_epsilon, args.average_radius, args.gaussian_radius, args.gaussian_sigma
    )

    import matplotlib.pyplot as plt

    plt.imshow(fused.squeeze())
    plt.axis("off")
    plt.tight_layout()
    plt.show()
