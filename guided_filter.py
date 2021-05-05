import argparse
from time import time
from typing import List

import numpy as np
from PIL import Image


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


def convolve(image, kernel):
    assert (
        image.ndim == kernel.ndim == 4
    ), "The image and convolutional kernel must both be 4 dimensional (batch, height, width, channels)"

    _, kh, kw, _ = kernel.shape
    b, h, w, c = image.shape

    image = pad(image, padding=np.floor(kw / 2))

    output = np.zeros((b, h, w, c))

    for y in range(h):
        for x in range(w):
            output[:, y, x] = np.sum(kernel * image[:, y : y + kh, x : x + kw], axis=(1, 2))

    return output


def guided_filter(image, guide, radius, eps):
    assert image.ndim == 2, "Input to guided filter must have a single channel"
    assert guide.ndim == 3, "Guide in guided filter must have 3 channels"

    ks = 2 * radius + 1
    ka = ks ** 2
    h, w, c = guide.shape

    image = pad(image, radius)
    guide = pad(guide, radius)
    output = pad(np.zeros((h, w)), radius)

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

    return output[radius:-radius, radius:-radius]


def info(x):
    print(x.shape, x.min(), np.median(x), x.max())


def fuse_images(images: List[np.ndarray], guide_radius, guide_epsilon, average_radius, gaussian_radius, gaussian_sigma):
    images = np.stack(images)
    if not (images.min() == 0 and images.max() == 1):
        images -= images.min()
        images /= images.max()

    print("average filter")
    t = time()
    images_base = convolve(images, average_kernel(average_radius))
    # info(images_base)
    images_detail = images - images_base
    # info(images_detail)
    print("Took", time() - t)
    t = time()

    print("saliency")
    images_highpass = convolve(images, laplacian_kernel())
    # info(images_highpass)
    saliency = convolve(np.abs(images_highpass), gaussian_kernel(gaussian_radius, gaussian_sigma))
    saliency = np.sum(saliency, axis=-1)
    # info(saliency)

    weight_map = np.zeros_like(saliency)
    max_idxs = np.argmax(saliency, axis=0)
    weight_map[max_idxs] = 1
    # info(weight_map)
    print("Took", time() - t)
    t = time()

    print("guided filter")
    base_map, detail_map = [], []
    for i in range(len(images)):
        base_map.append(guided_filter(weight_map[i], images_base[i], radius=guide_radius, eps=guide_epsilon))
        detail_map.append(guided_filter(weight_map[i], images_detail[i], radius=guide_radius, eps=guide_epsilon))
    base_map, detail_map = np.stack(base_map)[..., np.newaxis], np.stack(detail_map)[..., np.newaxis]

    base_map /= base_map.sum(0)  # weight maps must sum to 1 over the batch dimension
    detail_map /= detail_map.sum(0)
    # info(base_map)
    # info(detail_map)
    print("Took", time() - t)
    t = time()

    out_base = np.sum(base_map * images_base, axis=0)
    out_detail = np.sum(detail_map * images_detail, axis=0)
    # info(out_base)
    # info(out_detail)

    fused = out_base + out_detail
    # info(fused)

    return fused


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=str)
    parser.add_argument("-r", "--guide_radius", type=int, default=7)
    parser.add_argument("-e", "--guide_epsilon", type=float, default=3.0)
    parser.add_argument("-ar", "--average_radius", type=int, default=31)
    parser.add_argument("-gr", "--gaussian_radius", type=int, default=5)
    parser.add_argument("-gs", "--gaussian_sigma", type=float, default=5.0)
    args = parser.parse_args()

    images = []
    for img_file in args.images:
        images.append(Image.open(img_file))

    fused = fuse_images(
        images, args.guide_radius, args.guide_epsilon, args.average_radius, args.gaussian_radius, args.gaussian_sigma
    )

    import matplotlib.pyplot as plt

    plt.imshow(fused)
    plt.show()

