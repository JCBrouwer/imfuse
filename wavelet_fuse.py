import argparse

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.ndimage import maximum_filter1d
from skimage.transform import resize

from util import box_filter, timeout

WAVELET_CHOICES = [
    "bior1.1",
    "bior1.3",
    "bior1.5",
    "bior2.2",
    "bior2.4",
    "bior2.6",
    "bior2.8",
    "bior3.1",
    "bior3.3",
    "bior3.5",
    "bior3.7",
    "bior3.9",
    "bior4.4",
    "bior5.5",
    "bior6.8",
    *[f"coif{i}" for i in range(1, 17)],
    *[f"db{i}" for i in range(1, 38)],
    "dmey",
    "haar",
    "rbio1.1",
    "rbio1.3",
    "rbio1.5",
    "rbio2.2",
    "rbio2.4",
    "rbio2.6",
    "rbio2.8",
    "rbio3.1",
    "rbio3.3",
    "rbio3.5",
    "rbio3.7",
    "rbio3.9",
    "rbio4.4",
    "rbio5.5",
    "rbio6.8",
    *[f"sym{i}" for i in range(2, 20)],
]


def decision_map(img1, img2, ks):
    # maximum filter is separable so we perform two 1D filters in sequence
    max1 = maximum_filter1d(maximum_filter1d(img1, axis=1, size=ks, mode="mirror"), axis=2, size=ks, mode="mirror")
    max2 = maximum_filter1d(maximum_filter1d(img2, axis=1, size=ks, mode="mirror"), axis=2, size=ks, mode="mirror")
    return max1 > max2


def majority_filter(map, ks):
    # because the the map is binary, comparing the sum with the area of the kernel is equivalent to majority vote
    radius = int((ks - 1) / 2)
    return box_filter(map.T, radius).T > (ks ** 2) / 2


def fuse(img1, img2, ks=5):
    binary_map = decision_map(img1, img2, ks)
    binary_map = 1 - majority_filter(1 - majority_filter(binary_map, ks), ks)
    binary_map = binary_map.astype(bool)
    img1[~resize(binary_map, img1.shape)] = 0  # resize deals with mismatching number of channels
    img2[resize(binary_map, img2.shape)] = 0
    return img1 + img2


@timeout(error_message="Wavelet fusion timed out. You probably need a larger kernel size for this wavelet")
def fuse_images(images, kernel_size=37, wavelet="sym13", max_depth=999):
    # ensure images have the right shape for the following operations
    for i in range(2):
        if len(images[i].shape) < 3:
            images[i] = images[i][..., None]
        images[i] = images[i].T
        if not (images[i].min() == 0 and images[i].max() == 1):
            images[i] = images[i] - images[i].min()
            images[i] = images[i] / images[i].max()
    x, y = images

    # move down the wavelet pyramid, fuse along the way
    depth = 1
    stack = []
    while x.shape[-2:] >= (2 * kernel_size, 2 * kernel_size):
        if depth > max_depth:
            break
        x, (hx, vx, dx) = pywt.dwt2(x, wavelet)
        y, (hy, vy, dy) = pywt.dwt2(y, wavelet)
        stack.append(fuse(dx, dy, ks=kernel_size))
        stack.append(fuse(vx, vy, ks=kernel_size))
        stack.append(fuse(hx, hy, ks=kernel_size))
        depth += 1

    # fuse the DC offset
    fused = fuse(x, y, kernel_size)

    # inverse wavelet transform back up
    while len(stack) > 0:
        fused = resize(fused, stack[-1].shape)
        fused = pywt.idwt2((fused, (stack.pop(-1), stack.pop(-1), stack.pop(-1))), wavelet)

    # clip to image range and resize to original shape
    fused = np.clip(fused, 0, 1)
    fused = resize(fused, (fused.shape[0], *images[0].shape[-2:]))
    return fused.T


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs=2, type=str)
    parser.add_argument("-w", "--wavelet", type=str, default="db2", choices=WAVELET_CHOICES)
    parser.add_argument("-ks", "--kernel_size", type=int, default=3)
    parser.add_argument("-d", "--max_depth", type=int, default=999)
    parser.add_argument("-s", "--simple", action="store_true")
    args = parser.parse_args()

    images = []
    for img_file in args.images:
        images.append(np.asfarray(imageio.imread(img_file)))

    fused = fuse_images(images, kernel_size=args.kernel_size, wavelet=args.wavelet)

    import matplotlib.pyplot as plt

    plt.imshow(fused.squeeze())
    plt.axis("off")
    plt.tight_layout()
    plt.show()
