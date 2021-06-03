import argparse

import matplotlib.pyplot as plt
import numpy as np
import pywt
from PIL import Image
from scipy.ndimage import maximum_filter1d, median_filter
from skimage.transform import resize

from util import box_filter


def decision_map(img1, img2, ks):
    max1 = maximum_filter1d(maximum_filter1d(img1, axis=1, size=ks, mode="mirror"), axis=1, size=ks, mode="mirror")
    max2 = maximum_filter1d(maximum_filter1d(img2, axis=1, size=ks, mode="mirror"), axis=1, size=ks, mode="mirror")
    return max1 > max2


def majority_filter(map, ks):
    return box_filter(map, int((ks - 1) / 2)) > (ks ** 2) / 2


def fuse(img1, img2, ks=5):
    binary_map = decision_map(img1, img2, ks)
    binary_map = 1 - majority_filter(1 - majority_filter(binary_map, ks), ks)
    binary_map = binary_map.astype(np.bool)
    img1[~binary_map] = 0
    img2[binary_map] = 0
    return img1 + img2


def fuse_images(x, y, kernel_size=5, wavelet="haar"):
    x, y = x.T, y.T
    if not (x.min() == 0 == y.min() and x.max() == 1 == y.max()):
        x = x - x.min()
        x = x / x.max()
        y = y - y.min()
        y = y / y.max()

    stack = []
    while x.shape[-2:] >= (2 * kernel_size, 2 * kernel_size):
        x, (hx, vx, dx) = pywt.dwt2(x, wavelet)
        y, (hy, vy, dy) = pywt.dwt2(y, wavelet)
        stack.append(fuse(dx, dy, ks=kernel_size))
        stack.append(fuse(vx, vy, ks=kernel_size))
        stack.append(fuse(hx, hy, ks=kernel_size))

    fused = fuse(x, y, kernel_size)

    while len(stack) > 0:
        fused = resize(fused, stack[-1].shape)
        fused = pywt.idwt2((fused, (stack.pop(-1), stack.pop(-1), stack.pop(-1))), wavelet)

    fused = np.clip(fused, 0, 1)
    return fused.T


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs=2, type=str)
    parser.add_argument("-ks", "--kernel_size", type=int, default=5)
    parser.add_argument(
        "-w",
        "--wavelet",
        type=str,
        default="haar",
        choices=[
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
            "coif1",
            "coif2",
            "coif3",
            "coif4",
            "coif5",
            "coif6",
            "coif7",
            "coif8",
            "coif9",
            "coif10",
            "coif11",
            "coif12",
            "coif13",
            "coif14",
            "coif15",
            "coif16",
            "coif17",
            "db1",
            "db2",
            "db3",
            "db4",
            "db5",
            "db6",
            "db7",
            "db8",
            "db9",
            "db10",
            "db11",
            "db12",
            "db13",
            "db14",
            "db15",
            "db16",
            "db17",
            "db18",
            "db19",
            "db20",
            "db21",
            "db22",
            "db23",
            "db24",
            "db25",
            "db26",
            "db27",
            "db28",
            "db29",
            "db30",
            "db31",
            "db32",
            "db33",
            "db34",
            "db35",
            "db36",
            "db37",
            "db38",
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
            "sym2",
            "sym3",
            "sym4",
            "sym5",
            "sym6",
            "sym7",
            "sym8",
            "sym9",
            "sym10",
            "sym11",
            "sym12",
            "sym13",
            "sym14",
            "sym15",
            "sym16",
            "sym17",
            "sym18",
            "sym19",
            "sym20",
        ],
    )
    args = parser.parse_args()

    images = []
    for img_file in args.images:
        images.append(np.asfarray(Image.open(img_file)))

    fused = fuse_images(*images, kernel_size=args.kernel_size, wavelet=args.wavelet)

    print(fused.min(), fused.mean(), fused.max())

    import matplotlib.pyplot as plt

    plt.imshow(fused)
    plt.show()
