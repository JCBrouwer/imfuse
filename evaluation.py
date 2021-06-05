import itertools
import warnings
from glob import glob

import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import normalized_mutual_information, structural_similarity
from skimage.transform import resize
from tqdm import tqdm

from guided_filter_fuse import fuse_images as guided_filter_fuse
from util import box_filter
from wavelet_fuse import WAVELET_CHOICES
from wavelet_fuse import fuse_images as wavelet_fuse

warnings.filterwarnings("ignore", "invalid value encountered in true_divide")


def normalize(x):
    x = x - np.nanmin(x)
    return x / np.nanmax(x)


def var(img, radius):
    h, w, c = img.shape
    N = box_filter(np.ones((h, w, 1)), radius)
    mu = box_filter(img, radius) / N
    var = box_filter(img ** 2, radius) / N - mu ** 2
    return var


def yang_Q_y(ref_A, ref_B, fused, ks=7, figname=None):
    ref_A = resize(ref_A, fused.shape)  # handle channel mismatches
    ref_B = resize(ref_B, fused.shape)

    _, ssim_A = structural_similarity(ref_A, fused, win_size=ks, channel_axis=2, full=True)
    var_A = var(ref_A, radius=int(ks - 1) // 2)

    _, ssim_B = structural_similarity(ref_B, fused, win_size=ks, channel_axis=2, full=True)
    var_B = var(ref_B, radius=int(ks - 1) // 2)

    lambda_w = var_A / (var_A + var_B)

    Q_y = np.zeros_like(fused)
    Q_y = lambda_w * ssim_A + (1 - lambda_w) * ssim_B
    use_max = lambda_w < 0.75
    Q_y[use_max] = np.maximum(ssim_A[use_max], ssim_B[use_max])

    if figname is not None:
        _, ax = plt.subplots(3, 3, figsize=(15, 15))
        [axi.set_axis_off() for axi in ax.ravel()]
        ax[0, 0].imshow(ref_A.squeeze())
        ax[0, 0].set_title("A")
        ax[0, 1].imshow(ref_B.squeeze())
        ax[0, 1].set_title("B")
        ax[0, 2].imshow(fused.squeeze())
        ax[0, 2].set_title("Fused")
        ax[1, 0].imshow(ssim_A.squeeze() / 2 + 0.4999)
        ax[1, 0].set_title(r"$SSIM_A$")
        ax[1, 1].imshow(ssim_B.squeeze() / 2 + 0.4999)
        ax[1, 1].set_title(r"$SSIM_B$")
        ax[1, 2].imshow(np.nan_to_num(Q_y, nan=1).squeeze() / 2 + 0.4999)
        ax[1, 2].set_title(r"$Q_y$")
        ax[2, 0].imshow(normalize(var_A).squeeze())
        ax[2, 0].set_title(r"$\sigma^2_B$")
        ax[2, 1].imshow(normalize(var_B).squeeze())
        ax[2, 1].set_title(r"$\sigma^2_B$")
        plt.tight_layout()
        plt.savefig("results/SSIM_" + figname + ".jpg")
        plt.close()

    return np.nanmean(Q_y)


def uiqi(img1, img2, ks=15):
    radius = int(ks - 1) // 2

    h, w, c = img1.shape
    N = box_filter(np.ones((h, w, 1)), radius)

    mu1 = box_filter(img1, radius) / N
    mu2 = box_filter(img2, radius) / N

    mu1sq = mu1 ** 2
    mu2sq = mu2 ** 2

    var1 = box_filter(img1 ** 2, radius) / N - mu1sq
    var2 = box_filter(img2 ** 2, radius) / N - mu2sq
    cov12 = box_filter(img1 * img2, radius) / N - mu1 * mu2

    quality = (4 * cov12 * mu1 * mu2) / ((var1 + var2) * (mu1sq + mu2sq))

    return np.nanmean(quality)


def cov(x, y):
    return np.sum((x - x.mean()) * (y - y.mean())) / (len(x) - 1)


def cvejic_Q_c(ref_A, ref_B, fused):
    cov_A = cov(ref_A, fused)
    cov_B = cov(ref_B, fused)
    mu = np.clip(cov_A / (cov_A + cov_B), 0, 1)
    return mu * uiqi(ref_A, fused) + (1 - mu) * uiqi(ref_B, fused)


def nmi(inputs, fused):
    return np.mean([normalized_mutual_information(input.ravel(), fused.ravel()) for input in inputs])


def ssim(inputs, fused):
    return np.mean([structural_similarity(input, fused, channel_axis=2) for input in inputs])


if __name__ == "__main__":
    pairs = []

    pairs.append(glob("data/stained-glass*.jpg"))
    glass_A, glass_B = [np.asfarray(imageio.imread(file)) / 255 for file in glob("data/stained-glass*.jpg")]
    for name, fuse in ("guided", guided_filter_fuse), ("wavelet", wavelet_fuse):
        print(f"\n stained glass {name}")
        glass_fused = fuse([glass_A, glass_B])

        _, ax = plt.subplots(1, 3, figsize=(15, 5))
        [axi.set_axis_off() for axi in ax.ravel()]
        ax[0].imshow(glass_A)
        ax[1].imshow(glass_B)
        ax[2].imshow(glass_fused)
        plt.tight_layout()
        plt.savefig(f"results/stained_glass_{name}.jpg")
        plt.close()

        print("NMI", nmi([glass_A, glass_B], glass_fused))
        print("Q_c", cvejic_Q_c(glass_A, glass_B, glass_fused))
        print("Q_y", yang_Q_y(glass_A, glass_B, glass_fused, figname=f"stained_glass_{name}"))

    pairs.append(glob("data/clock*.jpg"))
    clock_A, clock_B = [np.asfarray(imageio.imread(file)) / 255 for file in glob("data/clock*.jpg")]
    for name, fuse in ("guided", guided_filter_fuse), ("wavelet", wavelet_fuse):
        print(f"\n clock {name}")
        clock_fused = fuse([clock_A, clock_B])

        _, ax = plt.subplots(1, 3, figsize=(15, 5))
        [axi.set_axis_off() for axi in ax.ravel()]
        ax[0].imshow(clock_A)
        ax[1].imshow(clock_B)
        ax[2].imshow(clock_fused)
        plt.tight_layout()
        plt.savefig(f"results/clock_{name}.jpg")
        plt.close()

        print("NMI", nmi([clock_A, clock_B], clock_fused))
        print("Q_c", cvejic_Q_c(clock_A, clock_B, clock_fused))
        print("Q_y", yang_Q_y(clock_A, clock_B, clock_fused, figname=f"clock_{name}"))

    for dir in glob("data/brains/*"):
        if "README" in dir:
            continue
        pairs.append(glob(f"{dir}/*.jpg"))
        for name, fuse in ("guided", guided_filter_fuse), ("wavelet", wavelet_fuse):
            print("\n", dir, name)
            brain_A, brain_B = [np.asfarray(imageio.imread(file)) / 255 for file in glob(f"{dir}/*.jpg")]
            brain_A = brain_A[..., None]
            brain_fused = fuse([brain_A, brain_B])

            _, ax = plt.subplots(1, 3, figsize=(15, 5))
            [axi.set_axis_off() for axi in ax.ravel()]
            ax[0].imshow(resize(brain_A, brain_fused.shape))
            ax[1].imshow(brain_B)
            ax[2].imshow(brain_fused)
            plt.tight_layout()
            plt.savefig(f"results/{dir.replace('data/','').replace('/','_')}_{name}.jpg")
            plt.close()

            print("NMI", nmi([brain_A, brain_B], brain_fused))
            print("Q_c", cvejic_Q_c(brain_A, brain_B, brain_fused))
            print(
                "Q_y",
                yang_Q_y(brain_A, brain_B, brain_fused, figname=f"{dir.replace('data/','').replace('/','_')}_{name}",),
            )

    horse_images = [np.asfarray(imageio.imread(file)) / 255 for file in glob("data/horse/*.jpg")]
    horse_fused = guided_filter_fuse(horse_images)

    print("\nhorse")
    print("normalized mutual information", nmi(horse_images, horse_fused))
    print("structural similarity", ssim(horse_images, horse_fused))

    _, ax = plt.subplots(1, 4, figsize=(20, 5))
    [axi.set_axis_off() for axi in ax.ravel()]
    ax[0].imshow(horse_images[0])
    ax[1].imshow(horse_images[1])
    ax[2].imshow(horse_images[2])
    ax[3].imshow(horse_fused)
    plt.tight_layout()
    plt.savefig("results/horse_guided_filter.jpg")

    print("\n\n\nWAVELET KERNEL SIZE SHOWDOWN!")

    pairs = [[np.asfarray(imageio.imread(file)) / 255 for file in pair] for pair in pairs]

    scores = []
    for wavelet, kernel_size in tqdm(list(itertools.product(WAVELET_CHOICES, range(3, 41, 2)))):
        NMIs = []
        Q_cs = []
        Q_ys = []
        try:
            for A, B in pairs:
                if len(A.shape) == 2:
                    A = A[..., None]
                F = wavelet_fuse([A, B], kernel_size=int(kernel_size), wavelet=wavelet)

                NMIs.append(nmi([A, B], F))
                Q_cs.append(cvejic_Q_c(A, B, F))
                Q_ys.append(yang_Q_y(A, B, F, figname=None))

            print(f"\n{wavelet}", kernel_size)
            print("NMI", np.mean(NMIs))
            print("Q_c", np.mean(Q_cs))
            print("Q_y", np.mean(Q_ys))
            scores.append([(wavelet, kernel_size), [np.mean(NMIs), np.mean(Q_cs), np.mean(Q_ys)]])
        except:
            print(f"\n{wavelet}", kernel_size, "FAILED")
            scores.append([(wavelet, kernel_size), [-1, -1, -1]])

    print("\n\n\nTop 10 wavelet + kernel size settings:")
    for (wavelet, kernel_size), (mnmi, qc, qy) in list(sorted(scores, key=lambda x: x[1], reverse=True))[:10]:
        print(wavelet, kernel_size)
        print("NMI", mnmi)
        print("Q_c", qc)
        print("Q_y", qy)
        print()
