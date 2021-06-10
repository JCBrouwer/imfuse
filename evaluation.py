import itertools
import os
import warnings
from functools import partial
from glob import glob
from time import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        pltssimA = ssim_A
        pltssimB = ssim_B
        pltQy = Q_y
        if pltssimA.shape[-1] != 3:
            pltssimA = np.concatenate([pltssimA] * 3, axis=-1)
        if pltssimB.shape[-1] != 3:
            pltssimB = np.concatenate([pltssimB] * 3, axis=-1)
        if pltQy.shape[-1] != 3:
            pltQy = np.concatenate([pltQy] * 3, axis=-1)

        _, ax = plt.subplots(1, 3, figsize=(15, 5))
        [axi.set_axis_off() for axi in ax.ravel()]
        # ax[0, 0].imshow(ref_A.squeeze())
        # ax[0, 0].set_title("A")
        # ax[0, 1].imshow(ref_B.squeeze())
        # ax[0, 1].set_title("B")
        # ax[0, 2].imshow(fused.squeeze())
        # ax[0, 2].set_title("Fused")
        ax[0].imshow(pltssimA.squeeze() / 2 + 0.4999)
        ax[0].set_title(r"$SSIM_A$")
        ax[1].imshow(pltssimB.squeeze() / 2 + 0.4999)
        ax[1].set_title(r"$SSIM_B$")
        ax[2].imshow(np.nan_to_num(pltQy, nan=1).squeeze() / 2 + 0.4999)
        ax[2].set_title(r"$Q_y$")
        # ax[2, 0].imshow(normalize(var_A).squeeze())
        # ax[2, 0].set_title(r"$\sigma^2_B$")
        # ax[2, 1].imshow(normalize(var_B).squeeze())
        # ax[2, 1].set_title(r"$\sigma^2_B$")
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
    if not os.path.exists("results/time-analysis.csv"):
        results = []
        for size in [128, 256, 512, 724, 1024]:  # , 1448, 2048]:

            A = np.random.randn(size, size, 3)
            B = np.random.randn(size, size, 3)

            for name, fuse in (("guided_filter", guided_filter_fuse), ("wavelet", wavelet_fuse)):
                times = []
                for _ in range(100):
                    t = time()
                    fuse([A, B])
                    times.append(time() - t)

                results.append(
                    {
                        "pixels": 3 * size ** 2,
                        "fuse": name,
                        "time": 1000 * np.mean(times),
                        "time_std": 1000 * np.std(times),
                    }
                )
                print(results[-1])

        for name, fuse in (
            ("slow_guided_filter", partial(guided_filter_fuse, slow=True)),
            ("slow_wavelet", partial(wavelet_fuse, slow=True)),
        ):
            A = np.random.randn(256, 256, 3)
            B = np.random.randn(256, 256, 3)

            times = []
            for _ in range(3):
                t = time()
                fuse([A, B])
                times.append(time() - t)

            results.append(
                {"pixels": 3 * 256 ** 2, "fuse": name, "time": 1000 * np.mean(times), "time_std": 1000 * np.std(times)}
            )
            print(results[-1])

        # from matlab (see fuse.m)
        results += [
            {"pixels": 3 * 128 ** 2, "fuse": "wfusimg", "time": 35, "time_std": 18},
            {"pixels": 3 * 256 ** 2, "fuse": "wfusimg", "time": 72, "time_std": 4},
            {"pixels": 3 * 512 ** 2, "fuse": "wfusimg", "time": 183, "time_std": 11},
            {"pixels": 3 * 724 ** 2, "fuse": "wfusimg", "time": 336, "time_std": 5},
            {"pixels": 3 * 1024 ** 2, "fuse": "wfusimg", "time": 659, "time_std": 20},
            {"pixels": 3 * 1448 ** 2, "fuse": "wfusimg", "time": 1712, "time_std": 60},
            {"pixels": 3 * 2048 ** 2, "fuse": "wfusimg", "time": 4267, "time_std": 68},
        ]

        results = pd.DataFrame(results)
        results.to_csv("results/time-analysis.csv")
    else:
        results = pd.read_csv("results/time-analysis.csv")
    print(results)

    plt.plot(
        results[results.fuse == "wavelet"].pixels / 3,
        results[results.fuse == "wavelet"].time,
        label="wavelet",
        color="r",
    )
    plt.fill_between(
        results[results.fuse == "wavelet"].pixels / 3,
        results[results.fuse == "wavelet"].time - results[results.fuse == "wavelet"].time_std * 3,
        results[results.fuse == "wavelet"].time + results[results.fuse == "wavelet"].time_std * 3,
        color="r",
        alpha=0.333,
    )
    plt.plot(
        results[results.fuse == "guided_filter"].pixels / 3,
        results[results.fuse == "guided_filter"].time,
        label="guided_filter",
        color="g",
    )
    plt.fill_between(
        results[results.fuse == "guided_filter"].pixels / 3,
        results[results.fuse == "guided_filter"].time - results[results.fuse == "guided_filter"].time_std * 3,
        results[results.fuse == "guided_filter"].time + results[results.fuse == "guided_filter"].time_std * 3,
        color="g",
        alpha=0.333,
    )
    plt.plot(
        results[results.fuse == "wfusimg"].pixels[:5] / 3,
        results[results.fuse == "wfusimg"].time[:5],
        label="wfusimg",
        color="b",
    )
    plt.fill_between(
        results[results.fuse == "wfusimg"].pixels[:5] / 3,
        results[results.fuse == "wfusimg"].time[:5] - results[results.fuse == "wfusimg"].time_std[:5] * 3,
        results[results.fuse == "wfusimg"].time[:5] + results[results.fuse == "wfusimg"].time_std[:5] * 3,
        color="b",
        alpha=0.333,
    )
    plt.xlabel("Image Area (# Pixels)")
    plt.ylabel("Execution Time (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/time-comparison.jpg")
    plt.show()

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

    fusedw = np.asfarray(imageio.imread("results/stained_glass_wfusimg_level2.jpg")) / 255
    _, ax = plt.subplots(1, 3, figsize=(15, 5))
    [axi.set_axis_off() for axi in ax.ravel()]
    ax[0].imshow(glass_A)
    ax[1].imshow(glass_B)
    ax[2].imshow(resize(fusedw, (*fusedw.shape[:2], 3)))
    plt.tight_layout()
    plt.savefig(f"results/stained_glass_wfuseimg.jpg")
    plt.close()

    print("\nwfusimg")
    print("NMI", nmi([glass_A, glass_B], fusedw))
    print("Q_c", cvejic_Q_c(glass_A, glass_B, fusedw))
    print("Q_y", yang_Q_y(glass_A, glass_B, fusedw, figname=f"stained_glass_wfuseimg"))

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

    fusedw = np.asfarray(imageio.imread("results/clock_wfusimg_raw.jpg")) / 255
    _, ax = plt.subplots(1, 3, figsize=(15, 5))
    [axi.set_axis_off() for axi in ax.ravel()]
    ax[0].imshow(clock_A)
    ax[1].imshow(clock_B)
    ax[2].imshow(resize(fusedw, (*fusedw.shape[:2], 3)))
    plt.tight_layout()
    plt.savefig(f"results/clock_wfusimg.jpg")
    plt.close()

    print("\nwfusimg")
    print("NMI", nmi([clock_A, clock_B], fusedw))
    print("Q_c", cvejic_Q_c(clock_A, clock_B, fusedw))
    print("Q_y", yang_Q_y(clock_A, clock_B, fusedw, figname=f"clock_wfusimg"))

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
                yang_Q_y(brain_A, brain_B, brain_fused, figname=f"{dir.replace('data/','').replace('/','_')}_{name}"),
            )

        fusedw = (
            np.asfarray(
                imageio.imread(f"results/{dir.replace('data/','').replace('/','_')}_wfusimg_raw.jpg")[..., None]
            )
            / 255
        )
        _, ax = plt.subplots(1, 3, figsize=(15, 5))
        [axi.set_axis_off() for axi in ax.ravel()]
        ax[0].imshow(resize(brain_A, (*fusedw.shape[:2], 3)))
        ax[1].imshow(brain_B)
        ax[2].imshow(resize(fusedw, (*fusedw.shape[:2], 3)))
        plt.tight_layout()
        plt.savefig(f"results/{dir.replace('data/','').replace('/','_')}_wfusimg.jpg")
        plt.close()

        print("\nwfusimg")
        print("NMI", nmi([brain_A, brain_B], fusedw))
        print("Q_c", cvejic_Q_c(brain_A, brain_B, fusedw))
        print("Q_y", yang_Q_y(brain_A, brain_B, fusedw, figname=f"{dir.replace('data/','').replace('/','_')}_wfusimg"))

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

    pairs = [[np.asfarray(imageio.imread(file)) / 255 for file in pair] for pair in pairs]

    print("\n\n\nWAVELET KERNEL SIZE SHOWDOWN!")
    print("wavelet,size,NMI,Qc,Qy")

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

            print(f"{wavelet},{kernel_size},{np.mean(NMIs)},{np.mean(Q_cs)},{np.mean(Q_ys)}")
        except:
            pass

    import pandas as pd

    data = pd.read_csv("results/wavelet-comparison.csv")
    data["Sum"] = data.NMI + data.Qy + data.Qc
    data["NormSum"] = (
        ((data.NMI - data.NMI.min()) / (data.NMI.max() - data.NMI.min()))
        + ((data.Qy - data.Qy.min()) / (data.Qy.max() - data.Qy.min()))
        + ((data.Qc - data.Qc.min()) / (data.Qc.max() - data.Qc.min()))
    )

    sort_by = ["NMI", "Qy", "Qc", "Sum", "NormSum"]
    best = pd.concat([data.sort_values(by=by, ascending=False)[:5] for by in sort_by])

    for j, (A, B) in enumerate(pairs):
        if len(A.shape) == 2:
            A = A[..., None]
        fig, ax = plt.subplots(5, 5, figsize=(20, 20))
        for i, (_, row) in enumerate(best.iterrows()):
            axi = ax[i // 5, i % 5]
            if i % 5 == 0:
                axi.set_ylabel(sort_by[i // 5], fontsize=40)
            axi.imshow(wavelet_fuse([A, B], kernel_size=row["size"], wavelet=row["wavelet"]))
            axi.set_title(row["wavelet"] + ", " + str(row["size"]), y=-0.01, color="white")
            axi.tick_params(
                axis="both",
                which="both",
                left=False,
                right=False,
                bottom=False,
                top=False,
                labelleft=False,
                labelbottom=False,
            )
        plt.tight_layout()
        plt.savefig(f"results/wavelet-comparison{j}.pdf")

