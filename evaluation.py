from glob import glob

import numpy as np
from PIL import Image
from sewar.full_ref import ssim, uqi, vifp

from guided_filter_fuse import fuse_images as guided_filter_fuse
from mutual_info import mutual_information_2d


def yang_Q_y(ref_A, ref_B, fused):
    Q_y = 0
    for y in range(ref_A.shape[0] - 7):
        for x in range(ref_A.shape[1] - 7):
            A_w = ref_A[y : y + 7, x : x + 7]
            B_w = ref_B[y : y + 7, x : x + 7]
            F_w = fused[y : y + 7, x : x + 7]
            lambda_w = np.var(A_w) / (np.var(A_w) + np.var(B_w))
            ssim_A = ssim(A_w, F_w)[0]
            ssim_B = ssim(B_w, F_w)[0]
            weighted_avg = lambda_w * ssim_A + (1 - lambda_w) * ssim_B
            if weighted_avg > 0.75:
                Q_y += weighted_avg
            else:
                Q_y += max(ssim_A, ssim_B)
    return Q_y


def cvejic(ref_A, ref_B, fused):
    covariances = [np.cov(ref_A.ravel(), fused.ravel()), np.cov(ref_B.ravel(), fused.ravel())]
    mu = np.clip(covariances[0] / sum(covariances), 0, 1)
    return mu * uqi(ref_A, fused) + (1 - mu) * uqi(ref_B, fused)


def nmi(inputs, fused):
    return [mutual_information_2d(input.ravel(), fused.ravel()) for input in inputs]


def structural_similarity(inputs, fused):
    return [ssim(input, fused)[0] for input in inputs]


def visual_information_fidelity(inputs, fused):
    return [vifp(input, fused) for input in inputs]


if __name__ == "__main__":
    horse_images = [np.asarray(Image.open(file)) for file in glob("data/horse/*.png")]
    horse_fused = guided_filter_fuse(horse_images)

    print("stained glass")
    glass_A, glass_B = [np.asarray(Image.open(file)) for file in glob("data/stained-glass*.png")]
    glass_fused = guided_filter_fuse([glass_A, glass_B])
    print("Q_y", yang_Q_y(glass_A, glass_B, glass_fused))
    print("Q_c", cvejic(glass_A, glass_B, glass_fused))

    print("brains")
    for dir in glob("data/brain-atlas-normal-aging/*"):
        print(dir)
        brain_A, brain_B = [np.asarray(Image.open(file)) for file in glob(f"{dir}/*.gif")]
        print(brain_A.shape, brain_B.shape)
        brain_fused = guided_filter_fuse([brain_A, brain_B])
        print("Q_y", yang_Q_y(brain_A, brain_B, brain_fused))
        print("Q_c", cvejic(brain_A, brain_B, brain_fused))
    print()

    print("horse")
    print("normalized mutual information", nmi(horse_images, horse_fused))
    print("structural similarity", structural_similarity(horse_images, horse_fused))
    print("visual information fidelity", visual_information_fidelity(horse_images, horse_fused))
    print()
