import numpy as np
from skimage.filters import threshold_minimum, threshold_otsu


def optimal_thresholding(img):  # utility function
    h = np.zeros(256)
    tint = 0
    for x in img.flatten():
        h[int(x)] += 1
    for x in range(h.shape[0]):
        tint += x * h[int(x)]
    h_c = np.zeros(256)
    h_c[0] = h[0]
    for i in range(1, 256):
        h_c[i] = h[i] + h_c[i - 1]
    tint = int(round(tint / h_c[-1]))
    iterator = 0
    while 1:
        iterator += 1
        tint_l = 0
        tint_hi = 0
        l = h[0:tint]
        hi = h[tint:256]
        for x in range(l.shape[0]):
            tint_l += x * l[int(x)]
        if h_c[tint] == 0:
            tint_l = 0
        else:
            tint_l = int(round(tint_l / (h_c[tint])))

        for x in range(hi.shape[0]):
            tint_hi += x * hi[int(x)]
        if h_c[-1] == h_c[tint]:
            tint_hi = 0
        else:
            tint_hi = int(round(tint_hi / (h_c[-1] - h_c[tint])))

        tint_prev = tint
        tint = round(tint_hi + tint_l / 2)
        if tint_prev == tint or iterator > 1000:
            break

    return int(tint)


def global_thresholding(img):  # utility function

    return int(0.4 * 255)  # global threshold


def choose_thresholding_type(img, mode):  # utility function
    if mode == 1:  # default
        return optimal_thresholding(img)
    elif mode == 2:
        return threshold_otsu(img)
    elif mode == 3:
        return threshold_minimum(img)
    else:
        return global_thresholding(img)


def median_filter(img, size):  # utility function
    M = img.shape[0]
    N = img.shape[1]
    img2 = np.ones((M, N))
    W = int(size / 2)
    L = int(size / 2)
    for i in range(W, M - W):
        for j in range(L, N - L):
            temp = img[i - W:i + W + 1, j - L:j + L + 1]
            med = np.median(temp.flatten())
            img2[i, j] = int(med)
    return img2


def max_filter(img, size):  # utility function
    M = img.shape[0]
    N = img.shape[1]
    img2 = np.ones((M, N))
    W = int(size / 2)
    L = int(size / 2)
    for i in range(W, M - W):
        for j in range(L, N - L):
            temp = img[i - W:i + W + 1, j - L:j + L + 1]
            maxim = np.median(temp.flatten())
            img2[i, j] = int(maxim)
    return img2


def min_filter(img, size):  # utility function
    M = img.shape[0]
    N = img.shape[1]
    img2 = np.ones((M, N))
    W = int(size / 2)
    L = int(size / 2)
    for i in range(W, M - W):
        for j in range(L, N - L):
            temp = img[i - W:i + W + 1, j - L:j + L + 1]
            mini = np.median(temp.flatten())
            img2[i, j] = int(mini)
    return img2


def binarize(img, mode=1):  # interface function
    return 1 * (img > choose_thresholding_type(img, mode))


def apply_filter(img, size, mode=1):  # interface function
    if mode == 1:  # default
        return max_filter(img, size)  # brighter image
    elif mode == 2:
        return min_filter(img, size)  # darker image
    else:
        return median_filter(img, size)
