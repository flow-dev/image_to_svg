# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
Binarization.py

python Binarization.py --img NikeJustDoIt.png --method default

--methodは,hist_threshのハイパーパラメータの指定
'default', 'otsu', 'met', 'percentile'の4パタンある.
opencvのotsuと差を出すには,defaultを使っておけば良い.
percentile も使いどころはありそう.

--method is a hyperparameter for hist_thresh.
There are four patterns: 'default', 'otsu', 'met', and 'percentile'.
If you want to make a difference with opencv's otsu, you should use default.
percentile is also useful.

#hist_thresh
#https://github.com/jonbarron/hist_thresh

#古くて新しい2値化画像処理を動かしてみる(ECCV 2020論文)
#https://tech-blog.optim.co.jp/entry/2020/10/12/090000

"""

__author__  = "flow-dev"
__version__ = "1.00"
__date__    = "25 Feb 2022"



import cv2
import numpy as np
import argparse
import os

# --------------- Arguments ---------------

parser = argparse.ArgumentParser(description='Binarization')
parser.add_argument('--img', type=str, required=True)
parser.add_argument('--method', type=str, choices=['default', 'otsu', 'met', 'percentile'])
parser.add_argument('--mono_inv', action='store_true')
args = parser.parse_args()


csum = lambda z: np.cumsum(z)[:-1]
dsum = lambda z: np.cumsum(z[::-1])[-2::-1]
argmax = lambda x, f: np.mean(x[:-1][f == np.max(f)])  # Use the mean for ties.
clip = lambda z: np.maximum(1e-30, z)


def preliminaries(n, x):
    """Some math that is shared across multiple algorithms."""
    assert np.all(n >= 0)
    x = np.arange(len(n), dtype=n.dtype) if x is None else x
    assert np.all(x[1:] >= x[:-1])
    w0 = clip(csum(n))
    w1 = clip(dsum(n))
    p0 = w0 / (w0 + w1)
    p1 = w1 / (w0 + w1)
    mu0 = csum(n * x) / w0
    mu1 = dsum(n * x) / w1
    d0 = csum(n * x**2) - w0 * mu0**2
    d1 = dsum(n * x**2) - w1 * mu1**2
    return x, w0, w1, p0, p1, mu0, mu1, d0, d1


def GHT(n, x=None, nu=0, tau=0, kappa=0, omega=0.5, prelim=None):
    assert nu >= 0
    assert tau >= 0
    assert kappa >= 0
    assert omega >= 0 and omega <= 1
    x, w0, w1, p0, p1, _, _, d0, d1 = prelim or preliminaries(n, x)
    v0 = clip((p0 * nu * tau**2 + d0) / (p0 * nu + w0))
    v1 = clip((p1 * nu * tau**2 + d1) / (p1 * nu + w1))
    f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa *      omega)  * np.log(w0)
    f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
    return argmax(x, f0 + f1), f0 + f1


def im2hist(im, zero_extents=False):
    # Convert an image to grayscale, bin it, and optionally zero out the first and last bins.
    max_val = np.iinfo(im.dtype).max
    x = np.arange(max_val+1)
    e = np.arange(-0.5, max_val+1.5)
    assert len(im.shape) in [2, 3]
    im_bw = np.amax(im[...,:3], -1) if len(im.shape) == 3 else im
    n = np.histogram(im_bw, e)[0]
    if zero_extents:
        n[0] = 0
        n[-1] = 0
    return n, x, im_bw


def ght_thresh(im, nu=None, tau=None, kappa=None, omega=0.5):
    n, x, im_bw = im2hist(im)
    prelim = preliminaries(n, x)

    nu = np.sum(n) if nu is None else nu
    tau = np.sqrt(1/12) if tau is None else tau
    kappa = np.sum(n) if kappa is None else kappa

    t, score = GHT(n, x, nu, tau, kappa, omega, prelim)
    return t, score


def example():
    img_file_path = args.img
    im = cv2.imread(img_file_path)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    if(args.method=="otsu"):
        t, _ = ght_thresh(gray, nu=1e30, tau=1.0, kappa=1e-30) #Otsu's Method
        ght_filename = "result_otsu_ght.png"
    elif(args.method=="met"):
        t, _ = ght_thresh(gray, nu=1e-30, kappa=1e-30) #MET
        ght_filename = "result_met_ght.png"
    elif(args.method=="percentile"):
        t, _ = ght_thresh(gray, nu=1e-30, kappa=1e30) #Percentile
        ght_filename = "result_percentile_ght.png"
    else:
        t, _ = ght_thresh(gray) #Default
        ght_filename = "result_ght.png"

    _, bw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    if(args.mono_inv):
        bw = cv2.bitwise_not(bw)
    cv2.imwrite(ght_filename, bw)

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    if(args.mono_inv):
        bw = cv2.bitwise_not(bw)
    cv2.imwrite("result_otsu_opencv.png", bw)

    # import matplotlib.pyplot as plt
    # plt.imshow(bw)#, cmap = 'gray')
    # plt.colorbar()
    # plt.show()

if __name__ == '__main__':
    example()
