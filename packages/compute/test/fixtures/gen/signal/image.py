"""Golden values for aifn-compute/signal/image from scikit-image and scipy.ndimage: gradients, corners, blobs, edges,
Hough
lines, morphology and pyramid reduction."""

import numpy as np
from scipy import ndimage
from skimage import feature, transform  # pyright: ignore[reportMissingTypeStubs]


def shapes(h: int, w: int) -> np.ndarray:
    """A square, a disc and a bar on a dark background, lightly blurred."""
    rr, cc = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w))
    img[8:20, 6:18] = 1.0
    img[(rr - 26) ** 2 + (cc - 34) ** 2 <= 36] = 0.8
    img[30:34, 4:24] = 0.6
    return np.asarray(ndimage.gaussian_filter(img, 0.7))


def blobs(h: int, w: int) -> np.ndarray:
    """Three Gaussian blobs of standard deviations 2, 3 and 4.5."""
    rr, cc = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w))
    for r, c, s in [(12, 12, 2.0), (30, 16, 3.0), (22, 42, 4.5)]:
        img += np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * s * s))
    return img


def cases() -> dict[str, object]:
    img = shapes(40, 48)
    scharr = np.array([3.0, 10.0, 3.0])
    d = np.array([-1.0, 0.0, 1.0])
    gx_scharr = ndimage.correlate(img, np.outer(scharr, d), mode="reflect")
    gy_scharr = ndimage.correlate(img, np.outer(d, scharr), mode="reflect")
    rr_, rc_, cc_ = feature.structure_tensor(img, sigma=1.0, order="rc")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportArgumentType]
    b = blobs(44, 56)
    found = feature.blob_log(b, min_sigma=1, max_sigma=8, num_sigma=10, threshold=0.1)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    edges = np.zeros((30, 40))
    for x in range(40):
        y = round(0.5 * x + 3)
        if 0 <= y < 30:
            edges[y, x] = 1
    edges[:, 25] = 1
    acc, angles, dists = transform.hough_line(edges)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    _, peak_angles, peak_dists = transform.hough_line_peaks(acc, angles, dists, num_peaks=2)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    disk = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    square5 = np.ones((5, 5), dtype=bool)
    k5 = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16
    blurred = ndimage.correlate1d(ndimage.correlate1d(img, k5, axis=0, mode="mirror"), k5, axis=1, mode="mirror")
    return {
        "image": img,
        "prewitt": {"x": ndimage.prewitt(img, axis=1), "y": ndimage.prewitt(img, axis=0)},
        "scharr": {"x": gx_scharr, "y": gy_scharr},
        "structureTensor": {"rr": rr_, "rc": rc_, "cc": cc_},
        "harrisResponse": feature.corner_harris(img, method="k", k=0.05, sigma=1.0),  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        "shiTomasiResponse": feature.corner_shi_tomasi(img, sigma=1.0),  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        "gaussianLaplace": ndimage.gaussian_laplace(img, 2.0),
        "blobsLog": {"image": b, "blobs": np.asarray(found)},  # pyright: ignore[reportUnknownArgumentType]
        "canny": {
            "sigma": 1.5,
            "edges": feature.canny(img, sigma=1.5).astype(float),  # pyright: ignore[reportUnknownMemberType]
        },
        "houghLines": {
            "edges": edges,
            "votes": np.asarray(acc),  # pyright: ignore[reportUnknownArgumentType]
            "angles": np.asarray(peak_angles),  # pyright: ignore[reportUnknownArgumentType]
            "distances": np.asarray(peak_dists),  # pyright: ignore[reportUnknownArgumentType]
        },
        "morphology": {
            "erode": ndimage.grey_erosion(img, footprint=disk),
            "dilate": ndimage.grey_dilation(img, footprint=disk),
            "erode5": ndimage.grey_erosion(img, footprint=square5),
            "opening": ndimage.grey_opening(img, footprint=square5),
            "closing": ndimage.grey_closing(img, footprint=square5),
        },
        "pyramidReduce": blurred[::2, ::2],
    }
