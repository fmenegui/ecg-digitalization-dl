import numpy as np
import pytest
import torch

from ecgdl.geometry.lattice import Lattice, LatticeError, apply_homography, build_lattice, fit_homography, smooth_grid
from ecgdl.geometry.peaks import decode_peaks
from ecgdl.geometry.rectify import region_coords, sample_image, sample_region


def warped_grid(rows=24, cols=34, spacing=22.0, curl=6.0, seed=0):
    """Grid nodes seen through a perspective plus a smooth non-planar bend."""
    rng = np.random.default_rng(seed)
    gi, gj = np.meshgrid(np.arange(cols), np.arange(rows))
    ij = np.stack([gi.ravel(), gj.ravel()], 1)
    src = np.array([[0, 0], [cols, 0], [cols, rows], [0, rows]], float)
    dst = src * spacing + 60 + rng.uniform(-40, 40, (4, 2))
    h = fit_homography(src, dst)
    xy = apply_homography(h, ij.astype(float))
    xy[:, 1] += curl * np.sin(ij[:, 0] / cols * np.pi)  # paper curl
    return ij, xy


def test_homography_fit_is_exact_with_four_points():
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    dst = np.array([[10, 12], [50, 8], [55, 60], [6, 52]], float)
    assert np.allclose(apply_homography(fit_homography(src, dst), src), dst, atol=1e-8)


def test_smooth_grid_interpolates_missing_values():
    y, x = np.mgrid[0:20, 0:30]
    truth = 0.02 * x**2 - 0.3 * y
    w = np.ones_like(truth)
    w[5:9, 10:15] = 0
    est = smooth_grid(np.where(w > 0, truth, 0.0), w, 0.05)
    assert np.abs(est - truth)[5:9, 10:15].max() < 0.5


@pytest.mark.parametrize("rotation", [0.0, 0.5, 1.6, 3.0])
def test_lattice_from_unordered_noisy_points(rotation):
    ij, xy = warped_grid()
    c, s = np.cos(rotation), np.sin(rotation)
    xy = (xy - xy.mean(0)) @ np.array([[c, s], [-s, c]]) + xy.mean(0)
    rng = np.random.default_rng(1)
    keep = rng.random(len(xy)) > 0.15  # missed detections
    noisy = xy[keep] + rng.normal(0, 0.4, (keep.sum(), 2))
    clutter = xy.min(0) + rng.random((25, 2)) * np.ptp(xy, 0)  # false positives
    pts = rng.permutation(np.concatenate([noisy, clutter]))
    lattice = build_lattice(pts)
    assert lattice.quality["nodes"] > 0.95 * keep.sum()
    # every true node, detected or not, is predicted by the smooth map
    i0, i1, j0, j1 = lattice.support
    gi, gj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
    pred = lattice.index_to_image(torch.tensor(np.stack([gi.ravel(), gj.ravel()], 1), dtype=torch.float32)).numpy()
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(pred).query(xy)
    assert np.median(dist) < 0.3 and np.percentile(dist, 95) < 0.8
    assert abs(lattice.spacing_px - 22) < 3


def test_lattice_orientation_follows_direction():
    ij, xy = warped_grid(curl=0.0)
    lattice = build_lattice(xy)
    for d in ([1, 0], [0, 1], [-1, 0], [0, -1]):
        lat = lattice.oriented(np.array(d, float))
        a = lat.mm_to_image(torch.tensor([[50.0, 50.0], [60.0, 50.0], [50.0, 60.0]])).numpy()
        time_dir, down_dir = a[1] - a[0], a[2] - a[0]
        assert time_dir @ np.array(d) > 0.9 * np.linalg.norm(time_dir)
        assert time_dir[0] * down_dir[1] - time_dir[1] * down_dir[0] > 0  # not mirrored
        assert lat.extent_mm[0] > 0 and lat.extent_mm[1] > 0


def test_lattice_millimetres_are_metric():
    ij, xy = warped_grid(curl=0.0)
    lattice = Lattice.from_correspondences(ij, xy)
    p = lattice.mm_to_image(torch.tensor([[25.0, 40.0]]))[0].numpy()  # node (5, 8)
    assert np.linalg.norm(p - xy[(ij[:, 0] == 5) & (ij[:, 1] == 8)][0]) < 0.05


def test_lattice_rejects_noise():
    with pytest.raises(LatticeError):
        build_lattice(np.random.default_rng(0).random((200, 2)) * 500)
    with pytest.raises(LatticeError):
        build_lattice(np.zeros((3, 2)))


def test_decode_peaks_subpixel():
    centres = np.array([[20.3, 15.7], [60.8, 40.2], [90.1, 70.5]])
    yy, xx = np.mgrid[0:96, 0:128]
    heat = sum(np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.2**2)) for cx, cy in centres)
    xy, score = decode_peaks(torch.tensor(heat, dtype=torch.float32), None, 0.3)
    assert len(xy) == 3
    found = xy[np.argsort(xy[:, 0])]
    assert np.abs(found - centres).max() < 0.05
    assert (score > 0.7).all()


def test_sample_region_reads_the_right_pixels():
    ij, xy = warped_grid(curl=0.0)
    lattice = Lattice.from_correspondences(ij, xy)
    image = torch.zeros(1, 700, 1000)
    px, py = lattice.mm_to_image(torch.tensor([[50.0, 30.0]]))[0].round().long().tolist()
    image[0, py - 1 : py + 2, px - 1 : px + 2] = 1.0  # a dot at (50 mm, 30 mm)
    crop, inside = sample_region(image, lattice, 40.0, 20.0, 20.0, 20.0, 80, 80)
    assert inside.all()
    y, x = np.unravel_index(int(crop[0].argmax()), crop[0].shape)
    assert abs(x - 40) <= 2 and abs(y - 40) <= 2
    coords = region_coords(lattice, 40.0, 20.0, 20.0, 20.0, 80, 80)
    far, inside2 = sample_image(image, coords + 5000)
    assert not inside2.any() and far.shape == crop.shape
