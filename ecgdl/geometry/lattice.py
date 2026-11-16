"""From unordered grid-intersection detections to a metric map of the paper.

ECG paper carries its own ruler: major grid lines every 5 mm. A network detects the
intersections; this module turns that unordered point cloud into a lattice with integer
indices and then into a smooth mapping ``(mm on paper) -> (pixel in the photo)``.

Three classical steps:

1. **Growth.** Starting from a point whose four neighbours form a clean cross, the lattice
   is grown best-first: the most certain neighbour is indexed next and the local basis is
   carried along, so perspective and paper curvature are followed gradually.
2. **Projective fit.** A homography ``index -> pixel`` is fitted with iteratively
   reweighted least squares. It explains a flat sheet seen by a pinhole camera.
3. **Smooth residual.** What the homography cannot explain (curl, folds, lens distortion)
   is a smooth 2D field on the index grid. It is estimated with robust penalised
   least squares in the DCT domain (Garcia 2010), which also interpolates the nodes that
   were not detected and extrapolates past the border.

No template of a specific ECG format is used anywhere, unlike registration-based
competition pipelines: the lattice only assumes a 5 mm grid.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from scipy.fft import dctn, idctn
from scipy.spatial import cKDTree
from torch.nn import functional as F

from ..constants import MAJOR_GRID_MM


class LatticeError(RuntimeError):
    """The grid could not be recovered from the detections."""


@dataclass(frozen=True)
class LatticeConfig:
    knn: int = 8
    #: accepted mismatch between a predicted and a detected neighbour, as a fraction of the local spacing
    growth_tolerance: float = 0.30
    #: weight of the newly measured step when the local basis is propagated
    basis_update: float = 0.6
    min_points: int = 16
    max_seeds: int = 6
    irls_iterations: int = 6
    #: smoothing strength of the residual field (higher = stiffer paper)
    smooth: float = 1.0
    #: outlier threshold on the final residual, as a fraction of the grid spacing
    outlier_fraction: float = 0.12
    #: cells of extrapolation kept around the detected support
    margin_cells: int = 3
    reattach_tolerance: float = 0.22


# --------------------------------------------------------------------------------------
# robust building blocks
# --------------------------------------------------------------------------------------
def _normalise(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = p.mean(0)
    scale = np.sqrt(2.0) / max(np.sqrt(((p - mean) ** 2).sum(1).mean()), 1e-9)
    t = np.array([[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1.0]])
    return (p - mean) * scale, t


def fit_homography(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted DLT homography mapping ``src[N, 2] -> dst[N, 2]`` (Hartley-normalised)."""
    if len(src) < 4:
        raise LatticeError("a homography needs at least 4 correspondences")
    s, ts = _normalise(src.astype(np.float64))
    d, td = _normalise(dst.astype(np.float64))
    w = np.ones(len(src)) if weights is None else np.sqrt(np.maximum(weights, 0.0))
    zero = np.zeros(len(src))
    one = np.ones(len(src))
    ax = np.stack([-s[:, 0], -s[:, 1], -one, zero, zero, zero, d[:, 0] * s[:, 0], d[:, 0] * s[:, 1], d[:, 0]], 1)
    ay = np.stack([zero, zero, zero, -s[:, 0], -s[:, 1], -one, d[:, 1] * s[:, 0], d[:, 1] * s[:, 1], d[:, 1]], 1)
    a = np.concatenate([ax * w[:, None], ay * w[:, None]], 0)
    _, _, vt = np.linalg.svd(a, full_matrices=a.shape[0] < 9)  # the null vector needs the full basis
    h = vt[-1].reshape(3, 3)
    h = np.linalg.inv(td) @ h @ ts
    return h / h[2, 2] if abs(h[2, 2]) > 1e-12 else h


def apply_homography(h: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = np.concatenate([p, np.ones((*p.shape[:-1], 1))], -1) @ h.T
    return q[..., :2] / q[..., 2:3]


def robust_homography(src: np.ndarray, dst: np.ndarray, iterations: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """IRLS (Huber) homography. Returns ``(H, weights)``."""
    w = np.ones(len(src))
    h = fit_homography(src, dst, w)
    for _ in range(iterations):
        r = np.linalg.norm(apply_homography(h, src) - dst, axis=1)
        scale = max(1.4826 * np.median(r), 1e-3)
        w = np.minimum(1.0, 2.0 * scale / np.maximum(r, 1e-9))
        h = fit_homography(src, dst, w)
    return h, w


def smooth_grid(values: np.ndarray, weights: np.ndarray, smooth: float, iterations: int = 200) -> np.ndarray:
    """Robust penalised least-squares smoothing of gridded data with missing values.

    Solves ``min_z  sum w (y - z)^2 + smooth * |Laplacian z|^2`` in the DCT domain
    (Garcia, *Comput. Stat. Data Anal.* 2010). Entries with zero weight are interpolated.
    """
    y = np.nan_to_num(values.astype(np.float64))
    w = np.clip(weights.astype(np.float64), 0.0, None)
    if w.max() <= 0:
        return np.zeros_like(y)
    w = w / w.max()
    nj, ni = y.shape
    lam = (2 - 2 * np.cos(np.pi * np.arange(nj) / nj))[:, None] + (2 - 2 * np.cos(np.pi * np.arange(ni) / ni))[None, :]
    gamma = 1.0 / (1.0 + smooth * lam**2)
    z = np.full_like(y, (w * y).sum() / w.sum())
    relax = 1.75 if (w < 1).any() else 1.0
    for _ in range(iterations):
        z_new = idctn(gamma * dctn(w * (y - z) + z, norm="ortho"), norm="ortho")
        z_new = relax * z_new + (1 - relax) * z
        if np.abs(z_new - z).max() < 1e-4:
            z = z_new
            break
        z = z_new
    return z


# --------------------------------------------------------------------------------------
# lattice object
# --------------------------------------------------------------------------------------
@dataclass
class Lattice:
    """Smooth map between paper millimetres and photo pixels.

    Internal index coordinates ``(i, j)`` count major grid lines. Public coordinates are
    millimetres ``(x, y)`` in an upright frame chosen by :meth:`oriented`: ``x`` grows with
    time, ``y`` grows downwards, and ``(0, 0)`` is the top-left corner of the detected grid.
    """

    homography: np.ndarray  #: 3x3, index -> pixel
    residual: np.ndarray  #: ``[2, nj, ni]`` pixel correction on the index grid
    i0: int  #: index of ``residual[:, :, 0]``
    j0: int  #: index of ``residual[:, 0, :]``
    support: tuple[int, int, int, int]  #: (i_min, i_max, j_min, j_max) of the detected nodes
    spacing_px: float
    axes: np.ndarray = field(default_factory=lambda: np.eye(2))  #: upright (a, b) = axes @ (i, j)
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2))  #: upright index of mm (0, 0)
    quality: dict = field(default_factory=dict)
    mm_per_cell: float = MAJOR_GRID_MM

    # ------------------------------------------------------------------ coordinates
    def index_to_image(self, ij: torch.Tensor) -> torch.Tensor:
        """Map real-valued index coordinates ``[..., 2]`` to pixel coordinates ``[..., 2]``."""
        dev, dt = ij.device, torch.float32
        h = torch.as_tensor(self.homography, dtype=torch.float64, device=dev)
        flat = ij.reshape(-1, 2).to(torch.float64)
        q = torch.cat([flat, torch.ones_like(flat[:, :1])], 1) @ h.T
        xy = (q[:, :2] / q[:, 2:3]).to(dt)
        res = torch.as_tensor(self.residual, dtype=dt, device=dev)[None]
        nj, ni = res.shape[-2:]
        gx = (flat[:, 0].to(dt) - self.i0) / max(ni - 1, 1) * 2 - 1
        gy = (flat[:, 1].to(dt) - self.j0) / max(nj - 1, 1) * 2 - 1
        grid = torch.stack((gx, gy), -1)[None, None]
        mode = "bicubic" if min(nj, ni) >= 4 else "bilinear"
        corr = F.grid_sample(res, grid, mode=mode, padding_mode="border", align_corners=True)[0, :, 0].T
        return (xy + corr).reshape(*ij.shape[:-1], 2)

    def mm_to_index(self, xy_mm: torch.Tensor) -> torch.Tensor:
        inv = torch.as_tensor(np.linalg.inv(self.axes), dtype=torch.float32, device=xy_mm.device)
        ab = xy_mm.to(torch.float32) / self.mm_per_cell + torch.as_tensor(
            self.origin, dtype=torch.float32, device=xy_mm.device
        )
        return ab @ inv.T

    def mm_to_image(self, xy_mm: torch.Tensor) -> torch.Tensor:
        """Map upright paper millimetres ``[..., 2]`` to photo pixels ``[..., 2]``."""
        return self.index_to_image(self.mm_to_index(xy_mm))

    @property
    def extent_mm(self) -> tuple[float, float]:
        """Width and height (mm) of the detected grid support in the upright frame."""
        i_min, i_max, j_min, j_max = self.support
        corners = np.array([[i_min, j_min], [i_max, j_min], [i_min, j_max], [i_max, j_max]], float) @ self.axes.T
        span = corners.max(0) - corners.min(0)
        return float(span[0] * self.mm_per_cell), float(span[1] * self.mm_per_cell)

    # ------------------------------------------------------------------ orientation
    def oriented(self, direction: np.ndarray | None) -> Lattice:
        """Return a copy whose ``x`` axis is the grid axis closest to ``direction`` (pixels).

        ``direction`` is the image-space direction in which time increases on the page.
        ``None`` keeps the axis closest to image +x. The ``y`` axis is chosen so that the
        frame is not mirrored.
        """
        i_min, i_max, j_min, j_max = self.support
        centre = torch.tensor([[(i_min + i_max) / 2, (j_min + j_max) / 2]], dtype=torch.float32)
        base = replace(self, axes=np.eye(2), origin=np.zeros(2))
        step = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        p0 = base.index_to_image(centre)[0].numpy()
        u = base.index_to_image(centre + step[0])[0].numpy() - p0
        v = base.index_to_image(centre + step[1])[0].numpy() - p0
        d = np.array([1.0, 0.0]) if direction is None else np.asarray(direction, float)
        d = d / max(np.linalg.norm(d), 1e-9)
        cands = [(np.array([1, 0]), u), (np.array([-1, 0]), -u), (np.array([0, 1]), v), (np.array([0, -1]), -v)]
        row_a, vec_a = max(cands, key=lambda c: float(c[1] @ d) / max(np.linalg.norm(c[1]), 1e-9))
        # the second axis must make (time, down) right-handed in image coordinates (y down)
        row_b, _ = max(
            (c for c in cands if abs(int(c[0] @ row_a)) == 0),
            key=lambda c: float(vec_a[0] * c[1][1] - vec_a[1] * c[1][0]),
        )
        axes = np.stack([row_a, row_b]).astype(float)
        corners = np.array([[i_min, j_min], [i_max, j_min], [i_min, j_max], [i_max, j_max]], float) @ axes.T
        return replace(self, axes=axes, origin=corners.min(0))

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_correspondences(
        cls,
        ij: np.ndarray,
        xy: np.ndarray,
        config: LatticeConfig | None = None,
    ) -> Lattice:
        """Fit the map from indexed detections (``ij`` integer indices, ``xy`` pixels)."""
        cfg = config or LatticeConfig()
        ij = np.asarray(ij, dtype=np.int64)
        xy = np.asarray(xy, dtype=np.float64)
        if len(ij) < 4:
            raise LatticeError("too few indexed grid points")
        h, _ = robust_homography(ij.astype(np.float64), xy, cfg.irls_iterations)
        i_min, j_min = ij.min(0)
        i_max, j_max = ij.max(0)
        m = cfg.margin_cells
        i0, j0 = int(i_min - m), int(j_min - m)
        ni, nj = int(i_max - i_min + 1 + 2 * m), int(j_max - j_min + 1 + 2 * m)
        r = xy - apply_homography(h, ij.astype(np.float64))
        spacing = float(np.median(np.linalg.norm(
            apply_homography(h, ij + np.array([1.0, 0.0])) - apply_homography(h, ij.astype(float)), axis=1)))
        gi, gj = ij[:, 0] - i0, ij[:, 1] - j0
        weights = np.zeros((nj, ni))
        weights[gj, gi] = 1.0
        field_ = np.zeros((2, nj, ni))
        inlier = np.ones(len(ij), bool)
        for _ in range(2):
            for c in range(2):
                vals = np.zeros((nj, ni))
                vals[gj, gi] = r[:, c]
                field_[c] = smooth_grid(vals, weights, cfg.smooth)
            err = np.linalg.norm(r - field_[:, gj, gi].T, axis=1)
            limit = max(3.0 * 1.4826 * np.median(err[inlier]), cfg.outlier_fraction * spacing)
            new_inlier = err <= limit
            if (new_inlier == inlier).all():
                break
            inlier = new_inlier
            weights[:] = 0.0
            weights[gj[inlier], gi[inlier]] = 1.0
        if inlier.sum() < 4:
            raise LatticeError("grid fit rejected almost every point")
        err_in = np.linalg.norm(r[inlier] - field_[:, gj[inlier], gi[inlier]].T, axis=1)
        quality = {
            "nodes": int(inlier.sum()),
            "outliers": int((~inlier).sum()),
            "residual_px": float(np.sqrt(np.mean(err_in**2))),
            "nonplanar_px": float(np.sqrt(np.mean(np.sum(r[inlier] ** 2, 1)))),
            "coverage": float(inlier.sum() / max((i_max - i_min + 1) * (j_max - j_min + 1), 1)),
        }
        return cls(h, field_, i0, j0, (int(i_min), int(i_max), int(j_min), int(j_max)), spacing, quality=quality)


# --------------------------------------------------------------------------------------
# growth of the index lattice from unordered points
# --------------------------------------------------------------------------------------
def _seed_candidates(points: np.ndarray, tree: cKDTree, knn: int) -> list[tuple[float, int, np.ndarray, np.ndarray]]:
    k = min(knn + 1, len(points))
    _, nn = tree.query(points, k=k)
    out = []
    for p in range(len(points)):
        vec = points[nn[p, 1:]] - points[p]
        length = np.linalg.norm(vec, axis=1)
        if len(vec) < 4 or length[0] < 1e-6:
            continue
        a = vec[0]
        la = length[0]
        opp = np.linalg.norm(vec + a, axis=1)
        if opp.min() > 0.3 * la:
            continue
        cos = np.abs(vec @ a) / (length * la + 1e-9)
        ok = (cos < 0.4) & (length > 0.4 * la) & (length < 2.6 * la)
        if not ok.any():
            continue
        cand = np.nonzero(ok)[0]
        b = vec[cand[np.argmin(length[cand])]]
        opp_b = np.linalg.norm(vec + b, axis=1)
        lb = np.linalg.norm(b)
        if opp_b.min() > 0.3 * lb:
            continue
        if a[0] * b[1] - a[1] * b[0] < 0:
            b = -b
        out.append((float(opp.min() / la + opp_b.min() / lb), p, a, b))
    out.sort(key=lambda t: t[0])
    return out


def _grow(points: np.ndarray, tree: cKDTree, seed: int, u0: np.ndarray, v0: np.ndarray, cfg: LatticeConfig):
    n = len(points)
    index = np.full((n, 2), np.iinfo(np.int32).min, dtype=np.int64)
    assigned = np.zeros(n, bool)
    taken: dict[tuple[int, int], int] = {}
    basis_u = np.zeros((n, 2))
    basis_v = np.zeros((n, 2))
    heap: list[tuple[float, int, int, int, int, int]] = []
    counter = 0

    def finalize(q: int, key: tuple[int, int], u: np.ndarray, v: np.ndarray) -> None:
        nonlocal counter
        assigned[q] = True
        index[q] = key
        taken[key] = q
        basis_u[q], basis_v[q] = u, v
        for di, dj, step in ((1, 0, u), (-1, 0, -u), (0, 1, v), (0, -1, -v)):
            nk = (key[0] + di, key[1] + dj)
            if nk in taken:
                continue
            dist, cand = tree.query(points[q] + step)
            err = dist / max(np.linalg.norm(step), 1e-9)
            if err <= cfg.growth_tolerance and not assigned[cand]:
                counter += 1
                heapq.heappush(heap, (float(err), counter, q, int(cand), di, dj))

    finalize(seed, (0, 0), u0, v0)
    while heap:
        _, _, p, q, di, dj = heapq.heappop(heap)
        key = (int(index[p, 0] + di), int(index[p, 1] + dj))
        if assigned[q] or key in taken:
            continue
        actual = points[q] - points[p]
        u, v = basis_u[p].copy(), basis_v[p].copy()
        a = cfg.basis_update
        if di:
            u = (1 - a) * u + a * actual * di
        else:
            v = (1 - a) * v + a * actual * dj
        finalize(q, key, u, v)
    return index, assigned


def build_lattice(
    points: np.ndarray,
    scores: np.ndarray | None = None,
    config: LatticeConfig | None = None,
) -> Lattice:
    """Index unordered grid-intersection detections and fit the paper map.

    Raises :class:`LatticeError` when no consistent grid is found; callers decide on the
    fallback (for example a page without a printed grid).
    """
    cfg = config or LatticeConfig()
    points = np.asarray(points, dtype=np.float64)
    if len(points) < cfg.min_points:
        raise LatticeError(f"only {len(points)} grid points detected")
    tree = cKDTree(points)
    seeds = _seed_candidates(points, tree, cfg.knn)
    if not seeds:
        raise LatticeError("no locally regular grid neighbourhood found")
    if scores is not None and len(seeds) > cfg.max_seeds:
        # prefer regular neighbourhoods that are also confident and central
        centre = np.median(points, 0)
        span = max(np.ptp(points, 0).max(), 1e-9)
        seeds = sorted(seeds, key=lambda t: t[0] + 0.5 * np.linalg.norm(points[t[1]] - centre) / span)
    best_index, best_assigned = None, None
    tried: set[int] = set()
    for _, p, a, b in seeds:
        if len(tried) >= cfg.max_seeds:
            break
        if best_assigned is not None and best_assigned[p]:
            continue
        tried.add(p)
        index, assigned = _grow(points, tree, p, a, b, cfg)
        if best_assigned is None or assigned.sum() > best_assigned.sum():
            best_index, best_assigned = index, assigned
        if best_assigned.sum() > 0.8 * len(points):
            break
    assert best_index is not None and best_assigned is not None
    if best_assigned.sum() < cfg.min_points:
        raise LatticeError(f"grid growth stopped at {int(best_assigned.sum())} points")

    ij, xy = best_index[best_assigned], points[best_assigned]
    lattice = Lattice.from_correspondences(ij, xy, cfg)

    # Re-attach detections that growth could not reach (gaps behind occlusions, tears).
    free = np.nonzero(~best_assigned)[0]
    if len(free):
        i_min, i_max, j_min, j_max = lattice.support
        m = cfg.margin_cells
        gi, gj = np.meshgrid(np.arange(i_min - m, i_max + m + 1), np.arange(j_min - m, j_max + m + 1))
        nodes = np.stack([gi.ravel(), gj.ravel()], 1)
        used = {(int(a), int(b)) for a, b in ij}
        nodes = np.array([nd for nd in nodes if (int(nd[0]), int(nd[1])) not in used])
        if len(nodes):
            pred = lattice.index_to_image(torch.from_numpy(nodes).float()).numpy()
            dist, near = cKDTree(points[free]).query(pred)
            hit = dist <= cfg.reattach_tolerance * lattice.spacing_px
            if hit.any():
                order = np.argsort(dist[hit])
                seen: set[int] = set()
                add_ij, add_xy = [], []
                for k in np.nonzero(hit)[0][order]:
                    if int(near[k]) in seen:
                        continue
                    seen.add(int(near[k]))
                    add_ij.append(nodes[k])
                    add_xy.append(points[free[near[k]]])
                ij = np.concatenate([ij, np.array(add_ij)])
                xy = np.concatenate([xy, np.array(add_xy)])
                lattice = Lattice.from_correspondences(ij, xy, cfg)
    lattice.quality["detections"] = int(len(points))
    lattice.quality["indexed_fraction"] = float(lattice.quality["nodes"] / max(len(points), 1))
    return lattice.oriented(None)
