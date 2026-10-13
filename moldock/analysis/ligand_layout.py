"""
2D depiction layout for small-molecule ligands.

Rendering a ligand by dropping the Z coordinate produces a tangled picture:
atoms that sit above or below the plane land on top of each other and the
molecule becomes unreadable.  This module builds a clean, flat depiction
instead:

1. **Best-fit plane projection** — a principal-component analysis of the 3D
   coordinates chooses the plane that shows the molecule with maximum spread,
   so the fewest atoms are hidden behind one another.
2. **Overlap relaxation** — a short force-directed pass pushes any remaining
   clashing atoms apart while keeping bonded atoms at a regular bond length,
   guaranteeing that every atom is individually visible.

The result is a set of 2D coordinates in "drawing units" together with the
on-screen bond length, which the renderer uses to size atoms and labels.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from moldock.core.atom import Atom
from moldock.analysis.bond_orders import BondKey, bond_key

Point = Tuple[float, float]

# Fraction of the drawing box the depiction is allowed to use
BOX_FILL = 0.92
# Non-bonded atoms closer than this fraction of a bond length repel each other
MIN_SEPARATION_FACTOR = 0.62
# Force-directed relaxation settings
RELAX_ITERATIONS = 260
RELAX_DAMPING = 0.35
RELAX_STEP = 0.5
# Bond length spring: pull towards the median bond length with this strength
BOND_SPRING = 0.22
MIN_BOND_UNITS = 1e-3


def _principal_plane(coords: np.ndarray) -> np.ndarray:
    """Return the (N, 2) projection of *coords* onto its best-fit plane."""
    if coords.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)

    centered = coords - coords.mean(axis=0)
    if coords.shape[0] < 3 or np.allclose(centered, 0.0):
        # Degenerate: fall back to the XY plane
        return centered[:, :2].copy()

    # Covariance eigenvectors, ascending eigenvalue order
    covariance = centered.T @ centered
    try:
        _values, vectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return centered[:, :2].copy()

    # Two largest eigenvalues span the best plane (columns 2 and 1)
    basis = vectors[:, [2, 1]]

    # Deterministic orientation: make the largest-magnitude entry positive
    for col in range(basis.shape[1]):
        column = basis[:, col]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0:
            basis[:, col] = -column

    return centered @ basis


def _initial_bond_length(pts: np.ndarray, clean_bonds: Sequence[BondKey]) -> float:
    """Median length of the bonded pairs in the projected layout."""
    if not clean_bonds:
        return 1.0
    lengths = []
    for i, j in clean_bonds:
        lengths.append(float(np.linalg.norm(pts[j] - pts[i])))
    lengths = [d for d in lengths if d > MIN_BOND_UNITS]
    if not lengths:
        return 1.0
    return float(np.median(lengths))


def _break_coincidences(pts: np.ndarray, tolerance: float = 1e-4) -> None:
    """Nudge atoms that project onto the exact same point (in place).

    Repulsion needs a direction to act in, and a perfectly coincident pair has
    none, so each such pair is given a small deterministic offset.
    """
    rng = np.random.default_rng(0)
    n_atoms = pts.shape[0]
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if float(np.linalg.norm(pts[j] - pts[i])) < tolerance:
                offset = rng.normal(scale=0.15, size=2)
                pts[i] -= offset
                pts[j] += offset


def _non_bonded_pairs(n_atoms: int, bonded: set) -> Tuple[np.ndarray, np.ndarray]:
    """Arrays of (i, j) index pairs for every non-bonded atom pair."""
    rows: List[int] = []
    cols: List[int] = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if (i, j) in bonded:
                continue
            rows.append(i)
            cols.append(j)
    return np.array(rows, dtype=int), np.array(cols, dtype=int)


def relax_layout(
    pts: np.ndarray,
    bonds: Sequence[Tuple[int, int]],
    iterations: int = RELAX_ITERATIONS,
) -> np.ndarray:
    """Separate clashing atoms while regularising bond lengths.

    Returns a new (N, 2) array.  Atoms hidden on top of each other in the raw
    projection are pushed apart until every atom is individually visible, while
    bonded atoms are settled onto a common bond length so the skeleton still
    reads as a molecule.
    """
    n_atoms = pts.shape[0]
    if n_atoms < 2:
        return pts.copy()

    clean_bonds = [bond_key(i, j) for i, j in bonds
                   if 0 <= i < n_atoms and 0 <= j < n_atoms and i != j]
    bonded = set(clean_bonds)

    current = pts.astype(np.float64).copy()
    if not np.all(np.isfinite(current)):
        current = np.zeros_like(current)

    _break_coincidences(current)

    ideal = _initial_bond_length(current, clean_bonds)
    min_sep = ideal * MIN_SEPARATION_FACTOR
    pair_i, pair_j = _non_bonded_pairs(n_atoms, bonded)

    for _ in range(iterations):
        shift = np.zeros_like(current)

        # Repulsion between non-bonded atoms keeps the depiction legible
        if pair_i.size:
            delta = current[pair_j] - current[pair_i]
            distances = np.linalg.norm(delta, axis=1)
            too_close = distances < min_sep
            if np.any(too_close):
                safe = np.maximum(distances, MIN_BOND_UNITS)
                push = np.where(too_close, (min_sep - safe) * RELAX_STEP, 0.0)
                direction = delta / safe[:, None]
                contribution = direction * push[:, None]
                np.add.at(shift, pair_i, -contribution)
                np.add.at(shift, pair_j, contribution)

        # Bond springs pull each bonded pair onto a uniform bond length
        for i, j in clean_bonds:
            delta = current[j] - current[i]
            distance = float(np.linalg.norm(delta))
            if distance < MIN_BOND_UNITS:
                continue
            correction = (distance - ideal) * BOND_SPRING
            direction = delta / distance
            shift[i] += direction * correction
            shift[j] -= direction * correction

        current += shift * RELAX_DAMPING

    return current


def _fit_to_box(
    pts: np.ndarray,
    box_width: float,
    box_height: float,
) -> Tuple[np.ndarray, float]:
    """Scale *pts* to fill the box and return (scaled, scale factor)."""
    if pts.shape[0] == 0:
        return pts, 1.0

    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = np.where(mx - mn < MIN_BOND_UNITS, 1.0, mx - mn)

    scale = min(box_width * BOX_FILL / span[0], box_height * BOX_FILL / span[1])
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    return pts * scale, float(scale)


def layout_ligand_depiction(
    atoms: Sequence[Atom],
    bonds: Sequence[Tuple[int, int]],
    box_width: float,
    box_height: float,
    center_x: float,
    center_y: float,
) -> Tuple[List[Point], float]:
    """Lay a ligand out as a flat 2D depiction.

    Parameters
    ----------
    atoms:
        Ligand atoms; only their 3D coordinates are used.
    bonds:
        Pairs of 0-based atom indices describing the molecular graph.
    box_width, box_height:
        Size of the drawing region in pixels.
    center_x, center_y:
        Canvas position of the depiction centre.

    Returns
    -------
    positions:
        One ``(x, y)`` pixel position per atom, in atom order.
    bond_pixels:
        On-screen length of a typical bond, used to size atom glyphs so they
        can never overlap.
    """
    if not atoms:
        return [], 1.0

    coords = np.array([a.coord for a in atoms], dtype=np.float64)
    pts = _principal_plane(coords)
    pts = relax_layout(pts, bonds)
    scaled, scale = _fit_to_box(pts, box_width, box_height)

    # Centre the depiction on the requested canvas position
    if scaled.shape[0]:
        span = scaled.max(axis=0) - scaled.min(axis=0)
        offset = np.array([center_x, center_y], dtype=np.float64) - (
            scaled.min(axis=0) + span / 2.0
        )
        scaled = scaled + offset

    # `scaled` is already in pixels, so the median bonded distance is the
    # on-screen bond length the renderer uses to size atom glyphs.
    clean_bonds = [bond_key(i, j) for i, j in bonds
                   if 0 <= i < len(atoms) and 0 <= j < len(atoms)]
    bond_pixels = max(_initial_bond_length(scaled, clean_bonds), 1.0)

    return [(float(x), float(y)) for x, y in scaled], bond_pixels
