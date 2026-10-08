"""
Root-Mean-Square Deviation (RMSD) computation and Kabsch alignment.
"""

from typing import Tuple
import numpy as np
from moldock.core.molecule import Molecule


def calculate_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """
    Standard coordinates RMSD between two ordered sets of 3D coordinates.
    """
    diff = coords_a - coords_b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))


def calculate_heavy_atom_rmsd(mol_a: Molecule, mol_b: Molecule) -> float:
    """
    Calculate RMSD across matching heavy atoms.
    """
    heavy_a = [a.coord for a in mol_a.get_heavy_atoms()]
    heavy_b = [a.coord for a in mol_b.get_heavy_atoms()]
    if len(heavy_a) != len(heavy_b) or len(heavy_a) == 0:
        return 0.0
    return calculate_rmsd(np.array(heavy_a), np.array(heavy_b))


def kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute minimal RMSD after optimal rigid-body rotation and translation using Kabsch algorithm.
    Returns: (min_rmsd, rotated_and_translated_coords_b)
    """
    a = np.asarray(coords_a, dtype=np.float64)
    b = np.asarray(coords_b, dtype=np.float64)

    centroid_a = np.mean(a, axis=0)
    centroid_b = np.mean(b, axis=0)

    a_centered = a - centroid_a
    b_centered = b - centroid_b

    # Covariance matrix H
    H = np.dot(b_centered.T, a_centered)
    U, S, Vt = np.linalg.svd(H)

    # Rotation matrix R
    R = np.dot(Vt.T, U.T)
    # Ensure a right-handed coordinate system (det = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    b_aligned = np.dot(b_centered, R.T) + centroid_a
    rmsd = calculate_rmsd(a, b_aligned)
    return rmsd, b_aligned

