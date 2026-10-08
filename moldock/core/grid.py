"""
Grid box utilities, active site spatial partitioning, and distance calculations.
"""

from typing import List, Tuple, Optional
import numpy as np
from moldock.core.atom import Atom
from moldock.core.molecule import Molecule


class SpatialGrid:
    """
    3D spatial grid / cell linked-list for fast neighbor lookups
    without requiring external C libraries.
    """

    def __init__(self, cutoff: float = 8.0):
        self.cutoff = cutoff
        self.cells: dict = {}
        self.coords: Optional[np.ndarray] = None
        self.atom_indices: Optional[np.ndarray] = None

    def build(self, coords: np.ndarray) -> None:
        """Partition 3D coordinates into grid cubes of edge length `cutoff`."""
        self.coords = np.asarray(coords, dtype=np.float64)
        self.cells = {}
        if len(self.coords) == 0:
            return

        cell_coords = np.floor(self.coords / self.cutoff).astype(int)
        for idx, cell in enumerate(cell_coords):
            key = (int(cell[0]), int(cell[1]), int(cell[2]))
            if key not in self.cells:
                self.cells[key] = []
            self.cells[key].append(idx)

    def query_pairs_within_cutoff(self, query_coords: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Find all (query_idx, target_idx, distance) pairs where distance <= cutoff.
        """
        query_coords = np.asarray(query_coords, dtype=np.float64)
        pairs = []
        if self.coords is None or len(self.coords) == 0 or len(query_coords) == 0:
            return pairs

        cell_coords = np.floor(query_coords / self.cutoff).astype(int)
        cutoff_sq = self.cutoff * self.cutoff

        # Neighbor cell offsets in 3D (27 adjacent cells)
        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]

        for q_idx, q_pt in enumerate(query_coords):
            c = cell_coords[q_idx]
            for dx, dy, dz in offsets:
                neighbor_key = (int(c[0] + dx), int(c[1] + dy), int(c[2] + dz))
                target_indices = self.cells.get(neighbor_key)
                if target_indices is None:
                    continue

                for t_idx in target_indices:
                    t_pt = self.coords[t_idx]
                    diff = q_pt - t_pt
                    d_sq = np.dot(diff, diff)
                    if d_sq <= cutoff_sq:
                        pairs.append((q_idx, t_idx, float(np.sqrt(d_sq))))

        return pairs


def compute_pairwise_distances(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix between coords_a (N, 3) and coords_b (M, 3).
    Returns (N, M) matrix.
    """
    coords_a = np.asarray(coords_a, dtype=np.float64)
    coords_b = np.asarray(coords_b, dtype=np.float64)
    # Using Euclidean distance with broadcasting
    diff = coords_a[:, np.newaxis, :] - coords_b[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))
