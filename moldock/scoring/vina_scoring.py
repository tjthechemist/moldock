"""
Vectorized AutoDock Vina empirical scoring function implementation in Python/NumPy.
Based on Trott & Olson (J. Comput. Chem. 2010, 31, 455-461).
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np

from moldock.core.atom import Atom
from moldock.core.molecule import Molecule
from moldock.core.config import ScoringWeights
from moldock.scoring.base import BaseScoringFunction


@dataclass
class VinaScoreComponents:
    gauss1: float = 0.0
    gauss2: float = 0.0
    repulsion: float = 0.0
    hydrophobic: float = 0.0
    hbond: float = 0.0
    unweighted_sum: float = 0.0
    intermolecular_energy: float = 0.0
    torsion_penalty: float = 1.0
    total_score: float = 0.0


class VinaScoringFunction(BaseScoringFunction):
    """
    AutoDock Vina scoring function implementation.
    Evaluates pair-wise empirical terms between receptor and ligand/partner atoms.
    """

    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        cutoff: float = 8.0,
    ):
        self.weights = weights or ScoringWeights()
        self.cutoff = cutoff

    def _prepare_atom_arrays(self, atoms: List[Atom]):
        """Convert list of Atoms into contiguous NumPy arrays for vectorized scoring."""
        n = len(atoms)
        coords = np.zeros((n, 3), dtype=np.float64)
        vdw = np.zeros(n, dtype=np.float64)
        is_donor = np.zeros(n, dtype=bool)
        is_acceptor = np.zeros(n, dtype=bool)
        is_hydrophobic = np.zeros(n, dtype=bool)

        for i, a in enumerate(atoms):
            coords[i] = a.coord
            vdw[i] = a.vdw_radius
            is_donor[i] = a.is_donor
            is_acceptor[i] = a.is_acceptor
            is_hydrophobic[i] = a.is_hydrophobic

        return coords, vdw, is_donor, is_acceptor, is_hydrophobic

    def evaluate_components(
        self,
        receptor_atoms: List[Atom],
        ligand_atoms: List[Atom],
        num_rotatable_bonds: int = 0,
    ) -> VinaScoreComponents:
        """
        Compute individual Vina scoring terms across all interacting atom pairs.
        """
        res = VinaScoreComponents()
        if not receptor_atoms or not ligand_atoms:
            return res

        r_coords, r_vdw, r_don, r_acc, r_hyd = self._prepare_atom_arrays(receptor_atoms)
        l_coords, l_vdw, l_don, l_acc, l_hyd = self._prepare_atom_arrays(ligand_atoms)

        # Pairwise distance matrix (N_rec, M_lig)
        diff = r_coords[:, np.newaxis, :] - l_coords[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)

        # Distance cutoff mask
        mask = dist_sq <= (self.cutoff * self.cutoff)
        if not np.any(mask):
            return res

        dists = np.sqrt(dist_sq[mask])
        r_vdw_grid = np.broadcast_to(r_vdw[:, np.newaxis], dist_sq.shape)[mask]
        l_vdw_grid = np.broadcast_to(l_vdw[np.newaxis, :], dist_sq.shape)[mask]

        # Surface distance d = r_ij - (R_i + R_j)
        d = dists - (r_vdw_grid + l_vdw_grid)

        # 1. Gauss 1 term: exp(-(d / 0.5)^2)
        g1_vals = np.exp(-((d / 0.5) ** 2))
        res.gauss1 = float(np.sum(g1_vals))

        # 2. Gauss 2 term: exp(-((d - 3.0) / 2.0)^2)
        g2_vals = np.exp(-(((d - 3.0) / 2.0) ** 2))
        res.gauss2 = float(np.sum(g2_vals))

        # 3. Repulsion term: d^2 if d < 0 else 0
        rep_mask = d < 0.0
        if np.any(rep_mask):
            res.repulsion = float(np.sum(d[rep_mask] ** 2))

        # 4. Hydrophobic term:
        # Favorable between hydrophobic-hydrophobic pairs
        r_hyd_grid = np.broadcast_to(r_hyd[:, np.newaxis], dist_sq.shape)[mask]
        l_hyd_grid = np.broadcast_to(l_hyd[np.newaxis, :], dist_sq.shape)[mask]
        hyd_pair_mask = r_hyd_grid & l_hyd_grid

        if np.any(hyd_pair_mask):
            d_hyd = d[hyd_pair_mask]
            hyd_vals = np.zeros_like(d_hyd)
            hyd_vals[d_hyd < 0.5] = 1.0
            ramp = (d_hyd >= 0.5) & (d_hyd < 1.5)
            hyd_vals[ramp] = 1.5 - d_hyd[ramp]
            res.hydrophobic = float(np.sum(hyd_vals))

        # 5. Hydrogen bonding term:
        # Between donor and acceptor pairs
        r_don_grid = np.broadcast_to(r_don[:, np.newaxis], dist_sq.shape)[mask]
        l_acc_grid = np.broadcast_to(l_acc[np.newaxis, :], dist_sq.shape)[mask]
        r_acc_grid = np.broadcast_to(r_acc[:, np.newaxis], dist_sq.shape)[mask]
        l_don_grid = np.broadcast_to(l_don[np.newaxis, :], dist_sq.shape)[mask]

        hbond_pair_mask = (r_don_grid & l_acc_grid) | (r_acc_grid & l_don_grid)
        if np.any(hbond_pair_mask):
            d_hb = d[hbond_pair_mask]
            hb_vals = np.zeros_like(d_hb)
            hb_vals[d_hb < -0.7] = 1.0
            ramp = (d_hb >= -0.7) & (d_hb < 0.0)
            hb_vals[ramp] = -d_hb[ramp] / 0.7
            res.hbond = float(np.sum(hb_vals))

        # Weighted intermolecular energy
        e_inter = (
            self.weights.weight_gauss1 * res.gauss1
            + self.weights.weight_gauss2 * res.gauss2
            + self.weights.weight_repulsion * res.repulsion
            + self.weights.weight_hydrophobic * res.hydrophobic
            + self.weights.weight_hbond * res.hbond
        )
        res.intermolecular_energy = float(e_inter)

        # Flexibility penalty (Trott & Olson 2010): 1 / (1 + w_rot * N_rot)
        rot_factor = 1.0 / (1.0 + self.weights.weight_rot * num_rotatable_bonds)
        res.torsion_penalty = rot_factor
        res.total_score = float(e_inter * rot_factor)

        return res

    def score(self, receptor: Molecule, ligand: Molecule) -> float:
        """Calculate total binding energy in kcal/mol."""
        num_rot = ligand.metadata.get("num_rotatable_bonds", 0)
        # In molecular docking, hydrogen atoms are either implicit or non-essential in Vina empirical terms
        r_heavy = receptor.get_heavy_atoms()
        l_heavy = ligand.get_heavy_atoms()
        comp = self.evaluate_components(r_heavy, l_heavy, num_rotatable_bonds=num_rot)
        return comp.total_score

    def score_breakdown(self, receptor: Molecule, ligand: Molecule) -> Dict[str, Any]:
        """Detailed breakdown of Vina energy components."""
        num_rot = ligand.metadata.get("num_rotatable_bonds", 0)
        r_heavy = receptor.get_heavy_atoms()
        l_heavy = ligand.get_heavy_atoms()
        comp = self.evaluate_components(r_heavy, l_heavy, num_rotatable_bonds=num_rot)
        return {
            "total_score": comp.total_score,
            "intermolecular_energy": comp.intermolecular_energy,
            "gauss1": comp.gauss1,
            "gauss2": comp.gauss2,
            "repulsion": comp.repulsion,
            "hydrophobic": comp.hydrophobic,
            "hbond": comp.hbond,
            "torsion_penalty": comp.torsion_penalty,
        }

