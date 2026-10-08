"""
Composite scoring function combining AutoDock Vina empirical terms,
metal ion coordination/polarization, and electrostatics.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from moldock.core.molecule import Molecule
from moldock.core.config import ScoringWeights, MetalConfig
from moldock.metals.detector import MetalDetector, MetalCenter
from moldock.metals.scoring import MetalCoordinationScorer, MetalScoreBreakdown
from moldock.scoring.base import BaseScoringFunction
from moldock.scoring.vina_scoring import VinaScoringFunction


class CompositeScoringFunction(BaseScoringFunction):
    """
    Unified scoring function:
        E_total = E_vina + E_metal + E_elec
    where E_metal handles one or multiple metal cofactors with coordinate-covalent wells
    and 12-6-4 polarization, and E_vina calculates standard AutoDock Vina terms.
    """

    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        metal_config: Optional[MetalConfig] = None,
        metal_centers: Optional[List[MetalCenter]] = None,
    ):
        self.weights = weights or ScoringWeights()
        self.metal_config = metal_config or MetalConfig()
        self.vina_scorer = VinaScoringFunction(weights=self.weights)
        self.metal_centers = metal_centers or []
        self.metal_scorer = MetalCoordinationScorer(
            metal_centers=self.metal_centers,
            weight=abs(self.weights.weight_metal),
            cutoff=self.metal_config.coordination_cutoff + 0.6,
            use_12_6_4=self.metal_config.use_12_6_4_potential,
        )

    def set_metal_centers(self, metal_centers: List[MetalCenter]) -> None:
        """Update metal centers detected in receptor."""
        self.metal_centers = metal_centers
        self.metal_scorer.metal_centers = metal_centers

    def compute_electrostatics(self, receptor: Molecule, ligand: Molecule) -> float:
        """
        Distance-dependent dielectric Coulombic electrostatic term:
        eps(r) = 4 * r (standard Mehler-Solmajer / AutoDock distance-dependent screening).
        """
        r_charged = [a for a in receptor.atoms if abs(a.charge) > 0.05]
        l_charged = [a for a in ligand.atoms if abs(a.charge) > 0.05]

        if not r_charged or not l_charged:
            return 0.0

        r_coords = np.array([a.coord for a in r_charged])
        l_coords = np.array([a.coord for a in l_charged])
        r_q = np.array([a.charge for a in r_charged])
        l_q = np.array([a.charge for a in l_charged])

        # Distance matrix
        diff = r_coords[:, np.newaxis, :] - l_coords[np.newaxis, :, :]
        r_mat = np.sqrt(np.sum(diff * diff, axis=-1))

        # Avoid singularity
        r_safe = np.maximum(r_mat, 1.0)
        # Coulomb constant in kcal*Å/(mol*e^2) ~ 332.06
        # With distance dependent dielectric epsilon(r) = 4 * r:
        # E = 332.06 * q1 * q2 / (4 * r^2)
        qq = r_q[:, np.newaxis] * l_q[np.newaxis, :]
        e_elec = 332.06 * qq / (4.0 * (r_safe ** 2))

        # Sum within 12 Å cutoff
        mask = r_mat <= 12.0
        return float(np.sum(e_elec[mask])) * self.weights.weight_electrostatic

    def score(self, receptor: Molecule, ligand: Molecule) -> float:
        """Calculate combined binding affinity (kcal/mol)."""
        vina_score = self.vina_scorer.score(receptor, ligand)
        metal_breakdown = self.metal_scorer.score(ligand.atoms)
        elec_score = self.compute_electrostatics(receptor, ligand)

        total = vina_score + metal_breakdown.total_score + elec_score
        return float(total)

    def score_breakdown(self, receptor: Molecule, ligand: Molecule) -> Dict[str, Any]:
        """Comprehensive breakdown across Vina terms and metal cofactors."""
        vina_dict = self.vina_scorer.score_breakdown(receptor, ligand)
        metal_breakdown = self.metal_scorer.score(ligand.atoms)
        elec_score = self.compute_electrostatics(receptor, ligand)

        total = (
            vina_dict["total_score"]
            + metal_breakdown.total_score
            + elec_score
        )

        return {
            "total_score": float(total),
            "vina_score": vina_dict["total_score"],
            "vina_breakdown": vina_dict,
            "metal_score": metal_breakdown.total_score,
            "metal_coordination_energy": metal_breakdown.coordination_energy,
            "metal_steric_penalty": metal_breakdown.steric_penalty,
            "metal_polarization": metal_breakdown.polarization_12_6_4,
            "metal_interactions_count": len(metal_breakdown.interactions),
            "per_metal_scores": metal_breakdown.per_metal_scores,
            "electrostatic_score": elec_score,
        }

