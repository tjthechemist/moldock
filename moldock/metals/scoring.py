"""
Scoring terms for metal-ligand and metal-protein coordination interactions.
Supports multi-metal sites, 12-6-4 Lennard-Jones-Coulomb potential, and coordination geometry wells.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
from moldock.core.atom import Atom
from moldock.metals.detector import MetalCenter
from moldock.metals.parameters import MetalProperties, get_metal_parameters


@dataclass
class MetalInteraction:
    metal_index: int
    metal_element: str
    donor_index: int
    donor_element: str
    distance: float
    energy: float


@dataclass
class MetalScoreBreakdown:
    total_score: float = 0.0
    coordination_energy: float = 0.0
    steric_penalty: float = 0.0
    polarization_12_6_4: float = 0.0
    interactions: List[MetalInteraction] = field(default_factory=list)
    per_metal_scores: Dict[int, float] = field(default_factory=dict)


class MetalCoordinationScorer:
    """
    Evaluates favorable coordinate-covalent, electrostatic, and polarization
    interactions between metal co-factors and incoming ligand or protein partner atoms.
    """

    def __init__(
        self,
        metal_centers: List[MetalCenter],
        weight: float = 1.25,
        cutoff: float = 3.5,
        use_12_6_4: bool = True,
    ):
        self.metal_centers = metal_centers
        self.weight = weight
        self.cutoff = cutoff
        self.use_12_6_4 = use_12_6_4

    def score(self, query_atoms: List[Atom]) -> MetalScoreBreakdown:
        """
        Calculate metal coordination score between the receptor's metal co-factors
        and the query molecule (ligand or docked partner protein).
        Lower (more negative) score indicates stronger favorable binding.
        """
        breakdown = MetalScoreBreakdown()
        if not self.metal_centers or not query_atoms:
            return breakdown

        # Filter query atoms for potential Lewis base donor atoms (O, N, S, halogens)
        donor_candidates = [
            a for a in query_atoms
            if a.element.strip().upper() in ("O", "N", "S", "CL", "F", "BR")
        ]
        if not donor_candidates:
            return breakdown

        coord_energy_total = 0.0
        repulsion_total = 0.0
        pol_total = 0.0

        for center in self.metal_centers:
            m_atom = center.metal_atom
            m_props = center.properties
            m_elem = m_props.element
            metal_idx = m_atom.index

            metal_score = 0.0
            donor_interactions_for_center = []

            for donor in donor_candidates:
                d_elem = donor.element.strip().upper()
                r = float(np.linalg.norm(m_atom.coord - donor.coord))
                if r > self.cutoff:
                    continue

                r0 = m_props.ideal_distances.get(d_elem, 2.10)
                sigma_coord = 0.35  # Å Gaussian dispersion

                # 1. Gaussian Coordination Attractive Well
                # Peaks at r = r0 with value -1.0 * weight
                coord_term = -float(np.exp(-0.5 * ((r - r0) / sigma_coord) ** 2))

                # 2. Steric Repulsion (if distance is unphysically short)
                r_rep_onset = r0 - 0.35
                rep_term = 0.0
                if r < r_rep_onset:
                    rep_term = 25.0 * ((r_rep_onset - r) ** 2)

                # 3. 12-6-4 Polarization Potential
                pol_term = 0.0
                if self.use_12_6_4 and r > 1.2:
                    # C4 term represents charge-induced dipole polarization
                    c4 = m_props.c4_polarization
                    pol_term = - (c4 / (r ** 4)) * 0.01  # scaled

                pair_energy = (self.weight * coord_term) + rep_term + pol_term
                donor_interactions_for_center.append((pair_energy, r, donor, coord_term, rep_term, pol_term))

            # Limit the number of favorable coordinate bonds to the available valency
            # of the metal center to avoid unphysical multi-coordination
            donor_interactions_for_center.sort(key=lambda x: x[0])
            max_slots = center.available_valency

            for rank, item in enumerate(donor_interactions_for_center):
                pair_e, r, donor, c_t, rep_t, pol_t = item
                # Diminishing returns if exceeding preferred coordination capacity
                scale = 1.0 if rank < max_slots else 0.25

                eff_energy = pair_e * scale
                coord_energy_total += (self.weight * c_t) * scale
                repulsion_total += rep_t
                pol_total += pol_t * scale
                metal_score += eff_energy

                breakdown.interactions.append(
                    MetalInteraction(
                        metal_index=metal_idx,
                        metal_element=m_elem,
                        donor_index=donor.index,
                        donor_element=donor.element.upper(),
                        distance=r,
                        energy=eff_energy,
                    )
                )

            breakdown.per_metal_scores[metal_idx] = metal_score

        breakdown.coordination_energy = coord_energy_total
        breakdown.steric_penalty = repulsion_total
        breakdown.polarization_12_6_4 = pol_total
        breakdown.total_score = coord_energy_total + repulsion_total + pol_total

        return breakdown
