"""
Detection and coordination sphere characterization of metal ion cofactors in proteins.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
from moldock.core.atom import Atom, KNOWN_METALS
from moldock.core.molecule import Molecule
from moldock.metals.parameters import get_metal_parameters, MetalProperties


@dataclass
class CoordinatedResidue:
    atom_name: str
    element: str
    res_name: str
    res_seq: int
    chain_id: str
    distance: float


@dataclass
class MetalCenter:
    metal_atom: Atom
    properties: MetalProperties
    coordinating_atoms: List[CoordinatedResidue] = field(default_factory=list)
    cluster_id: int = 0

    @property
    def coordination_number(self) -> int:
        return len(self.coordinating_atoms)

    @property
    def available_valency(self) -> int:
        """Estimate unoccupied coordination sites available to bind ligand/partner."""
        rem = self.properties.preferred_coord_num - self.coordination_number
        return max(rem, 1)  # At least 1 slot open for docking interaction


class MetalDetector:
    """
    Automated detection and analysis of metal ion co-factors from macromolecular structures.
    """

    def __init__(self, coordination_cutoff: float = 2.85):
        self.coordination_cutoff = coordination_cutoff

    def detect_metal_centers(
        self,
        receptor: Molecule,
        allowed_metals: Optional[List[str]] = None,
    ) -> List[MetalCenter]:
        """
        Identify all metal ions in the receptor and map their coordination environments.
        """
        allowed = set(m.upper() for m in allowed_metals) if allowed_metals else KNOWN_METALS
        metal_atoms: List[Atom] = []

        for atom in receptor.atoms:
            elem = atom.element.strip().upper()
            res = atom.res_name.strip().upper()
            if (elem in allowed or res in allowed) and (atom.is_metal or atom.is_hetatm):
                metal_atoms.append(atom)

        if not metal_atoms:
            return []

        # Find coordinating protein atoms (typically O, N, S from protein residues)
        protein_atoms = [
            a for a in receptor.atoms
            if not a.is_metal and a.element.strip().upper() in ("O", "N", "S")
        ]

        metal_centers: List[MetalCenter] = []
        for idx, metal in enumerate(metal_atoms):
            elem = metal.element.strip().upper()
            if elem not in KNOWN_METALS:
                elem = metal.res_name.strip().upper()

            props = get_metal_parameters(elem)
            center = MetalCenter(metal_atom=metal, properties=props, cluster_id=idx)

            for p_atom in protein_atoms:
                dist = float(np.linalg.norm(metal.coord - p_atom.coord))
                if dist <= self.coordination_cutoff:
                    coord_res = CoordinatedResidue(
                        atom_name=p_atom.name,
                        element=p_atom.element.upper(),
                        res_name=p_atom.res_name,
                        res_seq=p_atom.res_seq,
                        chain_id=p_atom.chain_id,
                        distance=dist,
                    )
                    center.coordinating_atoms.append(coord_res)

            metal_centers.append(center)

        # Cluster multi-metal centers (e.g. dinuclear Zn or di-iron centers within 4.5 Å)
        cluster_counter = 0
        assigned = {}
        for i, c1 in enumerate(metal_centers):
            if i in assigned:
                continue
            assigned[i] = cluster_counter
            c1.cluster_id = cluster_counter
            for j, c2 in enumerate(metal_centers):
                if j in assigned:
                    continue
                d = float(np.linalg.norm(c1.metal_atom.coord - c2.metal_atom.coord))
                if d <= 4.5:
                    assigned[j] = cluster_counter
                    c2.cluster_id = cluster_counter
            cluster_counter += 1

        return metal_centers
