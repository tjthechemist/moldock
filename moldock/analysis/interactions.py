"""
Intermolecular interaction profiler:
Hydrogen bonds, salt bridges, hydrophobic contacts, and metal coordination.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from moldock.core.atom import Atom
from moldock.core.molecule import Molecule


@dataclass
class InteractionContact:
    interaction_type: str  # "hbond", "salt_bridge", "hydrophobic", "metal_coordination"
    receptor_atom_idx: int
    receptor_res: str
    receptor_res_seq: int
    receptor_atom_name: str
    query_atom_idx: int
    query_atom_name: str
    query_element: str
    distance: float  # Å


class InteractionProfiler:
    """
    Identifies non-covalent interactions and metal coordination bonds at the binding interface.
    """

    def __init__(
        self,
        hbond_cutoff: float = 3.3,
        salt_bridge_cutoff: float = 4.0,
        hydrophobic_cutoff: float = 4.0,
        metal_cutoff: float = 2.85,
    ):
        self.hbond_cutoff = hbond_cutoff
        self.salt_bridge_cutoff = salt_bridge_cutoff
        self.hydrophobic_cutoff = hydrophobic_cutoff
        self.metal_cutoff = metal_cutoff

    def profile_interactions(
        self,
        receptor: Molecule,
        query: Molecule,
    ) -> List[InteractionContact]:
        """
        Scan all receptor-query atom pairs for specific interaction types.
        """
        interactions: List[InteractionContact] = []
        r_atoms = receptor.atoms
        q_atoms = query.atoms

        for r in r_atoms:
            for q in q_atoms:
                dist = float(np.linalg.norm(r.coord - q.coord))

                # 1. Metal coordination
                if r.is_metal and q.element.strip().upper() in ("O", "N", "S", "CL", "F"):
                    if dist <= self.metal_cutoff:
                        interactions.append(
                            InteractionContact(
                                interaction_type="metal_coordination",
                                receptor_atom_idx=r.index,
                                receptor_res=r.res_name,
                                receptor_res_seq=r.res_seq,
                                receptor_atom_name=r.name,
                                query_atom_idx=q.index,
                                query_atom_name=q.name,
                                query_element=q.element,
                                distance=dist,
                            )
                        )
                    continue

                # 2. Hydrogen bonding (polar heavy atom donor-acceptor)
                if (
                    ((r.is_donor and q.is_acceptor) or (r.is_acceptor and q.is_donor))
                    and dist <= self.hbond_cutoff
                ):
                    interactions.append(
                        InteractionContact(
                            interaction_type="hbond",
                            receptor_atom_idx=r.index,
                            receptor_res=r.res_name,
                            receptor_res_seq=r.res_seq,
                            receptor_atom_name=r.name,
                            query_atom_idx=q.index,
                            query_atom_name=q.name,
                            query_element=q.element,
                            distance=dist,
                        )
                    )
                    continue

                # 3. Salt bridges (oppositely charged groups)
                if (
                    r.charge * q.charge < -0.15
                    and dist <= self.salt_bridge_cutoff
                ):
                    interactions.append(
                        InteractionContact(
                            interaction_type="salt_bridge",
                            receptor_atom_idx=r.index,
                            receptor_res=r.res_name,
                            receptor_res_seq=r.res_seq,
                            receptor_atom_name=r.name,
                            query_atom_idx=q.index,
                            query_atom_name=q.name,
                            query_element=q.element,
                            distance=dist,
                        )
                    )
                    continue

                # 4. Hydrophobic contacts
                if (
                    r.is_hydrophobic and q.is_hydrophobic
                    and dist <= self.hydrophobic_cutoff
                ):
                    interactions.append(
                        InteractionContact(
                            interaction_type="hydrophobic",
                            receptor_atom_idx=r.index,
                            receptor_res=r.res_name,
                            receptor_res_seq=r.res_seq,
                            receptor_atom_name=r.name,
                            query_atom_idx=q.index,
                            query_atom_name=q.name,
                            query_element=q.element,
                            distance=dist,
                        )
                    )

        return interactions

