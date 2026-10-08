"""
Partner protein preparation pipeline for macromolecular (Protein-Protein) docking.
"""

import os
from typing import List, Optional, Set, Tuple
import numpy as np

from moldock.core.atom import Atom
from moldock.core.molecule import Molecule
from moldock.core.pdb_io import PDBIO, PDBQTIO


class PartnerPreparer:
    """
    Prepares a partner protein structure for protein-protein docking.
    """

    @classmethod
    def prepare_partner_from_pdb(
        cls,
        pdb_path: str,
        chain_ids: Optional[Set[str]] = None,
        remove_waters: bool = True,
    ) -> Molecule:
        """
        Load and sanitize partner protein from PDB.
        """
        mol = PDBIO.read_pdb(pdb_path, name="Partner")
        filtered_atoms: List[Atom] = []

        for atom in mol.atoms:
            if remove_waters and atom.res_name.strip().upper() in ("HOH", "WAT", "H2O", "SOL"):
                continue
            if chain_ids is not None and atom.chain_id not in chain_ids:
                continue
            filtered_atoms.append(atom)

        partner_mol = Molecule(name=mol.name, atoms=filtered_atoms)
        partner_mol.metadata["num_rotatable_bonds"] = 0  # Rigid body macromolecule

        # Tag surface residues for interface scoring efficiency
        cls.tag_surface_atoms(partner_mol)
        return partner_mol

    @staticmethod
    def tag_surface_atoms(molecule: Molecule, neighbor_radius: float = 9.0, max_neighbors: int = 18) -> None:
        """
        Estimate solvent-accessible surface atoms using local atom density.
        Atoms with fewer neighbors are exposed to solvent / surface.
        """
        coords = molecule.coords
        n = len(coords)
        if n == 0:
            return

        is_surface = []
        r_sq = neighbor_radius * neighbor_radius

        for i in range(n):
            pt = coords[i]
            diff = coords - pt
            d_sq = np.sum(diff * diff, axis=-1)
            # Count neighbors within sphere (excluding self)
            count = int(np.sum((d_sq <= r_sq) & (d_sq > 0.01)))
            is_surface.append(count <= max_neighbors)

        molecule.metadata["is_surface_atom"] = is_surface

