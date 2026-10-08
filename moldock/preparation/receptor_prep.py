"""
Receptor preparation pipeline: cleaning, protonation, preserving metal cofactors,
and exporting to PDB/PDBQT.
"""

import os
from typing import List, Optional, Tuple
import numpy as np

from moldock.core.atom import Atom, assign_vina_atom_type
from moldock.core.molecule import Molecule
from moldock.core.pdb_io import PDBIO, PDBQTIO
from moldock.core.config import MetalConfig
from moldock.metals.detector import MetalDetector, MetalCenter


class ReceptorPreparer:
    """
    Cleans macromolecular receptors from PDB files,
    explicitly preserving and parameterizing active-site metal cofactors.
    """

    def __init__(self, metal_config: Optional[MetalConfig] = None):
        self.metal_config = metal_config or MetalConfig()
        self.detector = MetalDetector(coordination_cutoff=self.metal_config.coordination_cutoff)

    def prepare_from_pdb(
        self,
        pdb_path: str,
        remove_waters: bool = True,
        remove_non_cofactor_hetatms: bool = True,
    ) -> Tuple[Molecule, List[MetalCenter]]:
        """
        Load and prepare receptor structure from a PDB file:
        - Extracts and preserves allowed metal co-factors.
        - Strips crystallographic waters and bulk solvents.
        - Assigns AutoDock atom types and partial charges.
        - Detects coordination spheres for all metal centers.
        """
        raw_mol = PDBIO.read_pdb(pdb_path, name="Receptor")
        prepared_atoms: List[Atom] = []

        allowed_metals = set(m.upper() for m in self.metal_config.allowed_metals)
        metal_centers: List[MetalCenter] = []

        # First pass: identify metal centers and coordinating environment
        raw_metal_centers = self.detector.detect_metal_centers(raw_mol, self.metal_config.allowed_metals)

        for atom in raw_mol.atoms:
            elem = atom.element.strip().upper()
            res = atom.res_name.strip().upper()

            # Water removal
            if remove_waters and res in ("HOH", "WAT", "H2O", "SOL", "TIP3"):
                continue

            # Metal check
            is_allowed_metal = (elem in allowed_metals or res in allowed_metals)
            if is_allowed_metal:
                if self.metal_config.retain_in_receptor:
                    # Assign formal / effective partial charge for metal ion
                    charge = self.metal_config.custom_charges.get(elem, self.metal_config.custom_charges.get(res, 2.0))
                    atom.charge = charge
                    atom.atom_type = elem if elem in allowed_metals else res
                    atom.is_hetatm = True
                    prepared_atoms.append(atom)
                continue

            # Non-metal HETATMs (e.g. buffer molecules, crystallization agents, co-crystallized small ligands)
            if remove_non_cofactor_hetatms and atom.is_hetatm:
                # Common crystallization agents to exclude
                if res in ("SO4", "PO4", "GOL", "EDO", "DMS", "PEG", "ACT", "FMT"):
                    continue
                # If it's a known ligand, remove it to leave the apo binding pocket ready
                continue

            # Protein amino acids
            # Assign standard partial charges for basic/acidic residues
            if res in ("ASP", "GLU") and atom.name in ("OD1", "OD2", "OE1", "OE2"):
                atom.charge = -0.5
            elif res in ("LYS", "ARG") and atom.name in ("NZ", "NH1", "NH2"):
                atom.charge = +0.5
            elif res == "HIS" and atom.name in ("ND1", "NE2"):
                atom.charge = +0.1

            prepared_atoms.append(atom)

        prepared_mol = Molecule(name=raw_mol.name, atoms=prepared_atoms)

        # Re-detect metal centers on the cleaned receptor structure
        metal_centers = self.detector.detect_metal_centers(prepared_mol, self.metal_config.allowed_metals)

        return prepared_mol, metal_centers

    def save_prepared(self, molecule: Molecule, output_path: str) -> str:
        """Save prepared receptor as .pdb or .pdbqt."""
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".pdbqt":
            PDBQTIO.write_pdbqt(molecule, output_path, torsdof=0, as_rigid=True)
        else:
            PDBIO.write_pdb(molecule, output_path)
        return output_path
