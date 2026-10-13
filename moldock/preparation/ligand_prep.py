"""
Ligand preparation pipeline: 3D coordinate parsing, rotatable bond detection,
donor/acceptor identification, and torsion tree representation.
"""

import os
from typing import List, Tuple, Dict, Set, Optional
import numpy as np

from moldock.core.atom import Atom, assign_vina_atom_type
from moldock.core.molecule import Molecule
from moldock.core.pdb_io import PDBIO, PDBQTIO
from moldock.core.sdf_io import SDFIO


class LigandPreparer:
    """
    Prepares small-molecule ligands for docking:
    - Builds molecular connectivity graph.
    - Identifies rotatable bonds for flexible conformational sampling.
    - Detects metal-coordinating Lewis donor atoms (O, N, S).
    - Sets up PDBQT torsion trees.
    """

    @staticmethod
    def infer_bonds(atoms: List[Atom]) -> List[Tuple[int, int]]:
        """
        Infer chemical bonds based on interatomic distances and covalent radii.
        """
        bonds = []
        n = len(atoms)
        if n < 2:
            return bonds

        covalent_radii = {
            "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66,
            "F": 0.57, "P": 1.07, "S": 1.05, "CL": 1.02,
            "BR": 1.20, "I": 1.39
        }

        for i in range(n):
            elem_i = atoms[i].element.upper()
            r_i = covalent_radii.get(elem_i, 0.75)
            for j in range(i + 1, n):
                elem_j = atoms[j].element.upper()
                r_j = covalent_radii.get(elem_j, 0.75)

                dist = float(np.linalg.norm(atoms[i].coord - atoms[j].coord))
                # Bond criterion: 0.4 Å <= dist <= (r_i + r_j + 0.45 Å)
                if 0.4 <= dist <= (r_i + r_j + 0.45):
                    bonds.append((i, j))
        return bonds

    @classmethod
    def identify_rotatable_bonds(
        cls,
        atoms: List[Atom],
        bonds: List[Tuple[int, int]],
    ) -> List[Tuple[int, int, List[int]]]:
        """
        Identify rotatable bonds (acyclic single bonds connecting non-terminal atoms)
        and determine which atoms rotate around the bond axis.
        Returns: [(atom_idx_a, atom_idx_b, list_of_moving_atoms)]
        """
        # Build adjacency graph
        adj: Dict[int, List[int]] = {i: [] for i in range(len(atoms))}
        for u, v in bonds:
            adj[u].append(v)
            adj[v].append(u)

        rotatable = []

        # Find bridge bonds (removing the bond splits graph into two components)
        for u, v in bonds:
            # Skip if terminal atom (degree 1)
            if len(adj[u]) <= 1 or len(adj[v]) <= 1:
                continue

            # BFS from v without crossing back to u to find moving branch
            visited = set()
            queue = [v]
            visited.add(v)
            is_cycle = False

            while queue:
                curr = queue.pop(0)
                for neighbor in adj[curr]:
                    if curr == v and neighbor == u:
                        continue  # don't traverse back across target bond initially
                    if neighbor == u:
                        # Cyclic bond! Not rotatable
                        is_cycle = True
                        break
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                if is_cycle:
                    break

            if not is_cycle and len(visited) > 0 and len(visited) < len(atoms):
                rotatable.append((u, v, sorted(list(visited))))

        return rotatable

    @classmethod
    def prepare_ligand(cls, filepath: str) -> Molecule:
        """Load a ligand from file, detect rotatable bonds, and annotate metadata.

        Supported formats:
            - ``.sdf`` / ``.mol``  — MDL SDF/MOL V2000 and V3000 (first record used)
            - ``.pdbqt``           — AutoDock PDBQT (charges and types already set)
            - ``.pdb``             — Standard PDB HETATM records (fallback)

        For SDF/MOL files the explicit bond table from the connection table is
        used directly, which gives more accurate rotatable-bond detection than
        the distance-heuristic used for PDB files.
        """
        ext = os.path.splitext(filepath)[1].lower()

        if ext in (".sdf", ".mol"):
            # SDFIO.read_sdf returns a list; take the first record
            mols = SDFIO.read_sdf(filepath, name="Ligand")
            mol = mols[0]
            # Use explicit bond table if present, otherwise infer
            bonds = mol.metadata.get("bonds") or cls.infer_bonds(mol.atoms)
        elif ext == ".pdbqt":
            mol = PDBQTIO.read_pdbqt(filepath, name="Ligand")
            bonds = cls.infer_bonds(mol.atoms)
        else:
            mol = PDBIO.read_pdb(filepath, name="Ligand")
            bonds = cls.infer_bonds(mol.atoms)

        rot_bonds = cls.identify_rotatable_bonds(mol.atoms, bonds)

        mol.metadata["bonds"] = bonds
        mol.metadata["rotatable_bonds"] = rot_bonds
        mol.metadata["num_rotatable_bonds"] = len(rot_bonds)

        # Assign simple electronegativity-based partial charges if zero
        for a in mol.atoms:
            elem = a.element.upper()
            if abs(a.charge) < 1e-4:
                if elem == "O":
                    a.charge = -0.35
                elif elem == "N":
                    a.charge = -0.25
                elif elem == "S":
                    a.charge = -0.15
                elif elem in ("F", "CL"):
                    a.charge = -0.20
                elif elem == "H":
                    a.charge = +0.15

        return mol

    @classmethod
    def prepare_from_sdf_batch(cls, filepath: str) -> List[Molecule]:
        """Load *all* molecules from a multi-molecule SDF file and prepare each.

        Useful for virtual screening: one call processes every record in the
        SDF and returns a list of dock-ready :class:`~moldock.core.molecule.Molecule`
        objects.

        Parameters
        ----------
        filepath:
            Path to a ``.sdf`` file that may contain multiple ``$$$$``-delimited
            records.

        Returns
        -------
        List[Molecule]
            Fully prepared molecules (rotatable bonds, charges, Vina atom types).
        """
        all_mols = SDFIO.read_sdf(filepath)
        prepared: List[Molecule] = []
        for mol in all_mols:
            bonds = mol.metadata.get("bonds") or cls.infer_bonds(mol.atoms)
            rot_bonds = cls.identify_rotatable_bonds(mol.atoms, bonds)
            mol.metadata["bonds"] = bonds
            mol.metadata["rotatable_bonds"] = rot_bonds
            mol.metadata["num_rotatable_bonds"] = len(rot_bonds)
            for a in mol.atoms:
                elem = a.element.upper()
                if abs(a.charge) < 1e-4:
                    if elem == "O":
                        a.charge = -0.35
                    elif elem == "N":
                        a.charge = -0.25
                    elif elem == "S":
                        a.charge = -0.15
                    elif elem in ("F", "CL"):
                        a.charge = -0.20
                    elif elem == "H":
                        a.charge = +0.15
            prepared.append(mol)
        return prepared

    @staticmethod
    def save_ligand_pdbqt(molecule: Molecule, output_path: str) -> str:
        """Save ligand into PDBQT format with TORSDOF record."""
        torsdof = molecule.metadata.get("num_rotatable_bonds", 0)
        PDBQTIO.write_pdbqt(molecule, output_path, torsdof=torsdof, as_rigid=False)
        return output_path

