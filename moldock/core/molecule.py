"""
Molecule representation for macromolecules, co-factors, and small ligands.
"""

from typing import List, Optional, Set, Dict, Tuple
import numpy as np
from moldock.core.atom import Atom


class Molecule:
    """
    Container representing a chemical or biological structure
    (e.g., protein receptor, ligand, or docked complex).
    """

    def __init__(self, name: str = "Molecule", atoms: Optional[List[Atom]] = None):
        self.name = name
        self.atoms: List[Atom] = atoms if atoms is not None else []
        self.metadata: Dict[str, any] = {}

    def __len__(self) -> int:
        return len(self.atoms)

    def add_atom(self, atom: Atom) -> None:
        self.atoms.append(atom)

    @property
    def coords(self) -> np.ndarray:
        """Return (N, 3) coordinate matrix of all atoms."""
        if not self.atoms:
            return np.empty((0, 3), dtype=np.float64)
        return np.array([a.coord for a in self.atoms], dtype=np.float64)

    @coords.setter
    def coords(self, new_coords: np.ndarray) -> None:
        """Update coordinates of all atoms from (N, 3) matrix."""
        new_coords = np.asarray(new_coords, dtype=np.float64)
        if len(new_coords) != len(self.atoms):
            raise ValueError(f"Coordinate count {len(new_coords)} does not match atom count {len(self.atoms)}")
        for atom, coord in zip(self.atoms, new_coords):
            atom.coord = np.copy(coord)

    @property
    def center_of_mass(self) -> np.ndarray:
        """Compute geometric center (or unweighted center of mass)."""
        if not self.atoms:
            return np.zeros(3, dtype=np.float64)
        return np.mean(self.coords, axis=0)

    def translate(self, vector: np.ndarray) -> "Molecule":
        """Translate all atoms by a 3D vector in-place and return self."""
        vec = np.asarray(vector, dtype=np.float64)
        for atom in self.atoms:
            atom.coord += vec
        return self

    def rotate(self, rotation_matrix: np.ndarray, origin: Optional[np.ndarray] = None) -> "Molecule":
        """Rotate all atoms around a reference origin (default: center of mass)."""
        if origin is None:
            origin = self.center_of_mass
        origin = np.asarray(origin, dtype=np.float64)
        rot = np.asarray(rotation_matrix, dtype=np.float64)

        for atom in self.atoms:
            rel = atom.coord - origin
            atom.coord = origin + np.dot(rot, rel)
        return self

    def center_at_origin(self) -> Tuple["Molecule", np.ndarray]:
        """Translate center of mass to (0, 0, 0). Returns (self, old_center)."""
        center = self.center_of_mass
        self.translate(-center)
        return self, center

    def get_metal_ions(self) -> List[Atom]:
        """Extract all metal co-factor atoms."""
        return [a for a in self.atoms if a.is_metal]

    def get_heavy_atoms(self) -> List[Atom]:
        """Extract non-hydrogen atoms."""
        return [a for a in self.atoms if a.element.upper() != "H"]

    def get_chains(self) -> List[str]:
        """Get unique chain IDs."""
        seen = set()
        chains = []
        for a in self.atoms:
            if a.chain_id not in seen:
                seen.add(a.chain_id)
                chains.append(a.chain_id)
        return chains

    def filter_by_chains(self, chain_ids: Set[str]) -> "Molecule":
        """Return a new Molecule containing only atoms in specified chains."""
        filtered_atoms = [a.copy() for a in self.atoms if a.chain_id in chain_ids]
        mol = Molecule(name=f"{self.name}_chains_{'_'.join(sorted(chain_ids))}", atoms=filtered_atoms)
        return mol

    def copy(self) -> "Molecule":
        """Deep copy of Molecule."""
        new_atoms = [a.copy() for a in self.atoms]
        new_mol = Molecule(name=self.name, atoms=new_atoms)
        new_mol.metadata = dict(self.metadata)
        return new_mol
