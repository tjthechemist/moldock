"""
Atom representation, elements, Van der Waals radii, and AutoDock Vina atom typing.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# Van der Waals radii (in Ångströms) used in AutoDock Vina and standard biophysics
VDW_RADII = {
    "C": 1.70,   # Aliphatic carbon
    "A": 1.70,   # Aromatic carbon
    "N": 1.55,   # Nitrogen (non-acceptor)
    "NA": 1.55,  # Nitrogen (acceptor)
    "O": 1.52,   # Oxygen
    "OA": 1.52,  # Oxygen (acceptor)
    "S": 1.80,   # Sulfur
    "SA": 1.80,  # Sulfur (acceptor)
    "P": 1.80,   # Phosphorus
    "H": 1.20,   # Non-polar hydrogen
    "HD": 1.00,  # Polar hydrogen (donor)
    "F": 1.47,   # Fluorine
    "CL": 1.75,  # Chlorine
    "BR": 1.85,  # Bromine
    "I": 1.98,   # Iodine
    # Metal ions
    "ZN": 1.10,  # Zinc ion
    "MG": 1.30,  # Magnesium ion
    "CA": 1.50,  # Calcium ion
    "MN": 1.30,  # Manganese ion
    "FE": 1.30,  # Iron ion
    "CU": 1.20,  # Copper ion
    "CO": 1.25,  # Cobalt ion
    "NI": 1.25,  # Nickel ion
    "NA+": 1.40, # Sodium ion
    "K": 1.70,   # Potassium ion
}

# Recognized transition and alkaline-earth metal elements
KNOWN_METALS = {
    "ZN", "MG", "CA", "MN", "FE", "CU", "CO", "NI", "NA", "K",
    "CD", "PT", "AU", "HG", "MO", "W", "V", "CR"
}

# Hydrogen bond donor/acceptor definition
HBOND_ACCEPTORS = {"OA", "NA", "SA", "O", "N", "F"}
HBOND_DONORS = {"HD"}
HYDROPHOBIC_TYPES = {"C", "A", "F", "CL", "BR", "I"}


@dataclass
class Atom:
    """
    Representation of an atom in a macromolecule or small ligand.
    """
    index: int
    name: str
    element: str
    res_name: str
    res_seq: int
    chain_id: str
    coord: np.ndarray  # 3D coordinates [x, y, z]
    charge: float = 0.0
    atom_type: str = "C"
    b_factor: float = 0.0
    occupancy: float = 1.0
    is_hetatm: bool = False

    @property
    def is_metal(self) -> bool:
        elem = self.element.strip().upper()
        res = self.res_name.strip().upper()
        return elem in KNOWN_METALS or res in KNOWN_METALS

    @property
    def is_donor(self) -> bool:
        # Polar hydrogen or nitrogen with attached polar hydrogen
        return self.atom_type in HBOND_DONORS or (
            self.element == "N" and self.atom_type != "NA"
        )

    @property
    def is_acceptor(self) -> bool:
        return self.atom_type in HBOND_ACCEPTORS or (
            self.element in ("O", "N", "F") and not self.is_donor
        )

    @property
    def is_hydrophobic(self) -> bool:
        return self.atom_type in HYDROPHOBIC_TYPES

    @property
    def vdw_radius(self) -> float:
        type_key = self.atom_type.upper()
        elem_key = self.element.upper()
        return VDW_RADII.get(type_key, VDW_RADII.get(elem_key, 1.70))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Atom):
            return False
        return (
            self.index == other.index
            and self.name == other.name
            and self.res_name == other.res_name
            and self.res_seq == other.res_seq
            and self.chain_id == other.chain_id
        )

    def __hash__(self) -> int:
        return hash((self.index, self.name, self.res_name, self.res_seq, self.chain_id))

    def copy(self) -> "Atom":

        return Atom(
            index=self.index,
            name=self.name,
            element=self.element,
            res_name=self.res_name,
            res_seq=self.res_seq,
            chain_id=self.chain_id,
            coord=np.copy(self.coord),
            charge=self.charge,
            atom_type=self.atom_type,
            b_factor=self.b_factor,
            occupancy=self.occupancy,
            is_hetatm=self.is_hetatm,
        )


def assign_vina_atom_type(name: str, element: str, res_name: str, charge: float = 0.0) -> str:
    """
    Deduce standard AutoDock/Vina atom type from element and chemical context.
    """
    elem = element.strip().upper()
    name = name.strip().upper()
    res = res_name.strip().upper()

    # Metal ion check
    if elem in KNOWN_METALS or res in KNOWN_METALS:
        return elem if elem in KNOWN_METALS else res

    if elem == "C":
        # Aromatic vs Aliphatic
        if res in ("PHE", "TYR", "TRP", "HIS"):
            if name in ("CG", "CD1", "CD2", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2"):
                return "A"
        return "C"

    if elem == "N":
        # Nitrogen acceptors vs neutral/donors
        if res == "HIS" and name in ("ND1", "NE2"):
            return "NA"
        if name.startswith("N"):
            return "N"
        return "NA"

    if elem == "O":
        return "OA"

    if elem == "S":
        return "SA"

    if elem == "H":
        # Polar hydrogen (HD) vs aliphatic/aromatic hydrogen
        if name in ("H", "H1", "H2", "H3", "HD1", "HE1", "HE2", "HH", "HG", "HZ"):
            return "HD"
        return "H"

    if elem in ("P", "F", "CL", "BR", "I"):
        return elem

    return elem

