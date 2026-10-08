"""
Biophysical parameters and forcefield definitions for transition metals and alkaline earth ions.
Includes ideal coordination distances, preferred coordination numbers, and 12-6-4 parameters.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class MetalProperties:
    name: str
    element: str
    oxidation_state: int
    vdw_radius: float          # Å
    ideal_distances: Dict[str, float]  # Donor element -> ideal distance in Å
    preferred_coord_num: int   # Preferred coordination number (e.g. 4 for tetrahedral, 6 for octahedral)
    c4_polarization: float     # C4 parameter for 12-6-4 charge-induced dipole potential (kcal/mol * Å^4)
    epsilon: float             # LJ well depth (kcal/mol)
    sigma: float               # LJ collision diameter (Å)


# Standard library of metal cofactors in metalloproteins
METAL_DATABASE: Dict[str, MetalProperties] = {
    "ZN": MetalProperties(
        name="Zinc",
        element="ZN",
        oxidation_state=2,
        vdw_radius=1.10,
        ideal_distances={"O": 2.05, "N": 2.08, "S": 2.32, "CL": 2.25},
        preferred_coord_num=4,
        c4_polarization=115.0,
        epsilon=0.25,
        sigma=1.95,
    ),
    "MG": MetalProperties(
        name="Magnesium",
        element="MG",
        oxidation_state=2,
        vdw_radius=1.30,
        ideal_distances={"O": 2.08, "N": 2.18, "F": 2.00},
        preferred_coord_num=6,
        c4_polarization=95.0,
        epsilon=0.15,
        sigma=2.00,
    ),
    "CA": MetalProperties(
        name="Calcium",
        element="CA",
        oxidation_state=2,
        vdw_radius=1.50,
        ideal_distances={"O": 2.40, "N": 2.45},
        preferred_coord_num=7,
        c4_polarization=85.0,
        epsilon=0.18,
        sigma=2.45,
    ),
    "MN": MetalProperties(
        name="Manganese",
        element="MN",
        oxidation_state=2,
        vdw_radius=1.30,
        ideal_distances={"O": 2.15, "N": 2.20, "S": 2.40},
        preferred_coord_num=6,
        c4_polarization=105.0,
        epsilon=0.20,
        sigma=2.10,
    ),
    "FE": MetalProperties(
        name="Iron",
        element="FE",
        oxidation_state=2,
        vdw_radius=1.30,
        ideal_distances={"O": 2.10, "N": 2.15, "S": 2.30},
        preferred_coord_num=6,
        c4_polarization=110.0,
        epsilon=0.22,
        sigma=2.05,
    ),
    "CU": MetalProperties(
        name="Copper",
        element="CU",
        oxidation_state=2,
        vdw_radius=1.20,
        ideal_distances={"O": 1.98, "N": 2.01, "S": 2.25},
        preferred_coord_num=4,
        c4_polarization=120.0,
        epsilon=0.28,
        sigma=1.90,
    ),
    "CO": MetalProperties(
        name="Cobalt",
        element="CO",
        oxidation_state=2,
        vdw_radius=1.25,
        ideal_distances={"O": 2.08, "N": 2.12, "S": 2.30},
        preferred_coord_num=6,
        c4_polarization=110.0,
        epsilon=0.24,
        sigma=2.00,
    ),
    "NI": MetalProperties(
        name="Nickel",
        element="NI",
        oxidation_state=2,
        vdw_radius=1.25,
        ideal_distances={"O": 2.05, "N": 2.08, "S": 2.28},
        preferred_coord_num=6,
        c4_polarization=112.0,
        epsilon=0.25,
        sigma=2.00,
    ),
    "NA": MetalProperties(
        name="Sodium",
        element="NA",
        oxidation_state=1,
        vdw_radius=1.40,
        ideal_distances={"O": 2.40, "N": 2.50},
        preferred_coord_num=6,
        c4_polarization=40.0,
        epsilon=0.08,
        sigma=2.40,
    ),
    "K": MetalProperties(
        name="Potassium",
        element="K",
        oxidation_state=1,
        vdw_radius=1.70,
        ideal_distances={"O": 2.75, "N": 2.85},
        preferred_coord_num=8,
        c4_polarization=30.0,
        epsilon=0.05,
        sigma=2.80,
    ),
}


def get_metal_parameters(symbol: str) -> MetalProperties:
    """Retrieve metal parameters for a given element symbol, falling back to Zinc-like defaults."""
    sym = symbol.strip().upper()
    if sym in METAL_DATABASE:
        return METAL_DATABASE[sym]
    # Default fallback for unlisted transition metal
    return MetalProperties(
        name=sym,
        element=sym,
        oxidation_state=2,
        vdw_radius=1.25,
        ideal_distances={"O": 2.10, "N": 2.15, "S": 2.30},
        preferred_coord_num=6,
        c4_polarization=100.0,
        epsilon=0.20,
        sigma=2.10,
    )

