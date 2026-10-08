"""
Configuration data models and parameters for molecular docking.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class DockingType(Enum):
    PROTEIN_LIGAND = "protein_ligand"
    PROTEIN_PROTEIN = "protein_protein"


@dataclass
class GridBox:
    """
    Search grid box for molecular docking.
    Coordinates and sizes are in Angstroms (Å).
    """
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0
    size_x: float = 20.0
    size_y: float = 20.0
    size_z: float = 20.0
    spacing: float = 0.375

    @property
    def center(self) -> np.ndarray:
        return np.array([self.center_x, self.center_y, self.center_z], dtype=np.float64)

    @property
    def size(self) -> np.ndarray:
        return np.array([self.size_x, self.size_y, self.size_z], dtype=np.float64)

    def contains_point(self, point: np.ndarray, tolerance: float = 0.0) -> bool:
        """Check if a 3D coordinate point lies within the grid box."""
        half_size = self.size / 2.0 + tolerance
        diff = np.abs(np.asarray(point, dtype=np.float64) - self.center)
        return bool(np.all(diff <= half_size))

    @classmethod
    def from_coordinates(
        cls,
        coords: np.ndarray,
        padding: float = 6.0,
        min_size: float = 15.0,
        spacing: float = 0.375,
    ) -> "GridBox":
        """
        Compute minimum bounding grid box centered on the given coordinates.
        Useful for binding pockets, co-crystallized ligands, or active site residues.
        """
        coords = np.asarray(coords, dtype=np.float64)
        if len(coords) == 0:
            return cls(spacing=spacing)

        min_bound = np.min(coords, axis=0)
        max_bound = np.max(coords, axis=0)
        center = (min_bound + max_bound) / 2.0
        size = np.maximum(max_bound - min_bound + 2 * padding, min_size)

        return cls(
            center_x=float(center[0]),
            center_y=float(center[1]),
            center_z=float(center[2]),
            size_x=float(size[0]),
            size_y=float(size[1]),
            size_z=float(size[2]),
            spacing=spacing,
        )

    @classmethod
    def from_metal_ions(
        cls,
        metal_coords: np.ndarray,
        box_radius: float = 12.0,
        spacing: float = 0.375,
    ) -> "GridBox":
        """
        Create a grid box centered around one or multiple metal ion cofactors.
        """
        metal_coords = np.asarray(metal_coords, dtype=np.float64)
        center = np.mean(metal_coords, axis=0)
        span = np.max(metal_coords, axis=0) - np.min(metal_coords, axis=0) if len(metal_coords) > 1 else np.zeros(3)
        size = np.maximum(span + 2 * box_radius, 20.0)

        return cls(
            center_x=float(center[0]),
            center_y=float(center[1]),
            center_z=float(center[2]),
            size_x=float(size[0]),
            size_y=float(size[1]),
            size_z=float(size[2]),
            spacing=spacing,
        )


@dataclass
class MetalConfig:
    """
    Configuration for detecting, preserving, and scoring metal co-factors.
    Supported standard metals: Zn, Mg, Ca, Mn, Fe, Cu, Co, Ni, Na, K.
    """
    allowed_metals: List[str] = field(
        default_factory=lambda: ["ZN", "MG", "CA", "MN", "FE", "CU", "CO", "NI", "NA", "K"]
    )
    retain_in_receptor: bool = True
    coordination_cutoff: float = 2.85  # Max coordination distance (Å)
    ideal_coordination_dist: float = 2.15  # Ideal metal-donor distance (Å)
    coordination_weight: float = 1.25  # Weight in total binding affinity score
    use_12_6_4_potential: bool = True  # Enhanced divalent ion polarization potential
    custom_charges: Dict[str, float] = field(
        default_factory=lambda: {
            "ZN": 2.0,
            "MG": 2.0,
            "CA": 2.0,
            "MN": 2.0,
            "FE": 2.0,
            "CU": 2.0,
            "CO": 2.0,
            "NI": 2.0,
            "NA": 1.0,
            "K": 1.0,
        }
    )


@dataclass
class ScoringWeights:
    """
    Weights for AutoDock Vina empirical scoring terms (Trott & Olson 2010),
    extended with metal-coordination and electrostatic interactions.
    """
    weight_gauss1: float = -0.035579
    weight_gauss2: float = -0.005156
    weight_repulsion: float = 0.840245
    weight_hydrophobic: float = -0.035069
    weight_hbond: float = -0.587439
    weight_rot: float = 0.05846
    # Extended terms
    weight_metal: float = -1.250000  # Favorable metal-donor coordination
    weight_electrostatic: float = 0.050000  # Coulombic term with distance-dependent dielectric


@dataclass
class DockingConfig:
    """
    Global configuration for a docking run.
    """
    docking_type: DockingType = DockingType.PROTEIN_LIGAND
    grid: GridBox = field(default_factory=GridBox)
    metal_config: MetalConfig = field(default_factory=MetalConfig)
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    num_modes: int = 9
    exhaustiveness: int = 8
    energy_range: float = 3.0  # Maximum energy difference from top pose (kcal/mol)
    seed: Optional[int] = 42
    cpu_threads: int = 4
    rigid_body_rotations: int = 360  # For protein-protein rotational sampling
    rigid_body_translations: int = 50  # For protein-protein translational sampling
