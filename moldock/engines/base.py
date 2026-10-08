"""
Base classes and result data structures for docking engines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

from moldock.core.molecule import Molecule


@dataclass
class DockingPose:
    """Represents a single docked conformation and its binding energy."""
    pose_id: int
    score: float  # kcal/mol (lower is better)
    coords: np.ndarray  # (N, 3) atom coordinates
    score_breakdown: Dict[str, Any] = field(default_factory=dict)
    rmsd_lb: float = 0.0  # RMSD lower bound
    rmsd_ub: float = 0.0  # RMSD upper bound

    def apply_to_molecule(self, template_molecule: Molecule) -> Molecule:
        """Create a new Molecule object with this pose's coordinates."""
        mol = template_molecule.copy()
        mol.coords = np.copy(self.coords)
        mol.metadata["docking_score"] = self.score
        mol.metadata["score_breakdown"] = self.score_breakdown
        return mol


@dataclass
class DockingResult:
    """Container for the output of a docking run."""
    docking_type: str
    receptor: Molecule
    query: Molecule  # Ligand or Partner protein
    poses: List[DockingPose] = field(default_factory=list)
    top_score: float = 0.0
    execution_time_seconds: float = 0.0
    config_summary: Dict[str, Any] = field(default_factory=dict)

    def get_top_pose(self) -> Optional[DockingPose]:
        return self.poses[0] if self.poses else None

    def export_poses_pdb(self, filepath: str) -> None:
        """Export all docked poses to a multi-model PDB file."""
        from moldock.core.pdb_io import PDBIO
        for pose in self.poses:
            mol = pose.apply_to_molecule(self.query)
            PDBIO.write_pdb(mol, filepath, model_id=pose.pose_id)

    def export_poses_pdbqt(self, filepath: str) -> None:
        """Export all docked poses to an AutoDock Vina multi-model PDBQT file."""
        from moldock.core.pdb_io import PDBQTIO
        torsdof = self.query.metadata.get("num_rotatable_bonds", 0)
        for pose in self.poses:
            mol = pose.apply_to_molecule(self.query)
            PDBQTIO.write_pdbqt(mol, filepath, torsdof=torsdof, model_id=pose.pose_id, as_rigid=False)


class BaseDockingEngine(ABC):
    """Abstract interface for docking engines."""

    @abstractmethod
    def dock(self) -> DockingResult:
        """Execute docking simulation and return poses."""
        pass
