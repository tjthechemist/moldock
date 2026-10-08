"""
Abstract base class for docking scoring functions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from moldock.core.molecule import Molecule


class BaseScoringFunction(ABC):
    """Abstract interface for scoring intermolecular complexes."""

    @abstractmethod
    def score(self, receptor: Molecule, ligand: Molecule) -> float:
        """Calculate total binding affinity score (in kcal/mol)."""
        pass

    @abstractmethod
    def score_breakdown(self, receptor: Molecule, ligand: Molecule) -> Dict[str, Any]:
        """Return breakdown of energy components (e.g. Gauss1, Gauss2, H-bond, Metal)."""
        pass

