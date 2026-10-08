"""
MolDock: Python Framework for Molecular Docking with AutoDock Vina Scoring & Metal Co-factor Coordination.

Provides:
- Protein-Ligand Docking (Monte Carlo simulated annealing, torsion flexibility)
- Protein-Protein Docking from PDB files (6D rigid-body search, interface refinement)
- AutoDock Vina Empirical Scoring (Gauss1, Gauss2, Repulsion, Hydrophobic, Hydrogen bonding)
- Explicit Multi-Metal Co-factor Handling (Zn, Mg, Ca, Mn, Fe, Cu, Co, Ni)
- Coordination Well & 12-6-4 Polarization Potentials
"""

from moldock.core.atom import Atom, assign_vina_atom_type
from moldock.core.molecule import Molecule
from moldock.core.config import (
    GridBox,
    MetalConfig,
    DockingConfig,
    DockingType,
    ScoringWeights,
)
from moldock.core.pdb_io import PDBIO, PDBQTIO
from moldock.metals.parameters import MetalProperties, get_metal_parameters
from moldock.metals.detector import MetalDetector, MetalCenter
from moldock.metals.scoring import MetalCoordinationScorer, MetalScoreBreakdown
from moldock.scoring.vina_scoring import VinaScoringFunction, VinaScoreComponents
from moldock.scoring.composite_scoring import CompositeScoringFunction
from moldock.preparation.receptor_prep import ReceptorPreparer
from moldock.preparation.ligand_prep import LigandPreparer
from moldock.preparation.partner_prep import PartnerPreparer
from moldock.engines.protein_ligand import ProteinLigandEngine
from moldock.engines.protein_protein import ProteinProteinEngine
from moldock.engines.base import DockingPose, DockingResult
from moldock.analysis.interactions import InteractionProfiler
from moldock.analysis.rmsd import calculate_rmsd, kabsch_rmsd
from moldock.analysis.reporter import DockingReporter

__version__ = "0.1.0"

__all__ = [
    "Atom",
    "assign_vina_atom_type",
    "Molecule",
    "GridBox",
    "MetalConfig",
    "DockingConfig",
    "DockingType",
    "ScoringWeights",
    "PDBIO",
    "PDBQTIO",
    "MetalProperties",
    "get_metal_parameters",
    "MetalDetector",
    "MetalCenter",
    "MetalCoordinationScorer",
    "MetalScoreBreakdown",
    "VinaScoringFunction",
    "VinaScoreComponents",
    "CompositeScoringFunction",
    "ReceptorPreparer",
    "LigandPreparer",
    "PartnerPreparer",
    "ProteinLigandEngine",
    "ProteinProteinEngine",
    "DockingPose",
    "DockingResult",
    "InteractionProfiler",
    "calculate_rmsd",
    "kabsch_rmsd",
    "DockingReporter",
]
