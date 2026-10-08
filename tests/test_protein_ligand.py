"""
Test suite for Protein-Ligand docking pipeline with metal co-factors.
"""

import os
import numpy as np

from moldock.preparation.receptor_prep import ReceptorPreparer
from moldock.preparation.ligand_prep import LigandPreparer
from moldock.core.config import DockingConfig, GridBox, MetalConfig
from moldock.engines.protein_ligand import ProteinLigandEngine


def test_protein_ligand_docking():
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_data")
    rec_path = os.path.join(sample_dir, "receptor_zinc.pdb")
    lig_path = os.path.join(sample_dir, "ligand_inhibitor.pdb")

    # Prepare receptor and ligand
    m_cfg = MetalConfig(allowed_metals=["ZN"])
    rec_prep = ReceptorPreparer(metal_config=m_cfg)
    receptor, centers = rec_prep.prepare_from_pdb(rec_path)
    assert len(centers) == 1

    ligand = LigandPreparer.prepare_ligand(lig_path)
    assert len(ligand.atoms) == 10

    # Auto-center grid on zinc cofactor
    metal_coords = np.array([c.metal_atom.coord for c in centers])
    grid = GridBox.from_metal_ions(metal_coords, box_radius=8.0)

    cfg = DockingConfig(
        grid=grid,
        metal_config=m_cfg,
        num_modes=3,
        exhaustiveness=2,
        seed=42,
    )

    engine = ProteinLigandEngine(receptor=receptor, ligand=ligand, config=cfg, metal_centers=centers)
    res = engine.dock()

    assert len(res.poses) >= 1
    top_pose = res.get_top_pose()
    assert top_pose is not None
    assert top_pose.score < 0.0  # Favorable binding score
    assert "vina_score" in top_pose.score_breakdown
    assert "metal_score" in top_pose.score_breakdown


import unittest


class TestProteinLigand(unittest.TestCase):
    def test_docking(self):
        test_protein_ligand_docking()


if __name__ == "__main__":
    unittest.main()

