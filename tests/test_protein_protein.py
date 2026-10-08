"""
Test suite for Protein-Protein docking pipeline from PDB files.
"""

import os
import numpy as np

from moldock.preparation.receptor_prep import ReceptorPreparer
from moldock.preparation.partner_prep import PartnerPreparer
from moldock.core.config import DockingConfig, GridBox, MetalConfig
from moldock.engines.protein_protein import ProteinProteinEngine


def test_protein_protein_docking():
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_data")
    rec_path = os.path.join(sample_dir, "receptor_zinc.pdb")
    partner_path = os.path.join(sample_dir, "protein_partner.pdb")

    m_cfg = MetalConfig(allowed_metals=["ZN"])
    rec_prep = ReceptorPreparer(metal_config=m_cfg)
    receptor, centers = rec_prep.prepare_from_pdb(rec_path)

    partner = PartnerPreparer.prepare_partner_from_pdb(partner_path)
    assert len(partner.atoms) == 23

    # Define interface search region centered near receptor
    r_center = receptor.center_of_mass
    grid = GridBox(
        center_x=r_center[0], center_y=r_center[1], center_z=r_center[2],
        size_x=25.0, size_y=25.0, size_z=25.0
    )

    cfg = DockingConfig(
        grid=grid,
        metal_config=m_cfg,
        num_modes=3,
        rigid_body_rotations=30,
        rigid_body_translations=15,
        exhaustiveness=2,
        seed=42,
    )

    engine = ProteinProteinEngine(receptor=receptor, partner=partner, config=cfg, metal_centers=centers)
    res = engine.dock()

    assert len(res.poses) >= 1
    top_pose = res.get_top_pose()
    assert top_pose is not None
    assert "vina_score" in top_pose.score_breakdown


import unittest


class TestProteinProtein(unittest.TestCase):
    def test_docking(self):
        test_protein_protein_docking()


if __name__ == "__main__":
    unittest.main()

