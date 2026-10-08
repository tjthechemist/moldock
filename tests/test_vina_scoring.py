"""
Test suite for AutoDock Vina scoring and Composite metal scoring.
"""

import os
import numpy as np
from moldock.core.atom import Atom
from moldock.core.molecule import Molecule
from moldock.core.config import ScoringWeights
from moldock.scoring.vina_scoring import VinaScoringFunction
from moldock.scoring.composite_scoring import CompositeScoringFunction
from moldock.metals.detector import MetalCenter
from moldock.metals.parameters import get_metal_parameters


def test_vina_scoring_components():
    scorer = VinaScoringFunction()

    # 1. Hydrogen bond pair: polar hydrogen HD and oxygen acceptor OA at 1.85 Å
    donor_h = Atom(
        index=1, name="HD1", element="H", res_name="ALA", res_seq=1, chain_id="A",
        coord=np.array([0.0, 0.0, 0.0]), atom_type="HD"
    )
    acc_o = Atom(
        index=2, name="O", element="O", res_name="LIG", res_seq=1, chain_id="B",
        coord=np.array([1.85, 0.0, 0.0]), atom_type="OA"
    )
    comp_hb = scorer.evaluate_components([donor_h], [acc_o])
    assert comp_hb.hbond > 0.0
    assert comp_hb.gauss1 > 0.0

    # 2. Hydrophobic pair: two aliphatic carbons at 3.6 Å
    c1 = Atom(
        index=3, name="C1", element="C", res_name="LEU", res_seq=2, chain_id="A",
        coord=np.array([0.0, 0.0, 0.0]), atom_type="C"
    )
    c2 = Atom(
        index=4, name="C2", element="C", res_name="LIG", res_seq=1, chain_id="B",
        coord=np.array([3.6, 0.0, 0.0]), atom_type="C"
    )
    comp_hyd = scorer.evaluate_components([c1], [c2])
    assert comp_hyd.hydrophobic > 0.0
    assert comp_hyd.intermolecular_energy < 0.0  # attractive overall



def test_composite_metal_scoring():
    scorer = CompositeScoringFunction()

    # Zinc metal atom
    zn_atom = Atom(
        index=1, name="ZN", element="ZN", res_name="ZN", res_seq=1, chain_id="A",
        coord=np.array([0.0, 0.0, 0.0]), atom_type="ZN", charge=2.0
    )
    center = MetalCenter(metal_atom=zn_atom, properties=get_metal_parameters("ZN"))
    scorer.set_metal_centers([center])

    # Ligand oxygen donor placed at ideal coordination distance (2.05 Å)
    lig_o = Atom(
        index=2, name="O1", element="O", res_name="LIG", res_seq=1, chain_id="B",
        coord=np.array([2.05, 0.0, 0.0]), atom_type="OA", charge=-0.4
    )
    mol_rec = Molecule(atoms=[zn_atom])
    mol_lig = Molecule(atoms=[lig_o])

    bd = scorer.score_breakdown(mol_rec, mol_lig)
    assert bd["metal_score"] < -0.5  # Strong favorable metal coordination
    assert bd["total_score"] < 0.0


import unittest


class TestVinaScoring(unittest.TestCase):
    def test_components(self):
        test_vina_scoring_components()

    def test_metal_score(self):
        test_composite_metal_scoring()


if __name__ == "__main__":
    unittest.main()

