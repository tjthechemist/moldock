"""
Test suite for MetalDetector, MetalCenter, and coordination parameters.
"""

import os
from moldock.core.pdb_io import PDBIO
from moldock.metals.detector import MetalDetector
from moldock.metals.parameters import get_metal_parameters


def test_metal_parameters():
    zn = get_metal_parameters("ZN")
    assert zn.name == "Zinc"
    assert zn.oxidation_state == 2
    assert "O" in zn.ideal_distances
    assert "N" in zn.ideal_distances
    assert zn.preferred_coord_num == 4

    mg = get_metal_parameters("MG")
    assert mg.name == "Magnesium"
    assert mg.preferred_coord_num == 6


def test_metal_detector_coordination():
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_data")
    pdb_path = os.path.join(sample_dir, "receptor_zinc.pdb")
    mol = PDBIO.read_pdb(pdb_path)

    detector = MetalDetector(coordination_cutoff=3.0)
    centers = detector.detect_metal_centers(mol)

    assert len(centers) == 1
    center = centers[0]
    assert center.properties.element == "ZN"
    assert center.coordination_number >= 1

    # Check that coordinating residues are identified
    coordinating_res_names = [cr.res_name for cr in center.coordinating_atoms]
    assert "HIS" in coordinating_res_names
    assert center.available_valency >= 1


import unittest


class TestMetalDetector(unittest.TestCase):
    def test_parameters(self):
        test_metal_parameters()

    def test_detector(self):
        test_metal_detector_coordination()


if __name__ == "__main__":
    unittest.main()

