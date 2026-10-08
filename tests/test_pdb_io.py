"""
Test suite for PDB and PDBQT parsing and generation.
"""

import os
import tempfile
import numpy as np

from moldock.core.pdb_io import PDBIO, PDBQTIO
from moldock.core.molecule import Molecule


def test_pdb_reading_and_writing():
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_data")
    pdb_path = os.path.join(sample_dir, "receptor_zinc.pdb")

    mol = PDBIO.read_pdb(pdb_path)
    assert len(mol.atoms) == 40
    assert mol.name == "receptor_zinc"

    # Check that Zn was parsed
    metals = mol.get_metal_ions()
    assert len(metals) == 1
    assert metals[0].element.upper() == "ZN"
    assert np.allclose(metals[0].coord, [0.0, -1.5, -1.8])

    # Test writing to temporary PDB
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        PDBIO.write_pdb(mol, tmp_path)
        reloaded = PDBIO.read_pdb(tmp_path)
        assert len(reloaded.atoms) == len(mol.atoms)
        assert np.allclose(reloaded.coords, mol.coords, atol=1e-3)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_pdbqt_writing():
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "examples", "sample_data")
    lig_path = os.path.join(sample_dir, "ligand_inhibitor.pdb")
    mol = PDBIO.read_pdb(lig_path)

    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        PDBQTIO.write_pdbqt(mol, tmp_path, torsdof=2, as_rigid=False)
        reloaded = PDBQTIO.read_pdbqt(tmp_path)
        assert len(reloaded.atoms) == len(mol.atoms)
        assert np.allclose(reloaded.coords, mol.coords, atol=1e-3)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


import unittest


class TestPDBIO(unittest.TestCase):
    def test_pdb_reading_and_writing(self):
        test_pdb_reading_and_writing()

    def test_pdbqt_writing(self):
        test_pdbqt_writing()


if __name__ == "__main__":
    unittest.main()

