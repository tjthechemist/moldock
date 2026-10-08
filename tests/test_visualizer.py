"""
Test suite for InteractionVisualizer PNG image generation.
"""

import os
import tempfile
import unittest
import numpy as np

from moldock.core.atom import Atom
from moldock.core.molecule import Molecule
from moldock.metals.detector import MetalCenter
from moldock.metals.parameters import get_metal_parameters
from moldock.analysis.interactions import InteractionContact
from moldock.analysis.visualizer import InteractionVisualizer


class TestInteractionVisualizer(unittest.TestCase):
    def test_render_interaction_diagram(self):
        # Create minimal receptor with zinc atom
        zn = Atom(
            index=1, name="ZN", element="ZN", res_name="ZN", res_seq=100, chain_id="A",
            coord=np.array([0.0, 0.0, 0.0]), atom_type="ZN", charge=2.0
        )
        rec = Molecule(name="Receptor", atoms=[zn])

        # Create query ligand with oxygen donor
        o = Atom(
            index=2, name="O1", element="O", res_name="LIG", res_seq=1, chain_id="B",
            coord=np.array([2.05, 0.0, 0.0]), atom_type="OA"
        )
        lig = Molecule(name="Ligand", atoms=[o])

        contact = InteractionContact(
            interaction_type="metal_coordination",
            receptor_atom_idx=1,
            receptor_res="ZN",
            receptor_res_seq=100,
            receptor_atom_name="ZN",
            query_atom_idx=2,
            query_atom_name="O1",
            query_element="O",
            distance=2.05,
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_png = tmp.name

        try:
            InteractionVisualizer.render_interaction_diagram(
                receptor=rec,
                query=lig,
                contacts=[contact],
                output_png_path=tmp_png,
                title="Test Interaction Diagram",
            )
            self.assertTrue(os.path.exists(tmp_png))
            self.assertGreater(os.path.getsize(tmp_png), 500)
        finally:
            if os.path.exists(tmp_png):
                os.remove(tmp_png)


if __name__ == "__main__":
    unittest.main()
