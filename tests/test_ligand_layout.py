"""
Tests for the 2D ligand depiction layout.
"""

import math
import unittest

import numpy as np

from moldock.core.atom import Atom
from moldock.analysis.bond_orders import infer_bonds_from_geometry
from moldock.analysis.ligand_layout import (
    layout_ligand_depiction,
    relax_layout,
)
from moldock.analysis.visualizer import (
    LEGIBLE_BOND_PIXELS,
    MAX_ATOM_RADIUS,
    MIN_ATOM_RADIUS,
    atom_radius_for_bond,
)


def make_atom(index, name, element, xyz):
    return Atom(
        index=index, name=name, element=element, res_name="LIG", res_seq=1,
        chain_id="L", coord=np.array(xyz, dtype=np.float64), is_hetatm=True,
    )


def deep_ligand():
    """Molecule whose raw XY projection stacks atoms on top of each other."""
    coords = [
        ("C1", "C", (0.000, 0.500, 0.000)),
        ("C2", "C", (1.200, 1.200, 0.000)),
        ("C3", "C", (1.200, 2.600, 0.000)),
        ("C4", "C", (0.000, 3.300, 0.000)),
        ("C5", "C", (-1.200, 2.600, 0.000)),
        ("C6", "C", (-1.200, 1.200, 0.000)),
        ("S1", "S", (0.000, -1.200, 0.000)),
        ("O1", "O", (1.250, -1.600, 0.550)),
        ("O2", "O", (-1.250, -1.600, 0.550)),
        ("N1", "N", (0.000, -1.500, -1.500)),
    ]
    return [make_atom(i + 1, name, elem, xyz)
            for i, (name, elem, xyz) in enumerate(coords)]


class TestOverlapRelaxation(unittest.TestCase):
    def test_coincident_atoms_are_separated(self):
        # Two atoms projected onto exactly the same point
        pts = np.array([[0.0, 0.0], [0.0, 0.0], [1.4, 0.0]])
        relaxed = relax_layout(pts, [(0, 1), (1, 2)])

        self.assertTrue(np.all(np.isfinite(relaxed)))
        d01 = float(np.linalg.norm(relaxed[0] - relaxed[1]))
        self.assertGreater(d01, 0.2)

    def test_single_atom_is_unchanged(self):
        pts = np.array([[3.0, 4.0]])
        relaxed = relax_layout(pts, [])
        np.testing.assert_allclose(relaxed, pts)


class TestLayoutLigandDepiction(unittest.TestCase):
    def test_one_position_per_atom(self):
        atoms = deep_ligand()
        bonds = infer_bonds_from_geometry(atoms)
        positions, bond_px = layout_ligand_depiction(
            atoms, bonds, box_width=400.0, box_height=400.0,
            center_x=200.0, center_y=200.0,
        )
        self.assertEqual(len(positions), len(atoms))
        self.assertTrue(all(math.isfinite(x) and math.isfinite(y) for x, y in positions))
        self.assertGreater(bond_px, 0.0)

    def test_no_atom_overlaps_another(self):
        """Every atom must be individually visible at the chosen glyph size."""
        atoms = deep_ligand()
        bonds = infer_bonds_from_geometry(atoms)
        positions, bond_px = layout_ligand_depiction(
            atoms, bonds, box_width=400.0, box_height=400.0,
            center_x=200.0, center_y=200.0,
        )

        radius = atom_radius_for_bond(bond_px)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                gap = math.dist(positions[i], positions[j])
                self.assertGreater(
                    gap, 2.0 * radius,
                    f"atoms {atoms[i].name} and {atoms[j].name} overlap "
                    f"({gap:.1f} px apart, glyph diameter {2 * radius:.1f} px)",
                )

    def test_deep_z_atoms_are_not_hidden(self):
        """N1 sits far below S1 in Z; the depiction must still separate them."""
        atoms = deep_ligand()
        bonds = infer_bonds_from_geometry(atoms)
        positions, _ = layout_ligand_depiction(
            atoms, bonds, box_width=400.0, box_height=400.0,
            center_x=200.0, center_y=200.0,
        )
        names = {a.name: i for i, a in enumerate(atoms)}
        gap = math.dist(positions[names["S1"]], positions[names["N1"]])
        self.assertGreater(gap, 20.0)

    def test_depiction_is_centred(self):
        atoms = deep_ligand()
        bonds = infer_bonds_from_geometry(atoms)
        positions, _ = layout_ligand_depiction(
            atoms, bonds, box_width=400.0, box_height=400.0,
            center_x=500.0, center_y=300.0,
        )
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        self.assertAlmostEqual((min(xs) + max(xs)) / 2.0, 500.0, delta=1.0)
        self.assertAlmostEqual((min(ys) + max(ys)) / 2.0, 300.0, delta=1.0)

    def test_stays_inside_the_box(self):
        atoms = deep_ligand()
        bonds = infer_bonds_from_geometry(atoms)
        box = 400.0
        positions, _ = layout_ligand_depiction(
            atoms, bonds, box_width=box, box_height=box,
            center_x=200.0, center_y=200.0,
        )
        for x, y in positions:
            self.assertGreaterEqual(x, 200.0 - box / 2.0 - 1.0)
            self.assertLessEqual(x, 200.0 + box / 2.0 + 1.0)
            self.assertGreaterEqual(y, 200.0 - box / 2.0 - 1.0)
            self.assertLessEqual(y, 200.0 + box / 2.0 + 1.0)

    def test_single_atom_ligand(self):
        atoms = [make_atom(1, "C1", "C", (0.0, 0.0, 0.0))]
        positions, bond_px = layout_ligand_depiction(
            atoms, [], box_width=200.0, box_height=200.0,
            center_x=100.0, center_y=100.0,
        )
        self.assertEqual(len(positions), 1)
        self.assertTrue(math.isfinite(positions[0][0]))
        self.assertGreater(bond_px, 0.0)

    def test_empty_ligand(self):
        positions, bond_px = layout_ligand_depiction(
            [], [], box_width=200.0, box_height=200.0,
            center_x=100.0, center_y=100.0,
        )
        self.assertEqual(positions, [])


class TestAtomSizing(unittest.TestCase):
    def test_radius_tracks_bond_length(self):
        self.assertLess(atom_radius_for_bond(10.0), atom_radius_for_bond(40.0))

    def test_radius_is_clamped(self):
        self.assertGreaterEqual(atom_radius_for_bond(0.1), MIN_ATOM_RADIUS)
        self.assertLessEqual(atom_radius_for_bond(10000.0), MAX_ATOM_RADIUS)

    def test_radius_never_reaches_half_the_minimum_separation(self):
        """At any legible bond length the glyph fits inside the closest gap.

        The layout never places two atoms closer than
        ``MIN_SEPARATION_FACTOR`` of a bond length, so a glyph diameter that
        stays below that gap means no atom can ever hide another.
        """
        from moldock.analysis.ligand_layout import MIN_SEPARATION_FACTOR

        for factor in (1.0, 1.5, 2.5, 6.25, 15.0, 50.0):
            bond_px = LEGIBLE_BOND_PIXELS * factor
            radius = atom_radius_for_bond(bond_px)
            allowed_gap = bond_px * MIN_SEPARATION_FACTOR
            self.assertLessEqual(
                2.0 * radius, allowed_gap + 1e-9,
                f"glyph diameter {2 * radius:.2f} px exceeds the "
                f"{allowed_gap:.2f} px minimum gap at bond length {bond_px:.1f} px",
            )


if __name__ == "__main__":
    unittest.main()
