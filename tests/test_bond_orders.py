"""
Tests for bond-order, ring and aromaticity perception.
"""

import unittest

import numpy as np

from moldock.core.atom import Atom
from moldock.analysis.bond_orders import (
    AROMATIC_ORDER,
    DOUBLE,
    SINGLE,
    bond_key,
    detect_aromatic_bonds,
    find_rings,
    infer_bonds_from_geometry,
    build_adjacency,
    perceive_bond_orders,
)


def make_atom(index, name, element, xyz):
    return Atom(
        index=index, name=name, element=element, res_name="LIG", res_seq=1,
        chain_id="L", coord=np.array(xyz, dtype=np.float64), is_hetatm=True,
    )


def benzothiazole_like():
    """The 10-atom inhibitor used by the protein-ligand example."""
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


class TestBondInference(unittest.TestCase):
    def test_infers_expected_graph(self):
        atoms = benzothiazole_like()
        bonds = infer_bonds_from_geometry(atoms)
        self.assertEqual(len(bonds), 10)

        by_name = {(atoms[i].name, atoms[j].name) for i, j in bonds}
        self.assertIn(("C1", "C2"), by_name)
        self.assertIn(("C1", "C6"), by_name)
        self.assertIn(("C1", "S1"), by_name)
        self.assertIn(("S1", "O1"), by_name)
        self.assertIn(("S1", "O2"), by_name)
        self.assertIn(("S1", "N1"), by_name)
        # Non-bonded atoms must not be linked
        self.assertNotIn(("O1", "O2"), by_name)
        self.assertNotIn(("C1", "C4"), by_name)

    def test_single_atom_has_no_bonds(self):
        atoms = [make_atom(1, "C1", "C", (0.0, 0.0, 0.0))]
        self.assertEqual(infer_bonds_from_geometry(atoms), [])


class TestRingPerception(unittest.TestCase):
    def test_finds_six_membered_ring(self):
        atoms = benzothiazole_like()
        bonds = infer_bonds_from_geometry(atoms)
        adj = build_adjacency(len(atoms), bonds)
        rings = find_rings(adj, bonds)

        six_rings = [r for r in rings if len(r) == 6]
        self.assertEqual(len(six_rings), 1)
        ring_names = {atoms[i].name for i in six_rings[0]}
        self.assertEqual(ring_names, {"C1", "C2", "C3", "C4", "C5", "C6"})

    def test_aromatic_ring_detected(self):
        atoms = benzothiazole_like()
        bonds = infer_bonds_from_geometry(atoms)
        adj = build_adjacency(len(atoms), bonds)
        rings = find_rings(adj, bonds)

        aromatic = detect_aromatic_bonds(atoms, adj, rings)
        names = {frozenset((atoms[i].name, atoms[j].name)) for i, j in aromatic}
        self.assertIn(frozenset(("C1", "C2")), names)
        self.assertIn(frozenset(("C4", "C5")), names)
        # The C-S exocyclic bond must not be flagged aromatic
        self.assertNotIn(frozenset(("C1", "S1")), names)

    def test_cyclohexane_ring_is_not_aromatic(self):
        """Long single bonds must not be mistaken for an aromatic ring."""
        names = ["C1", "C2", "C3", "C4", "C5", "C6"]
        atoms = []
        for i, name in enumerate(names):
            angle = 2.0 * np.pi * i / 6.0
            radius = 1.52 / (2.0 * np.sin(np.pi / 6.0))
            atoms.append(make_atom(i + 1, name, "C",
                                   (radius * np.cos(angle), radius * np.sin(angle), 0.0)))

        bonds = infer_bonds_from_geometry(atoms)
        adj = build_adjacency(len(atoms), bonds)
        rings = find_rings(adj, bonds)
        self.assertTrue(rings)
        self.assertEqual(detect_aromatic_bonds(atoms, adj, rings), set())


class TestBondOrders(unittest.TestCase):
    def test_perceived_orders(self):
        atoms = benzothiazole_like()
        bonds = infer_bonds_from_geometry(atoms)
        orders = perceive_bond_orders(atoms, bonds)

        name = {i: a.name for i, a in enumerate(atoms)}

        def order(a, b):
            for i, j in orders:
                if {name[i], name[j]} == {a, b}:
                    return orders[(i, j)]
            self.fail(f"bond {a}-{b} missing")

        self.assertEqual(order("C1", "C2"), AROMATIC_ORDER)
        self.assertEqual(order("C3", "C4"), AROMATIC_ORDER)
        self.assertEqual(order("S1", "O1"), DOUBLE)
        self.assertEqual(order("S1", "O2"), DOUBLE)
        self.assertEqual(order("S1", "N1"), SINGLE)
        self.assertEqual(order("C1", "S1"), SINGLE)

    def test_explicit_table_is_trusted(self):
        atoms = benzothiazole_like()
        bonds = infer_bonds_from_geometry(atoms)
        explicit = {bond_key(i, j): SINGLE for i, j in bonds}
        explicit[bond_key(0, 1)] = DOUBLE

        orders = perceive_bond_orders(atoms, bonds, existing=explicit)
        self.assertEqual(orders[bond_key(0, 1)], DOUBLE)

    def test_empty_input(self):
        self.assertEqual(perceive_bond_orders([], []), {})


if __name__ == "__main__":
    unittest.main()
