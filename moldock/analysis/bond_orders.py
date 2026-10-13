"""
Bond-order and ring perception for 2D ligand depictions.

Docking input files (PDB, PDBQT) carry no bond-order information, so a flat
"everything is a single bond" depiction makes rings look like plain hexagons
and turns carbonyl / sulfonyl groups into misleading single bonds.  This module
recovers chemically sensible bond orders straight from the 3D coordinates:

* **Rings** are detected with a smallest-cycle search over the bond graph.
* **Aromatic** 5- and 6-membered carbon/nitrogen/oxygen/sulfur rings are
  recognised when their bonds are measurably shorter than a pure single bond.
* Remaining **double bonds** are assigned when the measured bond length is at
  least ``DOUBLE_BOND_RATIO`` shorter than the sum of the covalent radii, and
  both endpoints still have spare valence.

Everything is a heuristic — it is used only for drawing, never for scoring —
so the rules deliberately stay conservative: anything ambiguous is left as a
single bond.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from moldock.core.atom import Atom

# A bond key is always stored with the smaller index first.
BondKey = Tuple[int, int]

AROMATIC_ORDER = 4
SINGLE, DOUBLE, TRIPLE = 1, 2, 3

RING_MAX_SIZE = 7
AROMATIC_RING_SIZES = (5, 6)
AROMATIC_RING_ELEMENTS = frozenset({"C", "N", "O", "S"})

# d / (r_i + r_j) below this value => the bond is short enough to be a double bond
DOUBLE_BOND_RATIO = 0.90
# mean ring bond ratio below this => the ring is aromatic
AROMATIC_RING_RATIO = 0.97

# Covalent radii (Å) — single-bond reference lengths
COVALENT_RADII: Dict[str, float] = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "P": 1.07, "S": 1.05, "CL": 1.02,
    "BR": 1.20, "I": 1.39,
}
DEFAULT_COVALENT_RADIUS = 0.75

# Highest bond order an element can support in a neutral molecule
VALENCE_TARGETS: Dict[str, int] = {
    "C": 4, "N": 3, "O": 2, "S": 2, "P": 5,
    "F": 1, "CL": 1, "BR": 1, "I": 1, "H": 1,
}
DEFAULT_VALENCE = 4

# Bond search window: distances between 0.4 Å and (r_i + r_j + tolerance) count
MIN_BOND_DISTANCE = 0.4
BOND_SEARCH_TOLERANCE = 0.45


def bond_key(i: int, j: int) -> BondKey:
    """Return the canonical (sorted) key for a bond between *i* and *j*."""
    return (i, j) if i <= j else (j, i)


def _valence_target(element: str, degree: int) -> int:
    """Maximum bond-order sum an atom of *element* can carry."""
    if element == "S":
        # Sulfur is 2-valent in thioethers, 6-valent in sulfones/sulfates.
        return 6 if degree >= 3 else 2
    return VALENCE_TARGETS.get(element, DEFAULT_VALENCE)


def build_adjacency(n_atoms: int, bonds: Sequence[Tuple[int, int]]) -> List[List[int]]:
    """Build a neighbour list for a bond list of 0-based atom indices."""
    adj: List[List[int]] = [[] for _ in range(n_atoms)]
    for raw_i, raw_j in bonds:
        if not (0 <= raw_i < n_atoms and 0 <= raw_j < n_atoms) or raw_i == raw_j:
            continue
        adj[raw_i].append(raw_j)
        adj[raw_j].append(raw_i)
    return adj


def _bond_length(atoms: Sequence[Atom], i: int, j: int) -> float:
    return float(np.linalg.norm(atoms[i].coord - atoms[j].coord))


def _bond_ratio(atoms: Sequence[Atom], i: int, j: int) -> float:
    """Measured bond length relative to the sum of the two covalent radii."""
    r_i = COVALENT_RADII.get(atoms[i].element.upper(), DEFAULT_COVALENT_RADIUS)
    r_j = COVALENT_RADII.get(atoms[j].element.upper(), DEFAULT_COVALENT_RADIUS)
    reference = r_i + r_j
    if reference <= 1e-6:
        return 1.0
    return _bond_length(atoms, i, j) / reference


def _shortest_path(adj: Sequence[Sequence[int]], start: int, goal: int, max_len: int) -> List[int]:
    """Breadth-first shortest path from *start* to *goal* (endpoints included)."""
    parent: Dict[int, int] = {start: -1}
    frontier = [start]
    depth = 0
    while frontier and depth < max_len:
        nxt: List[int] = []
        for node in frontier:
            for neighbour in adj[node]:
                if neighbour in parent:
                    continue
                parent[neighbour] = node
                if neighbour == goal:
                    path = [goal]
                    step = goal
                    while parent[step] != -1:
                        step = parent[step]
                        path.append(step)
                    return list(reversed(path))
                nxt.append(neighbour)
        frontier = nxt
        depth += 1
    return []


def find_rings(
    adj: Sequence[Sequence[int]],
    bonds: Sequence[Tuple[int, int]],
    max_size: int = RING_MAX_SIZE,
) -> List[Tuple[int, ...]]:
    """Return the unique smallest rings of the bond graph.

    Each bond is temporarily removed and the shortest alternative path between
    its two endpoints is traced; path + bond forms a candidate ring.  Duplicate
    rings (same atom set) are collapsed.
    """
    rings: List[Tuple[int, ...]] = []
    seen: Set[Tuple[int, ...]] = set()

    for raw_i, raw_j in bonds:
        # Neighbours of i and j with the candidate bond masked out
        i, j = bond_key(raw_i, raw_j)
        masked = [[n for n in adj[k] if not (k == i and n == j) and not (k == j and n == i)]
                  for k in range(len(adj))]
        path = _shortest_path(masked, i, j, max_size)
        if not path or len(path) > max_size:
            continue
        ring = tuple(sorted(path))
        if len(ring) < 3 or ring in seen:
            continue
        seen.add(ring)
        rings.append(ring)

    return rings


def infer_bonds_from_geometry(atoms: Sequence[Atom]) -> List[Tuple[int, int]]:
    """Infer the molecular graph from interatomic distances.

    Used when a structure carries no explicit connection table (PDB / PDBQT
    input).  Two atoms are bonded when their separation falls between
    ``MIN_BOND_DISTANCE`` and the sum of their covalent radii plus a tolerance.
    """
    bonds: List[Tuple[int, int]] = []
    n_atoms = len(atoms)

    for i in range(n_atoms):
        r_i = COVALENT_RADII.get(atoms[i].element.upper(), DEFAULT_COVALENT_RADIUS)
        for j in range(i + 1, n_atoms):
            r_j = COVALENT_RADII.get(atoms[j].element.upper(), DEFAULT_COVALENT_RADIUS)
            distance = _bond_length(atoms, i, j)
            if MIN_BOND_DISTANCE <= distance <= r_i + r_j + BOND_SEARCH_TOLERANCE:
                bonds.append((i, j))

    return bonds


def detect_aromatic_bonds(
    atoms: Sequence[Atom],
    adj: Sequence[Sequence[int]],
    rings: Sequence[Sequence[int]],
) -> Set[BondKey]:
    """Identify bonds that belong to an aromatic ring."""
    aromatic: Set[BondKey] = set()

    for ring in rings:
        if len(ring) not in AROMATIC_RING_SIZES:
            continue
        if any(atoms[a].element.upper() not in AROMATIC_RING_ELEMENTS for a in ring):
            continue
        # Every ring atom must be sp2-capable (degree ≤ 3 for C/N)
        if any(len(adj[a]) > 3 for a in ring):
            continue
        # Ring must be measurably shorter than a pure single-bonded ring
        ordered = list(ring) + [ring[0]]
        ratios = [_bond_ratio(atoms, ordered[k], ordered[k + 1]) for k in range(len(ring))]
        if sum(ratios) / len(ratios) > AROMATIC_RING_RATIO:
            continue

        for k in range(len(ring)):
            aromatic.add(bond_key(ordered[k], ordered[k + 1]))

    return aromatic


def _order_sums(n_atoms: int, orders: Dict[BondKey, int]) -> List[int]:
    """Total bond order per atom, counting aromatic bonds as order 1.5."""
    totals = [0.0] * n_atoms
    for (i, j), order in orders.items():
        weight = 1.5 if order == AROMATIC_ORDER else float(order)
        totals[i] += weight
        totals[j] += weight
    return [int(round(t)) for t in totals]


def perceive_bond_orders(
    atoms: Sequence[Atom],
    bonds: Sequence[Tuple[int, int]],
    existing: Dict[Tuple[int, int], int] | None = None,
) -> Dict[BondKey, int]:
    """Return a bond-order map for *bonds*.

    Parameters
    ----------
    atoms:
        Atoms whose 3D coordinates define the bond lengths.
    bonds:
        Pairs of 0-based atom indices.
    existing:
        Bond orders read from an explicit connection table (e.g. SDF).  When
        supplied it is trusted verbatim and only re-keyed canonically.
    """
    n_atoms = len(atoms)

    if existing:
        cleaned: Dict[BondKey, int] = {}
        for (i, j), order in existing.items():
            if 0 <= i < n_atoms and 0 <= j < n_atoms and i != j:
                cleaned[bond_key(i, j)] = int(order)
        if cleaned:
            return cleaned

    clean_bonds = [bond_key(i, j) for i, j in bonds
                   if 0 <= i < n_atoms and 0 <= j < n_atoms and i != j]
    orders: Dict[BondKey, int] = {key: SINGLE for key in clean_bonds}

    adj = build_adjacency(n_atoms, clean_bonds)
    rings = find_rings(adj, clean_bonds)

    for key in detect_aromatic_bonds(atoms, adj, rings):
        if key in orders:
            orders[key] = AROMATIC_ORDER

    # ---- Promote short non-aromatic bonds to double bonds --------------------
    totals = _order_sums(n_atoms, orders)
    candidates: List[Tuple[float, BondKey]] = []
    for key in clean_bonds:
        if orders[key] != SINGLE:
            continue
        ratio = _bond_ratio(atoms, key[0], key[1])
        if ratio <= DOUBLE_BOND_RATIO:
            candidates.append((ratio, key))

    for _ratio, (i, j) in sorted(candidates):
        if orders[(i, j)] != SINGLE:
            continue
        target_i = _valence_target(atoms[i].element.upper(), len(adj[i]))
        target_j = _valence_target(atoms[j].element.upper(), len(adj[j]))
        if totals[i] + 1 <= target_i and totals[j] + 1 <= target_j:
            orders[(i, j)] = DOUBLE
            totals[i] += 1
            totals[j] += 1

    return orders


def bond_order_label(order: int) -> str:
    """Human-readable description used by the legend and tests."""
    return {
        SINGLE: "single",
        DOUBLE: "double",
        TRIPLE: "triple",
        AROMATIC_ORDER: "aromatic",
    }.get(order, "single")
