"""
MDL SDF / MOL V2000 and V3000 reader — pure Python, zero external dependencies.

Parses:
  - V2000 connection tables (the most common legacy format)
  - V3000 connection tables (modern extended format)
  - Multiple molecules in a single .sdf file (record delimiter: $$$$)
  - Atom block: element symbols, 3D coordinates, and formal charges
  - Bond block: bond-order, used to refine atom typing (aromatic)
  - SDF data fields (M  END property block), exposed in Molecule.metadata["sdf_data"]

Returns one or more :class:`~moldock.core.molecule.Molecule` objects, each populated
with fully typed :class:`~moldock.core.atom.Atom` instances ready for docking.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Dict, Tuple

import numpy as np

from moldock.core.atom import Atom, assign_vina_atom_type
from moldock.core.molecule import Molecule

# ---------------------------------------------------------------------------
# Formal charge conversion table (V2000 charge flag → actual charge)
# ---------------------------------------------------------------------------
_V2000_CHARGE_MAP: Dict[int, float] = {
    0: 0.0,
    1: 3.0,
    2: 2.0,
    3: 1.0,
    4: 0.0,   # doublet radical — treat as neutral
    5: -1.0,
    6: -2.0,
    7: -3.0,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_element(raw: str) -> str:
    """Return an upper-case, stripped element symbol from a raw SDF field."""
    return raw.strip().rstrip("0123456789+-").upper()


def _vina_type_from_element_and_aromatic(element: str, aromatic: bool) -> str:
    """Quick AutoDock Vina type assignment for SDF atoms.

    Aromatic carbons → 'A', polar oxygen → 'OA', etc.
    Falls back to :func:`assign_vina_atom_type` for anything unusual.
    """
    elem = element.upper()
    if elem == "C":
        return "A" if aromatic else "C"
    if elem == "N":
        return "NA" if aromatic else "N"
    if elem == "O":
        return "OA"
    if elem == "S":
        return "SA"
    if elem == "H":
        return "HD"
    if elem in ("F", "CL", "BR", "I", "P"):
        return elem
    # Metals and everything else
    return assign_vina_atom_type(elem, elem, "LIG")


# ---------------------------------------------------------------------------
# V2000 parser
# ---------------------------------------------------------------------------

def _parse_v2000(lines: List[str], mol_name: str) -> Molecule:
    """Parse a V2000 connection table block into a :class:`Molecule`."""

    # ---- Find the counts line -------------------------------------------------
    # SDF header = 3 lines (name / program/date / comment) followed by counts.
    # After splitting on $$$$, some records start with blank lines, so we
    # locate the counts line by finding the first non-empty line that looks like
    # a molfile counts line: starts with digits in columns 0-6.
    counts_idx = -1
    header_lines_seen = 0
    for idx, line in enumerate(lines):
        # The 3 header lines may be blank; skip them
        if header_lines_seen < 3:
            header_lines_seen += 1
            continue
        stripped = line.strip()
        if not stripped:
            continue   # tolerate extra blank lines between records
        # Counts line: columns [0:3] and [3:6] are integers
        if len(line) >= 6:
            counts_idx = idx
            break

    if counts_idx < 0:
        raise ValueError(f"Could not locate V2000 counts line in: {mol_name!r}")

    counts_line = lines[counts_idx]
    try:
        n_atoms = int(counts_line[0:3].strip())
        n_bonds = int(counts_line[3:6].strip())
    except ValueError as exc:
        raise ValueError(f"Cannot parse V2000 counts line: {counts_line!r}") from exc

    # ---- Atom block (lines immediately after counts line) --------------------
    atoms: List[Atom] = []
    formal_charges: Dict[int, float] = {}   # 0-based atom index → charge

    atom_start = counts_idx + 1

    for i in range(n_atoms):
        line = lines[atom_start + i]
        try:
            x = float(line[0:10].strip())
            y = float(line[10:20].strip())
            z = float(line[20:30].strip())
            sym_raw = line[31:34].strip()
            element = _clean_element(sym_raw) if sym_raw else "C"
            charge_flag = int(line[36:39].strip()) if len(line) > 38 and line[36:39].strip() else 0
        except (ValueError, IndexError):
            x, y, z, element, charge_flag = 0.0, 0.0, 0.0, "C", 0

        charge = _V2000_CHARGE_MAP.get(charge_flag, 0.0)

        atom = Atom(
            index=i + 1,
            name=f"{element}{i + 1}",
            element=element,
            res_name="LIG",
            res_seq=1,
            chain_id="L",
            coord=np.array([x, y, z], dtype=np.float64),
            charge=charge,
            atom_type=_vina_type_from_element_and_aromatic(element, False),
            is_hetatm=True,
        )
        atoms.append(atom)

    # ---- Bond block (lines 4+n_atoms … 4+n_atoms+n_bonds) ---------------
    bond_start = atom_start + n_atoms
    bonds: List[Tuple[int, int, int]] = []         # (atom_a_0idx, atom_b_0idx, order)
    aromatic_atoms: set = set()

    for j in range(n_bonds):
        if bond_start + j >= len(lines):
            break
        bline = lines[bond_start + j]
        try:
            a1 = int(bline[0:3].strip()) - 1
            a2 = int(bline[3:6].strip()) - 1
            order = int(bline[6:9].strip())
        except (ValueError, IndexError):
            continue
        bonds.append((a1, a2, order))
        if order == 4:   # aromatic bond type in V2000
            aromatic_atoms.add(a1)
            aromatic_atoms.add(a2)

    # Refine aromatic atom types
    for idx in aromatic_atoms:
        if 0 <= idx < len(atoms):
            atoms[idx].atom_type = _vina_type_from_element_and_aromatic(atoms[idx].element, True)

    # ---- Property block: M  CHG overrides formal charge -------------------
    for line in lines[bond_start + n_bonds:]:
        if line.startswith("M  CHG"):
            # Format: M  CHG  nn  a1  c1  a2  c2 …
            parts = line.split()
            try:
                count = int(parts[2])
                for k in range(count):
                    atom_idx = int(parts[3 + 2 * k]) - 1
                    chg_val = float(parts[4 + 2 * k])
                    if 0 <= atom_idx < len(atoms):
                        atoms[atom_idx].charge = chg_val
            except (IndexError, ValueError):
                pass
        elif line.strip() == "M  END":
            break

    mol = Molecule(name=mol_name, atoms=atoms)
    bond_pairs = [(a, b) for a, b, _o in bonds]
    mol.metadata["bonds"] = bond_pairs
    mol.metadata["bond_orders"] = {(a, b): o for a, b, o in bonds}
    return mol


# ---------------------------------------------------------------------------
# V3000 parser
# ---------------------------------------------------------------------------

def _parse_v3000(lines: List[str], mol_name: str) -> Molecule:
    """Parse a V3000 connection table block into a :class:`Molecule`."""
    atoms: List[Atom] = []
    bonds: List[Tuple[int, int, int]] = []
    aromatic_atoms: set = set()

    in_atom_block = False
    in_bond_block = False

    for line in lines:
        stripped = line.strip()

        if stripped == "M  V30 BEGIN ATOM":
            in_atom_block = True
            continue
        if stripped == "M  V30 END ATOM":
            in_atom_block = False
            continue
        if stripped == "M  V30 BEGIN BOND":
            in_bond_block = True
            continue
        if stripped == "M  V30 END BOND":
            in_bond_block = False
            continue

        if in_atom_block and stripped.startswith("M  V30"):
            # M  V30  idx  symbol  x  y  z  [aamap] [CHG=n]
            parts = stripped.split()
            if len(parts) < 7:
                continue
            try:
                idx = int(parts[2])
                symbol = _clean_element(parts[3])
                x = float(parts[4])
                y = float(parts[5])
                z = float(parts[6])
            except (ValueError, IndexError):
                continue

            charge = 0.0
            for part in parts[7:]:
                if part.upper().startswith("CHG="):
                    try:
                        charge = float(part.split("=")[1])
                    except ValueError:
                        pass

            atom = Atom(
                index=idx,
                name=f"{symbol}{idx}",
                element=symbol,
                res_name="LIG",
                res_seq=1,
                chain_id="L",
                coord=np.array([x, y, z], dtype=np.float64),
                charge=charge,
                atom_type=_vina_type_from_element_and_aromatic(symbol, False),
                is_hetatm=True,
            )
            atoms.append(atom)

        if in_bond_block and stripped.startswith("M  V30"):
            # M  V30  idx  order  atom1  atom2  [flags]
            parts = stripped.split()
            if len(parts) < 6:
                continue
            try:
                order = int(parts[3])
                a1 = int(parts[4]) - 1
                a2 = int(parts[5]) - 1
            except (ValueError, IndexError):
                continue
            bonds.append((a1, a2, order))
            if order == 4:
                aromatic_atoms.add(a1)
                aromatic_atoms.add(a2)

    # Refine aromatic types
    for idx in aromatic_atoms:
        if 0 <= idx < len(atoms):
            atoms[idx].atom_type = _vina_type_from_element_and_aromatic(atoms[idx].element, True)

    mol = Molecule(name=mol_name, atoms=atoms)
    mol.metadata["bonds"] = [(a, b) for a, b, _o in bonds]
    mol.metadata["bond_orders"] = {(a, b): o for a, b, o in bonds}
    return mol


# ---------------------------------------------------------------------------
# SDF data-field parser
# ---------------------------------------------------------------------------

def _parse_sdf_data(lines: List[str]) -> Dict[str, str]:
    """Extract SDF data fields (> <TAG>\nvalue) into a dict."""
    data: Dict[str, str] = {}
    tag: Optional[str] = None
    value_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        m = re.match(r"^>\s*<(.+?)>", stripped)
        if m:
            if tag is not None:
                data[tag] = "\n".join(value_lines).strip()
            tag = m.group(1)
            value_lines = []
        elif tag is not None:
            if stripped == "" and value_lines:
                # End of value block
                data[tag] = "\n".join(value_lines).strip()
                tag = None
                value_lines = []
            else:
                value_lines.append(stripped)

    if tag is not None:
        data[tag] = "\n".join(value_lines).strip()

    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SDFIO:
    """Reader for MDL SDF / MOL files (V2000 and V3000)."""

    @staticmethod
    def read_sdf(filepath: str, name: Optional[str] = None) -> List[Molecule]:
        """Parse an SDF file and return a list of :class:`Molecule` objects.

        Multi-molecule SDF files (records separated by ``$$$$``) yield one
        ``Molecule`` per record.  Single-molecule ``.mol`` files are also
        supported.

        Parameters
        ----------
        filepath:
            Path to the ``.sdf`` or ``.mol`` file.
        name:
            Optional base name; individual records are named ``<name>_<n>``.
            Defaults to the filename stem.

        Returns
        -------
        List[Molecule]
            One entry per ``$$$$`` record (at least one).

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        ValueError
            If the file cannot be parsed as a valid SDF/MOL.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"SDF/MOL file not found: {filepath}")

        base_name = name or os.path.splitext(os.path.basename(filepath))[0]

        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            raw_text = fh.read()

        # Split into records ($$$$  delimiter)
        record_texts = raw_text.split("$$$$")
        molecules: List[Molecule] = []

        for rec_idx, record_text in enumerate(record_texts):
            lines = record_text.splitlines()

            # Skip empty trailing records
            non_empty = [l for l in lines if l.strip()]
            if not non_empty:
                continue

            mol_name = f"{base_name}_{rec_idx + 1}" if len(record_texts) > 2 else base_name

            # Header line 0 may contain molecule name
            if lines and lines[0].strip():
                mol_name = lines[0].strip() or mol_name

            try:
                # Detect format version
                counts_line = lines[3] if len(lines) > 3 else ""
                if "V3000" in counts_line or "V3000" in "".join(lines[:10]):
                    mol = _parse_v3000(lines, mol_name)
                else:
                    mol = _parse_v2000(lines, mol_name)
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse SDF record {rec_idx + 1} in {filepath}: {exc}"
                ) from exc

            # Attach SDF data fields to metadata
            sdf_data = _parse_sdf_data(lines)
            mol.metadata["sdf_data"] = sdf_data

            molecules.append(mol)

        if not molecules:
            raise ValueError(f"No valid molecular records found in: {filepath}")

        return molecules

    @staticmethod
    def read_mol(filepath: str, name: Optional[str] = None) -> Molecule:
        """Convenience wrapper: read a single ``.mol`` file.

        Equivalent to ``SDFIO.read_sdf(filepath)[0]``.
        """
        mols = SDFIO.read_sdf(filepath, name=name)
        return mols[0]

    @staticmethod
    def write_sdf(molecules: List[Molecule], filepath: str) -> None:
        """Write one or more :class:`Molecule` objects to an SDF file (V2000).

        Writes minimal V2000 connection tables with coordinates and formal
        charges from ``atom.charge``.  Bonds are taken from
        ``mol.metadata["bonds"]`` when available; bond orders from
        ``mol.metadata["bond_orders"]`` (defaults to 1 = single).

        Parameters
        ----------
        molecules:
            List of molecules to write.
        filepath:
            Destination path (created / overwritten).
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # Reverse charge map for writing
        _CHARGE_WRITE: Dict[float, int] = {v: k for k, v in _V2000_CHARGE_MAP.items() if v != 0.0}

        with open(filepath, "w", encoding="utf-8") as fh:
            for mol in molecules:
                atoms = mol.atoms
                bonds: List[Tuple[int, int]] = mol.metadata.get("bonds", [])
                bond_orders: Dict[Tuple[int, int], int] = mol.metadata.get("bond_orders", {})

                fh.write(f"{mol.name}\n")
                fh.write("  moldock\n\n")

                n_atoms = len(atoms)
                n_bonds = len(bonds)
                fh.write(f"{n_atoms:3d}{n_bonds:3d}  0  0  0  0  0  0  0  0999 V2000\n")

                for atom in atoms:
                    chg_raw = _CHARGE_WRITE.get(round(atom.charge), 0)
                    x, y, z = atom.coord
                    fh.write(
                        f"{x:10.4f}{y:10.4f}{z:10.4f} {atom.element:<3}  0 {chg_raw:2d}"
                        f"  0  0  0  0  0  0  0  0  0  0\n"
                    )

                for a, b in bonds:
                    order = bond_orders.get((a, b), bond_orders.get((b, a), 1))
                    fh.write(f"{a + 1:3d}{b + 1:3d}{order:3d}  0  0  0  0\n")

                fh.write("M  END\n$$$$\n")

