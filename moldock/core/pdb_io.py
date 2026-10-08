"""
PDB and PDBQT Input/Output handling.
Parses atomic coordinates, residue information, partial charges, and AutoDock atom types.
"""

import os
from typing import List, Optional, Tuple, Dict
import numpy as np
from moldock.core.atom import Atom, assign_vina_atom_type
from moldock.core.molecule import Molecule


class PDBIO:
    """Reader and writer for standard Protein Data Bank (.pdb) files."""

    @staticmethod
    def read_pdb(filepath: str, name: Optional[str] = None) -> Molecule:
        """Parse a standard .pdb file into a Molecule object."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"PDB file not found: {filepath}")

        mol_name = name or os.path.splitext(os.path.basename(filepath))[0]
        atoms: List[Atom] = []

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                record = line[:6].strip()
                if record in ("ATOM", "HETATM"):
                    try:
                        idx = int(line[6:11].strip())
                        atom_name = line[12:16].strip()
                        res_name = line[17:20].strip()
                        chain_id = line[21:22].strip() or "A"
                        res_seq = int(line[22:26].strip()) if line[22:26].strip() else 1

                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())

                        occ = float(line[54:60].strip()) if len(line) >= 60 and line[54:60].strip() else 1.0
                        b_fac = float(line[60:66].strip()) if len(line) >= 66 and line[60:66].strip() else 0.0

                        elem = line[76:78].strip() if len(line) >= 78 and line[76:78].strip() else ""
                        if not elem:
                            # Fallback to atom name
                            elem = "".join([c for c in atom_name if c.isalpha()])[:2]

                        vina_type = assign_vina_atom_type(atom_name, elem, res_name)

                        atom = Atom(
                            index=idx,
                            name=atom_name,
                            element=elem,
                            res_name=res_name,
                            res_seq=res_seq,
                            chain_id=chain_id,
                            coord=np.array([x, y, z], dtype=np.float64),
                            charge=0.0,
                            atom_type=vina_type,
                            b_factor=b_fac,
                            occupancy=occ,
                            is_hetatm=(record == "HETATM"),
                        )
                        atoms.append(atom)
                    except (ValueError, IndexError):
                        continue

        mol = Molecule(name=mol_name, atoms=atoms)
        return mol

    @staticmethod
    def write_pdb(molecule: Molecule, filepath: str, model_id: Optional[int] = None) -> None:
        """Write a Molecule into a standard .pdb file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        mode = "a" if model_id is not None and model_id > 1 else "w"

        with open(filepath, mode, encoding="utf-8") as f:
            if model_id is not None:
                f.write(f"MODEL        {model_id:4d}\n")

            for i, atom in enumerate(molecule.atoms, start=1):
                rec = "HETATM" if atom.is_hetatm or atom.is_metal else "ATOM  "
                elem = atom.element[:2].rjust(2)
                f.write(
                    f"{rec:<6}{i:5d} {atom.name:<4} {atom.res_name:3} {atom.chain_id:1}"
                    f"{atom.res_seq:4d}    "
                    f"{atom.coord[0]:8.3f}{atom.coord[1]:8.3f}{atom.coord[2]:8.3f}"
                    f"{atom.occupancy:6.2f}{atom.b_factor:6.2f}          {elem:>2}\n"
                )

            if model_id is not None:
                f.write("ENDMDL\n")
            else:
                f.write("END\n")


class PDBQTIO:
    """Reader and writer for AutoDock / AutoDock Vina .pdbqt format."""

    @staticmethod
    def read_pdbqt(filepath: str, name: Optional[str] = None) -> Molecule:
        """Parse a .pdbqt file with AutoDock atom types and partial charges."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"PDBQT file not found: {filepath}")

        mol_name = name or os.path.splitext(os.path.basename(filepath))[0]
        atoms: List[Atom] = []

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                record = line[:6].strip()
                if record in ("ATOM", "HETATM"):
                    try:
                        idx = int(line[6:11].strip())
                        atom_name = line[12:16].strip()
                        res_name = line[17:20].strip()
                        chain_id = line[21:22].strip() or "A"
                        res_seq = int(line[22:26].strip()) if line[22:26].strip() else 1

                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())

                        # PDBQT columns: charge (approx col 70-76), atom_type (col 77-79)
                        charge = float(line[69:76].strip()) if len(line) >= 76 and line[69:76].strip() else 0.0
                        atom_type = line[77:79].strip() if len(line) >= 79 else ""

                        if not atom_type:
                            elem = "".join([c for c in atom_name if c.isalpha()])[:2]
                            atom_type = assign_vina_atom_type(atom_name, elem, res_name)

                        elem = "".join([c for c in atom_type if c.isalpha()])

                        atom = Atom(
                            index=idx,
                            name=atom_name,
                            element=elem,
                            res_name=res_name,
                            res_seq=res_seq,
                            chain_id=chain_id,
                            coord=np.array([x, y, z], dtype=np.float64),
                            charge=charge,
                            atom_type=atom_type,
                            is_hetatm=(record == "HETATM"),
                        )
                        atoms.append(atom)
                    except (ValueError, IndexError):
                        continue

        return Molecule(name=mol_name, atoms=atoms)

    @staticmethod
    def write_pdbqt(
        molecule: Molecule,
        filepath: str,
        torsdof: int = 0,
        model_id: Optional[int] = None,
        as_rigid: bool = True,
    ) -> None:
        """Write a Molecule into an AutoDock/Vina compliant .pdbqt file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        mode = "a" if model_id is not None and model_id > 1 else "w"

        with open(filepath, mode, encoding="utf-8") as f:
            if model_id is not None:
                f.write(f"MODEL {model_id}\n")

            if not as_rigid:
                f.write("ROOT\n")

            for i, atom in enumerate(molecule.atoms, start=1):
                rec = "HETATM" if atom.is_hetatm or atom.is_metal else "ATOM  "
                atom_type = atom.atom_type.rjust(2)
                f.write(
                    f"{rec:<6}{i:5d} {atom.name:<4} {atom.res_name:3} {atom.chain_id:1}"
                    f"{atom.res_seq:4d}    "
                    f"{atom.coord[0]:8.3f}{atom.coord[1]:8.3f}{atom.coord[2]:8.3f}"
                    f"{atom.occupancy:6.2f}{atom.b_factor:6.2f}    "
                    f"{atom.charge:+7.3f} {atom_type:<2}\n"
                )

            if not as_rigid:
                f.write("ENDROOT\n")
                f.write(f"TORSDOF {torsdof}\n")

            if model_id is not None:
                f.write("ENDMDL\n")
