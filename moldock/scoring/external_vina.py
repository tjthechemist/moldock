"""
Integration wrapper for official AutoDock Vina Python library or CLI binary.
Gracefully falls back to pure-Python VinaScoringFunction if not installed.
"""

import shutil
import subprocess
import tempfile
import os
from typing import Optional, Tuple
from moldock.core.molecule import Molecule
from moldock.core.pdb_io import PDBQTIO
from moldock.core.config import GridBox


class ExternalVinaBackend:
    """
    Bridge to external AutoDock Vina installations.
    Supports official `vina` Python C-extension or `vina` command-line executable.
    """

    @staticmethod
    def is_python_vina_available() -> bool:
        try:
            import vina  # type: ignore
            return True
        except ImportError:
            return False

    @staticmethod
    def is_cli_vina_available() -> bool:
        return shutil.which("vina") is not None

    @classmethod
    def score_with_official_vina(
        cls,
        receptor_pdbqt_path: str,
        ligand_pdbqt_path: str,
        grid: Optional[GridBox] = None,
    ) -> Optional[float]:
        """
        Evaluate binding energy using official AutoDock Vina if installed.
        """
        if cls.is_python_vina_available():
            try:
                from vina import Vina  # type: ignore
                v = Vina(sf_name='vina', verbosity=0)
                v.set_receptor(receptor_pdbqt_path)
                v.set_ligand_from_file(ligand_pdbqt_path)
                energy = v.score()
                return float(energy[0])  # total affinity
            except Exception:
                pass

        if cls.is_cli_vina_available():
            try:
                cmd = [
                    "vina",
                    "--receptor", receptor_pdbqt_path,
                    "--ligand", ligand_pdbqt_path,
                    "--score_only",
                ]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for line in proc.stdout.splitlines():
                    if "Affinity:" in line:
                        parts = line.split()
                        return float(parts[1])
            except Exception:
                pass

        return None

