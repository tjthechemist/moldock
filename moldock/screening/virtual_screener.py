"""
Virtual Screening engine — dock a library of ligands against a single receptor
and aggregate the ranked outcomes.

Features
--------
* Accepts ligand libraries as:
    - A single multi-molecule ``.sdf`` file  (most common)
    - A directory of ``.sdf``/``.mol``/``.pdb``/``.pdbqt`` files
    - An explicit Python list of :class:`~moldock.core.molecule.Molecule` objects
* Parallel execution via :mod:`concurrent.futures` (configurable worker count).
* Optional early-termination filter: discard any ligand whose top docking
  score is above ``score_cutoff`` (default: no cutoff).
* Progress callback for CLI / notebook use.
* Returns a :class:`ScreeningCampaign` that can be sorted, filtered, and
  exported to CSV / PDB / SDF without repeating docking.
"""

from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import (
    Callable, Dict, List, Optional, Sequence, Tuple, Union
)

import numpy as np

from moldock.core.config import DockingConfig, GridBox, MetalConfig
from moldock.core.molecule import Molecule
from moldock.core.pdb_io import PDBIO
from moldock.core.sdf_io import SDFIO
from moldock.engines.base import DockingResult, DockingPose
from moldock.engines.protein_ligand import ProteinLigandEngine
from moldock.metals.detector import MetalCenter
from moldock.preparation.ligand_prep import LigandPreparer
from moldock.preparation.receptor_prep import ReceptorPreparer


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScreeningResult:
    """Outcome of docking a single ligand against the receptor.

    Attributes
    ----------
    ligand_name : str
        Molecule name (from SDF title line or filename stem).
    ligand_index : int
        Zero-based index in the original library.
    top_score : float
        Best binding affinity found (kcal/mol, lower = better).
    top_pose : DockingPose | None
        The highest-ranked docked pose, or ``None`` if docking failed.
    docking_result : DockingResult | None
        Full result object (all poses + breakdown), or ``None`` on failure.
    error : str | None
        Exception message if docking raised an error.
    elapsed_seconds : float
        Wall-clock time spent docking this ligand.
    num_rotatable_bonds : int
        Detected number of rotatable bonds in the ligand.
    sdf_data : Dict[str, str]
        Arbitrary SDF data fields carried through from the input file.
    """

    ligand_name: str
    ligand_index: int
    top_score: float = 0.0
    top_pose: Optional[DockingPose] = None
    docking_result: Optional[DockingResult] = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    num_rotatable_bonds: int = 0
    sdf_data: Dict[str, str] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """True when docking completed without error and produced at least one pose."""
        return self.error is None and self.top_pose is not None

    @property
    def score_breakdown(self) -> Dict:
        if self.top_pose is not None:
            return self.top_pose.score_breakdown
        return {}


@dataclass
class ScreeningCampaign:
    """Aggregates all :class:`ScreeningResult` objects from one virtual screening run.

    Results are stored in insertion order (library order). Call
    :meth:`ranked_hits` to retrieve them sorted by docking score.
    """

    receptor_name: str
    results: List[ScreeningResult] = field(default_factory=list)
    total_ligands: int = 0
    elapsed_seconds: float = 0.0
    config_summary: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def ranked_hits(
        self,
        top_n: Optional[int] = None,
        score_cutoff: Optional[float] = None,
        successful_only: bool = True,
    ) -> List[ScreeningResult]:
        """Return results sorted best-score-first.

        Parameters
        ----------
        top_n:
            Keep at most this many hits. ``None`` keeps all.
        score_cutoff:
            Discard any hit with ``top_score > score_cutoff``.
        successful_only:
            Exclude failed / errored ligands (default ``True``).
        """
        hits = [r for r in self.results if (not successful_only or r.success)]
        if score_cutoff is not None:
            hits = [r for r in hits if r.top_score <= score_cutoff]
        hits.sort(key=lambda r: r.top_score)
        if top_n is not None:
            hits = hits[:top_n]
        return hits

    @property
    def best_hit(self) -> Optional[ScreeningResult]:
        hits = self.ranked_hits(top_n=1)
        return hits[0] if hits else None

    @property
    def failed(self) -> List[ScreeningResult]:
        return [r for r in self.results if not r.success]

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)


# ---------------------------------------------------------------------------
# Helper: load a single ligand molecule from various input types
# ---------------------------------------------------------------------------

def _load_ligand_file(filepath: str) -> List[Molecule]:
    """Return prepared Molecule objects from a ligand file path."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".sdf", ".mol"):
        return LigandPreparer.prepare_from_sdf_batch(filepath)
    else:
        return [LigandPreparer.prepare_ligand(filepath)]


def _collect_ligands_from_dir(directory: str) -> List[Molecule]:
    """Walk *directory* and load all recognised ligand files."""
    supported = {".sdf", ".mol", ".pdb", ".pdbqt"}
    mols: List[Molecule] = []
    for fname in sorted(os.listdir(directory)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in supported:
            fpath = os.path.join(directory, fname)
            try:
                mols.extend(_load_ligand_file(fpath))
            except Exception as e:
                print(f"  [WARN] Could not load {fname}: {e}")
    return mols


# ---------------------------------------------------------------------------
# Worker function (module-level so it is picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _dock_one(
    args: Tuple,
) -> ScreeningResult:
    """Dock a single ligand against a pre-loaded receptor.

    This function is called inside a worker process/thread.  Because
    :class:`Molecule` objects are not serialisation-friendly for
    multiprocessing across all platforms we pass them by reference when
    using threads (``use_threads=True``), and re-load from the SDF bytes
    when using processes.

    Parameters
    ----------
    args : tuple
        ``(ligand_mol, ligand_index, receptor_mol, metal_centers, config, score_cutoff)``
    """
    (
        ligand_mol,
        ligand_index,
        receptor_mol,
        metal_centers,
        config,
        score_cutoff,
    ) = args

    t0 = time.perf_counter()
    result = ScreeningResult(
        ligand_name=ligand_mol.name,
        ligand_index=ligand_index,
        num_rotatable_bonds=ligand_mol.metadata.get("num_rotatable_bonds", 0),
        sdf_data=ligand_mol.metadata.get("sdf_data", {}),
    )

    try:
        engine = ProteinLigandEngine(
            receptor=receptor_mol,
            ligand=ligand_mol,
            config=config,
            metal_centers=metal_centers,
        )
        dr = engine.dock()
        result.docking_result = dr

        if dr.poses:
            top = dr.poses[0]
            result.top_pose = top
            result.top_score = top.score
        else:
            result.top_score = 0.0

        # Apply optional score filter early
        if score_cutoff is not None and result.top_score > score_cutoff:
            result.top_pose = None   # mark as not a hit

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"

    result.elapsed_seconds = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Main screener
# ---------------------------------------------------------------------------

class VirtualScreener:
    """Screen a library of ligands against a single receptor.

    Parameters
    ----------
    receptor_pdb : str
        Path to the receptor PDB file.
    config : DockingConfig, optional
        Docking configuration (grid, exhaustiveness, …).
    metal_config : MetalConfig, optional
        Which metal ions to retain and parameterise.
    score_cutoff : float, optional
        Ligands with ``top_score > score_cutoff`` are classified as
        non-hits (still stored, but flagged). ``None`` keeps all.
    max_workers : int, optional
        Number of parallel workers.  ``1`` disables parallelism (safe for
        debugging).  ``None`` uses ``os.cpu_count()``.
    use_threads : bool
        Use :class:`~concurrent.futures.ThreadPoolExecutor` instead of
        :class:`~concurrent.futures.ProcessPoolExecutor`.  Threads share
        memory (no serialisation overhead) and work better in Jupyter
        notebooks.  Defaults to ``True``.
    progress_callback : callable, optional
        Called after each ligand finishes:
        ``callback(done: int, total: int, result: ScreeningResult)``.
    """

    def __init__(
        self,
        receptor_pdb: str,
        config: Optional[DockingConfig] = None,
        metal_config: Optional[MetalConfig] = None,
        score_cutoff: Optional[float] = None,
        max_workers: int = 1,
        use_threads: bool = True,
        progress_callback: Optional[Callable[[int, int, ScreeningResult], None]] = None,
    ):
        self._receptor_pdb = receptor_pdb
        self._config = config or DockingConfig()
        self._metal_config = metal_config or MetalConfig()
        self._score_cutoff = score_cutoff
        self._max_workers = max_workers if max_workers and max_workers > 0 else 1
        self._use_threads = use_threads
        self._progress_callback = progress_callback

        # Prepare receptor once (shared across all ligands)
        prep = ReceptorPreparer(metal_config=self._metal_config)
        self._receptor, self._metal_centers = prep.prepare_from_pdb(receptor_pdb)

    # ------------------------------------------------------------------
    # Library loading helpers (public)
    # ------------------------------------------------------------------

    def load_library_from_sdf(self, sdf_path: str) -> List[Molecule]:
        """Load all ligands from a multi-molecule SDF file."""
        return LigandPreparer.prepare_from_sdf_batch(sdf_path)

    def load_library_from_dir(self, directory: str) -> List[Molecule]:
        """Load all recognised ligand files found in *directory*."""
        return _collect_ligands_from_dir(directory)

    def load_library_from_files(self, filepaths: Sequence[str]) -> List[Molecule]:
        """Load ligands from an explicit list of file paths."""
        mols: List[Molecule] = []
        for fp in filepaths:
            try:
                mols.extend(_load_ligand_file(fp))
            except Exception as e:
                print(f"  [WARN] Skipping {fp}: {e}")
        return mols

    # ------------------------------------------------------------------
    # Core screening method
    # ------------------------------------------------------------------

    def screen(
        self,
        library: Union[str, Sequence[str], List[Molecule]],
    ) -> ScreeningCampaign:
        """Run virtual screening on *library*.

        Parameters
        ----------
        library : str | list[str] | list[Molecule]
            Accepts:

            * A **string** path — if it ends in ``.sdf`` / ``.mol`` it is loaded
              as a multi-molecule SDF; if it is a directory every ligand file
              inside is loaded.
            * A **list of strings** — each element is a path to a single
              ligand file.
            * A **list of Molecule** — pre-loaded molecules (e.g. after
              calling :meth:`load_library_from_sdf` yourself).

        Returns
        -------
        ScreeningCampaign
            Ranked and filterable collection of :class:`ScreeningResult` objects.
        """
        # ---- Resolve library ----
        if isinstance(library, str):
            if os.path.isdir(library):
                ligands = self.load_library_from_dir(library)
            else:
                ligands = self.load_library_from_sdf(library)
        elif library and isinstance(library[0], str):
            ligands = self.load_library_from_files(library)  # type: ignore[arg-type]
        else:
            ligands = list(library)  # type: ignore[arg-type]

        if not ligands:
            raise ValueError("Library is empty — no ligands were loaded.")

        total = len(ligands)
        campaign = ScreeningCampaign(
            receptor_name=self._receptor.name,
            total_ligands=total,
            config_summary={
                "exhaustiveness": self._config.exhaustiveness,
                "num_modes": self._config.num_modes,
                "score_cutoff": self._score_cutoff,
                "max_workers": self._max_workers,
            },
        )

        print(f"\n{'='*64}")
        print(f"  VIRTUAL SCREENING  —  {total} ligands vs. {self._receptor.name}")
        print(f"  Workers: {self._max_workers}  |  Score cutoff: {self._score_cutoff}")
        print(f"{'='*64}\n")

        wall_t0 = time.perf_counter()

        # Build argument tuples
        arg_list = [
            (
                ligands[i],
                i,
                self._receptor,
                self._metal_centers,
                self._config,
                self._score_cutoff,
            )
            for i in range(total)
        ]

        results: List[ScreeningResult] = [None] * total  # type: ignore[list-item]

        if self._max_workers == 1:
            # Single-threaded path (simpler, better tracebacks)
            for idx, arg in enumerate(arg_list):
                sr = _dock_one(arg)
                results[sr.ligand_index] = sr
                self._print_progress(sr, idx + 1, total)
                if self._progress_callback:
                    self._progress_callback(idx + 1, total, sr)
        else:
            Executor = ThreadPoolExecutor if self._use_threads else ProcessPoolExecutor
            futures = {}
            with Executor(max_workers=self._max_workers) as pool:
                for arg in arg_list:
                    fut = pool.submit(_dock_one, arg)
                    futures[fut] = arg[1]  # ligand_index

                done_count = 0
                for fut in as_completed(futures):
                    sr: ScreeningResult = fut.result()
                    results[sr.ligand_index] = sr
                    done_count += 1
                    self._print_progress(sr, done_count, total)
                    if self._progress_callback:
                        self._progress_callback(done_count, total, sr)

        campaign.results = [r for r in results if r is not None]
        campaign.elapsed_seconds = time.perf_counter() - wall_t0

        # Summary
        hits = campaign.ranked_hits()
        failed = len(campaign.failed)
        print(f"\n{'='*64}")
        print(f"  Screening complete in {campaign.elapsed_seconds:.1f}s")
        print(f"  Successful: {len(hits)} / {total}   |   Failed: {failed}")
        if hits:
            best = hits[0]
            print(f"  Best hit: {best.ligand_name}  ({best.top_score:.2f} kcal/mol)")
        print(f"{'='*64}\n")

        return campaign

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_progress(sr: ScreeningResult, done: int, total: int) -> None:
        status = f"{sr.top_score:+.2f} kcal/mol" if sr.success else f"FAILED: {sr.error}"
        pct = done / total * 100
        bar_filled = int(pct / 5)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        print(f"  [{bar}] {done:>4}/{total}  {sr.ligand_name:<28}  {status}")

