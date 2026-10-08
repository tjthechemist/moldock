"""
Protein-Ligand docking engine.
Performs global conformational search (translation, rigid rotation, dihedral torsions)
scored with AutoDock Vina and metal co-factor coordination potentials.
"""

import time
import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from moldock.core.atom import Atom
from moldock.core.molecule import Molecule
from moldock.core.config import DockingConfig, GridBox
from moldock.metals.detector import MetalCenter
from moldock.scoring.composite_scoring import CompositeScoringFunction
from moldock.engines.base import BaseDockingEngine, DockingPose, DockingResult


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Compute 3x3 Rodrigues rotation matrix for rotation around an axis by angle."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C,      x * y * C - z * s,  x * z * C + y * s],
        [y * x * C + z * s,  c + y * y * C,      y * z * C - x * s],
        [z * x * C - y * s,  z * y * C + x * s,  c + z * z * C    ],
    ], dtype=np.float64)


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Generate a uniform random 3x3 rotation matrix in SO(3)."""
    # Shoemake subgroup algorithm
    u1, u2, u3 = rng.uniform(0.0, 1.0, 3)
    q0 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q1 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q2 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q3 = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    return np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),     1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1),     1 - 2*(q1**2 + q2**2)]
    ], dtype=np.float64)


class ProteinLigandEngine(BaseDockingEngine):
    """
    Search engine for small-molecule protein-ligand docking.
    Explores 6D rigid degrees of freedom + internal rotatable dihedral angles.
    """

    def __init__(
        self,
        receptor: Molecule,
        ligand: Molecule,
        config: Optional[DockingConfig] = None,
        metal_centers: Optional[List[MetalCenter]] = None,
    ):
        self.receptor = receptor
        self.ligand = ligand
        self.config = config or DockingConfig()
        self.metal_centers = metal_centers or []
        self.grid = self.config.grid

        # Initialize composite scorer
        self.scorer = CompositeScoringFunction(
            weights=self.config.scoring_weights,
            metal_config=self.config.metal_config,
            metal_centers=self.metal_centers,
        )

        # Prepare base ligand geometry
        self.base_coords = np.copy(ligand.coords)
        self.rotatable_bonds = ligand.metadata.get("rotatable_bonds", [])

    def _apply_conformation(
        self,
        translation: np.ndarray,
        rot_matrix: np.ndarray,
        torsions: np.ndarray,
    ) -> np.ndarray:
        """
        Apply internal dihedral rotations, followed by rigid body rotation and translation.
        """
        coords = np.copy(self.base_coords)

        # 1. Apply internal torsions
        for idx, (u, v, moving_indices) in enumerate(self.rotatable_bonds):
            if idx >= len(torsions):
                break
            angle = torsions[idx]
            if abs(angle) < 1e-4:
                continue

            axis = coords[v] - coords[u]
            axis_len = np.linalg.norm(axis)
            if axis_len < 1e-5:
                continue
            axis = axis / axis_len

            origin = coords[u]
            rot_m = rotation_matrix_from_axis_angle(axis, angle)

            for m_idx in moving_indices:
                rel = coords[m_idx] - origin
                coords[m_idx] = origin + np.dot(rot_m, rel)

        # 2. Rigid body rotation around ligand geometric center
        center = np.mean(coords, axis=0)
        coords = center + np.dot(coords - center, rot_matrix.T)

        # 3. Rigid body translation to target position
        curr_center = np.mean(coords, axis=0)
        coords = coords - curr_center + translation

        return coords

    def dock(self) -> DockingResult:
        """
        Execute Monte Carlo global search and local optimization across multiple runs.
        """
        start_time = time.time()
        rng = np.random.default_rng(self.config.seed)

        num_torsions = len(self.rotatable_bonds)
        exhaustiveness = max(self.config.exhaustiveness, 1)
        steps_per_run = 120 * exhaustiveness

        all_candidates: List[Tuple[float, np.ndarray, Dict[str, Any]]] = []

        # Temp molecule wrapper for scoring
        test_ligand = self.ligand.copy()

        # Center and bounds of search box
        box_center = self.grid.center
        box_half = self.grid.size / 2.0

        for run in range(exhaustiveness):
            # Initial random state inside grid box
            trans = box_center + rng.uniform(-box_half, box_half)
            rot = random_rotation_matrix(rng)
            tors = rng.uniform(-np.pi, np.pi, num_torsions) if num_torsions > 0 else np.zeros(0)

            coords = self._apply_conformation(trans, rot, tors)
            test_ligand.coords = coords
            curr_score = self.scorer.score(self.receptor, test_ligand)

            best_run_score = curr_score
            best_run_coords = np.copy(coords)

            # Simulated Annealing schedule
            temp_start = 2.0
            temp_end = 0.05

            for step in range(steps_per_run):
                frac = step / float(steps_per_run)
                temp = temp_start * ((temp_end / temp_start) ** frac)

                # Propose random perturbation
                step_scale = 1.0 - 0.7 * frac
                d_trans = rng.normal(0, 0.7 * step_scale, 3)
                angle = rng.normal(0, 0.25 * step_scale)
                axis = rng.normal(0, 1.0, 3)
                d_rot = rotation_matrix_from_axis_angle(axis, angle)

                new_trans = trans + d_trans
                # Constran to grid box
                diff = new_trans - box_center
                new_trans = box_center + np.clip(diff, -box_half, box_half)

                new_rot = np.dot(d_rot, rot)
                new_tors = np.copy(tors)
                if num_torsions > 0:
                    d_tors = rng.normal(0, 0.3 * step_scale, num_torsions)
                    new_tors = (new_tors + d_tors + np.pi) % (2 * np.pi) - np.pi

                cand_coords = self._apply_conformation(new_trans, new_rot, new_tors)
                test_ligand.coords = cand_coords
                cand_score = self.scorer.score(self.receptor, test_ligand)

                # Metropolis acceptance criterion
                delta_e = cand_score - curr_score
                if delta_e < 0 or (rng.uniform(0.0, 1.0) < math.exp(-delta_e / max(temp, 1e-4))):
                    trans = new_trans
                    rot = new_rot
                    tors = new_tors
                    curr_score = cand_score
                    coords = cand_coords

                    if cand_score < best_run_score:
                        best_run_score = cand_score
                        best_run_coords = np.copy(cand_coords)

            # Record best result from this exhaustiveness run
            test_ligand.coords = best_run_coords
            bd = self.scorer.score_breakdown(self.receptor, test_ligand)
            all_candidates.append((best_run_score, best_run_coords, bd))

        # Sort all candidates by score (lowest energy first)
        all_candidates.sort(key=lambda x: x[0])

        # Cluster poses by RMSD to avoid redundant poses
        distinct_poses: List[DockingPose] = []
        rmsd_cutoff = 1.8  # Ångströms

        for score, cand_coords, bd in all_candidates:
            # Energy range filter relative to best score
            if distinct_poses and (score - distinct_poses[0].score) > self.config.energy_range:
                continue

            # Check RMSD against existing poses
            is_duplicate = False
            for p in distinct_poses:
                diff = cand_coords - p.coords
                rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))
                if rmsd < rmsd_cutoff:
                    is_duplicate = True
                    break

            if not is_duplicate:
                pose_id = len(distinct_poses) + 1
                distinct_poses.append(
                    DockingPose(
                        pose_id=pose_id,
                        score=score,
                        coords=cand_coords,
                        score_breakdown=bd,
                    )
                )
                if len(distinct_poses) >= self.config.num_modes:
                    break

        elapsed = time.time() - start_time
        top_score = distinct_poses[0].score if distinct_poses else 0.0

        return DockingResult(
            docking_type="Protein-Ligand",
            receptor=self.receptor,
            query=self.ligand,
            poses=distinct_poses,
            top_score=top_score,
            execution_time_seconds=elapsed,
            config_summary={
                "grid_center": box_center.tolist(),
                "grid_size": self.grid.size.tolist(),
                "exhaustiveness": exhaustiveness,
                "modes_found": len(distinct_poses),
                "num_metal_centers": len(self.metal_centers),
            },
        )

