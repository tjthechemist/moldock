"""
Protein-Protein docking engine (PPD).
Performs 6D rigid-body exploration (Euler rotational sampling + surface translational grid)
evaluated using AutoDock Vina interface scoring and metal co-factor coordination.
"""

import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from moldock.core.atom import Atom
from moldock.core.molecule import Molecule
from moldock.core.config import DockingConfig, GridBox
from moldock.metals.detector import MetalCenter
from moldock.scoring.composite_scoring import CompositeScoringFunction
from moldock.engines.base import BaseDockingEngine, DockingPose, DockingResult
from moldock.engines.protein_ligand import rotation_matrix_from_axis_angle, random_rotation_matrix


def generate_fibonacci_sphere_rotations(num_orientations: int = 120) -> List[np.ndarray]:
    """
    Generate approximately uniform 3D rotation matrices distributed over SO(3).
    Combines Fibonacci spiral directions with in-plane roll angles.
    """
    rotations = []
    phi = (1 + np.sqrt(5)) / 2.0  # Golden ratio
    num_rolls = 3

    n_dirs = max(1, num_orientations // num_rolls)
    for i in range(n_dirs):
        y = 1.0 - (i / float(n_dirs - 1)) * 2.0 if n_dirs > 1 else 0.0
        radius = np.sqrt(max(0.0, 1.0 - y * y))
        theta = 2.0 * np.pi * i / phi

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        axis = np.array([x, y, z], dtype=np.float64)

        for r_step in range(num_rolls):
            roll_angle = 2.0 * np.pi * r_step / num_rolls
            rot_mat = rotation_matrix_from_axis_angle(axis, roll_angle)
            rotations.append(rot_mat)

    return rotations


class ProteinProteinEngine(BaseDockingEngine):
    """
    Macromolecular docking engine for Receptor-Partner protein complexes.
    Performs global surface sampling and interface refinement with AutoDock Vina scoring.
    """

    def __init__(
        self,
        receptor: Molecule,
        partner: Molecule,
        config: Optional[DockingConfig] = None,
        metal_centers: Optional[List[MetalCenter]] = None,
    ):
        self.receptor = receptor
        self.partner = partner
        self.config = config or DockingConfig()
        self.metal_centers = metal_centers or []
        self.grid = self.config.grid

        # Interface scorer using Vina empirical potential + metal coordination
        self.scorer = CompositeScoringFunction(
            weights=self.config.scoring_weights,
            metal_config=self.config.metal_config,
            metal_centers=self.metal_centers,
        )

        # Pre-filter heavy atoms for fast interface scoring
        self.receptor_heavy = receptor.get_heavy_atoms()
        self.partner_heavy = partner.get_heavy_atoms()

        # Cache partner base coordinates centered at origin
        self.partner_base_coords = np.copy(partner.coords)
        self.partner_origin_center = np.mean(self.partner_base_coords, axis=0)
        self.partner_centered_coords = self.partner_base_coords - self.partner_origin_center

    def _clash_and_contact_check(
        self,
        receptor_coords: np.ndarray,
        partner_coords: np.ndarray,
        clash_dist: float = 2.0,
        contact_dist: float = 8.0,
    ) -> Tuple[bool, bool, int]:
        """
        Quick check for interface quality:
        - True if unacceptably severe steric clash (e.g. > 15 pairs < clash_dist).
        - True if in contact (at least some atoms within contact_dist).
        """
        # Fast bounding box check
        r_min, r_max = np.min(receptor_coords, axis=0), np.max(receptor_coords, axis=0)
        p_min, p_max = np.min(partner_coords, axis=0), np.max(partner_coords, axis=0)

        # If bounding boxes are completely separated beyond contact_dist, return no contact
        if (
            np.any(p_min > (r_max + contact_dist)) or
            np.any(p_max < (r_min - contact_dist))
        ):
            return False, False, 0

        # Subsample for very fast sterical triage if proteins are large
        sub_r = receptor_coords[::3] if len(receptor_coords) > 200 else receptor_coords
        sub_p = partner_coords[::3] if len(partner_coords) > 200 else partner_coords

        diff = sub_r[:, np.newaxis, :] - sub_p[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)

        clashes = int(np.sum(dist_sq < (clash_dist * clash_dist)))
        contacts = int(np.sum(dist_sq < (contact_dist * contact_dist)))

        is_clashing = clashes > 12  # Severe steric core clash
        has_contact = contacts >= 3  # Meaningful interface contact

        return is_clashing, has_contact, contacts

    def dock(self) -> DockingResult:
        """
        Execute 6D rigid body orientation grid search and interface optimization.
        """
        start_time = time.time()
        rng = np.random.default_rng(self.config.seed)

        num_rotations = max(30, self.config.rigid_body_rotations)
        orientations = generate_fibonacci_sphere_rotations(num_rotations)

        r_coords = np.array([a.coord for a in self.receptor_heavy])
        r_center = np.mean(r_coords, axis=0)

        # Target center from user grid or receptor active site
        target_center = self.grid.center if np.any(self.grid.size > 0) else r_center
        box_half = self.grid.size / 2.0

        candidates: List[Tuple[float, np.ndarray, Dict[str, Any]]] = []
        test_partner = self.partner.copy()

        # Translational positions to probe around the search zone
        num_trans = max(20, self.config.rigid_body_translations)
        probe_translations = [target_center]
        for _ in range(num_trans - 1):
            offset = rng.uniform(-box_half, box_half)
            probe_translations.append(target_center + offset)

        # 1. Broad 6D Exploration Phase
        for rot_mat in orientations:
            rotated_partner_coords = np.dot(self.partner_centered_coords, rot_mat.T)

            for trans_pt in probe_translations:
                cand_coords = rotated_partner_coords + trans_pt

                # Clash & contact filter
                is_clash, has_contact, contact_count = self._clash_and_contact_check(
                    r_coords, cand_coords
                )
                if is_clash or not has_contact:
                    continue

                test_partner.coords = cand_coords
                score = self.scorer.score(self.receptor, test_partner)

                # Filter out clearly unfavorable energy
                if score < 50.0:
                    candidates.append((score, np.copy(cand_coords), rot_mat, trans_pt))

        # Sort candidates by initial score
        candidates.sort(key=lambda x: x[0])

        # 2. Local Interface Refinement Phase for top candidates
        top_candidates = candidates[:max(15, self.config.exhaustiveness * 2)]
        refined_poses: List[DockingPose] = []

        for initial_score, init_coords, rot_mat, trans_pt in top_candidates:
            best_ref_score = initial_score
            best_ref_coords = np.copy(init_coords)
            curr_rot = np.copy(rot_mat)
            curr_trans = np.copy(trans_pt)

            # Local Nelder-Mead / random walk perturbation to settle interface
            for step in range(30):
                d_trans = rng.normal(0, 0.4, 3)
                angle = rng.normal(0, 0.08)
                axis = rng.normal(0, 1.0, 3)
                d_rot = rotation_matrix_from_axis_angle(axis, angle)

                trial_rot = np.dot(d_rot, curr_rot)
                trial_trans = curr_trans + d_trans
                trial_coords = np.dot(self.partner_centered_coords, trial_rot.T) + trial_trans

                is_clash, has_contact, _ = self._clash_and_contact_check(r_coords, trial_coords)
                if is_clash or not has_contact:
                    continue

                test_partner.coords = trial_coords
                cand_score = self.scorer.score(self.receptor, test_partner)

                if cand_score < best_ref_score:
                    best_ref_score = cand_score
                    best_ref_coords = np.copy(trial_coords)
                    curr_rot = trial_rot
                    curr_trans = trial_trans

            test_partner.coords = best_ref_coords
            bd = self.scorer.score_breakdown(self.receptor, test_partner)
            bd["interface_contacts"] = contact_count

            # Check for redundancy against existing poses
            is_dup = False
            for p in refined_poses:
                diff = best_ref_coords - p.coords
                rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))
                if rmsd < 3.5:  # 3.5 Å RMSD cluster cutoff for proteins
                    is_dup = True
                    break

            if not is_dup:
                pose_id = len(refined_poses) + 1
                refined_poses.append(
                    DockingPose(
                        pose_id=pose_id,
                        score=best_ref_score,
                        coords=best_ref_coords,
                        score_breakdown=bd,
                    )
                )
                if len(refined_poses) >= self.config.num_modes:
                    break

        # If no poses passed filters (e.g. very distant box), evaluate at target center
        if not refined_poses:
            test_partner.coords = self.partner_centered_coords + target_center
            sc = self.scorer.score(self.receptor, test_partner)
            bd = self.scorer.score_breakdown(self.receptor, test_partner)
            refined_poses.append(
                DockingPose(
                    pose_id=1,
                    score=sc,
                    coords=np.copy(test_partner.coords),
                    score_breakdown=bd,
                )
            )

        refined_poses.sort(key=lambda x: x.score)
        for idx, pose in enumerate(refined_poses, start=1):
            pose.pose_id = idx

        elapsed = time.time() - start_time


        return DockingResult(
            docking_type="Protein-Protein",
            receptor=self.receptor,
            query=self.partner,
            poses=refined_poses,
            top_score=refined_poses[0].score,
            execution_time_seconds=elapsed,
            config_summary={
                "orientations_sampled": len(orientations),
                "translations_sampled": len(probe_translations),
                "modes_found": len(refined_poses),
                "num_metal_centers": len(self.metal_centers),
            },
        )
