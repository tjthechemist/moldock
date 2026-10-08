"""
Reporting, tabulating, and exporting docking outcomes and interaction profiles.
"""

import json
import csv
import os
from typing import List, Dict, Any, Optional
import numpy as np

from moldock.engines.base import DockingResult, DockingPose
from moldock.core.molecule import Molecule
from moldock.core.pdb_io import PDBIO
from moldock.analysis.rmsd import calculate_rmsd
from moldock.analysis.interactions import InteractionProfiler, InteractionContact



class DockingReporter:
    """
    Formats docking poses into Vina-style tables, CSV reports, and complex PDB structures.
    """

    @staticmethod
    def generate_table(result: DockingResult) -> str:
        """
        Generate a terminal-friendly summary table modeled after AutoDock Vina output.
        """
        lines = []
        lines.append("=" * 78)
        lines.append(f"  MOLDOCK DOCKING RESULTS: {result.docking_type}")
        lines.append("=" * 78)
        lines.append(f"  Execution Time: {result.execution_time_seconds:.2f} s")
        lines.append(f"  Total Poses Found: {len(result.poses)}")
        lines.append("-" * 78)
        lines.append(f"{'Mode':<6} | {'Affinity (kcal/mol)':<20} | {'rmsd l.b.':<10} | {'rmsd u.b.':<10} | {'Metal Score':<12}")
        lines.append("-" * 78)

        if not result.poses:
            lines.append("  No valid docking poses found.")
            lines.append("=" * 78)
            return "\n".join(lines)

        ref_coords = result.poses[0].coords

        for pose in result.poses:
            # RMSD relative to top pose
            diff = pose.coords - ref_coords
            rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1)))) if pose.pose_id > 1 else 0.0

            metal_sc = pose.score_breakdown.get("metal_score", 0.0)
            metal_str = f"{metal_sc:8.2f}" if abs(metal_sc) > 1e-4 else "    N/A   "

            lines.append(
                f"{pose.pose_id:<6} | {pose.score:<20.2f} | {rmsd:<10.2f} | {rmsd:<10.2f} | {metal_str:<12}"
            )

        lines.append("=" * 78)
        return "\n".join(lines)

    @staticmethod
    def export_csv(result: DockingResult, csv_path: str) -> None:
        """Export scores and breakdown to CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        ref_coords = result.poses[0].coords if result.poses else None

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Mode",
                "Total_Affinity_kcal_mol",
                "Vina_Score",
                "Metal_Score",
                "Electrostatic_Score",
                "RMSD_from_Top",
            ])

            for pose in result.poses:
                rmsd = 0.0
                if ref_coords is not None and pose.pose_id > 1:
                    diff = pose.coords - ref_coords
                    rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))

                writer.writerow([
                    pose.pose_id,
                    round(pose.score, 3),
                    round(pose.score_breakdown.get("vina_score", pose.score), 3),
                    round(pose.score_breakdown.get("metal_score", 0.0), 3),
                    round(pose.score_breakdown.get("electrostatic_score", 0.0), 3),
                    round(rmsd, 3),
                ])

    @staticmethod
    def save_complex_pdb(
        receptor: Molecule,
        pose: DockingPose,
        query_template: Molecule,
        output_pdb_path: str,
    ) -> None:
        """
        Merge receptor and docked pose into a single complex PDB file.
        Assigns distinct chain ID to query if same as receptor.
        """
        complex_mol = receptor.copy()
        docked_query = pose.apply_to_molecule(query_template)

        rec_chains = set(complex_mol.get_chains())
        new_chain = "L"
        for c in ["L", "B", "C", "D", "Z", "X", "Y"]:
            if c not in rec_chains:
                new_chain = c
                break

        for a in docked_query.atoms:
            a.chain_id = new_chain
            complex_mol.add_atom(a)

        PDBIO.write_pdb(complex_mol, output_pdb_path)
