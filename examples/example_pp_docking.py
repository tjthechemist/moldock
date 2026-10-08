"""
Example 2: Protein-Protein Docking from PDB Files with AutoDock Vina Interface Scoring
Demonstrates loading two macromolecular proteins (Receptor and Partner),
performing 6D rigid-body orientation and surface exploration,
and evaluating poses with Vina empirical scoring.
"""

import os
from moldock import (
    ReceptorPreparer,
    PartnerPreparer,
    GridBox,
    MetalConfig,
    DockingConfig,
    ProteinProteinEngine,
    DockingReporter,
    InteractionProfiler,
)

def run_example():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rec_path = os.path.join(base_dir, "sample_data", "receptor_zinc.pdb")
    partner_path = os.path.join(base_dir, "sample_data", "protein_partner.pdb")

    print("=== MolDock Example 2: Protein-Protein Docking from PDB Files ===")

    # 1. Prepare Receptor
    metal_config = MetalConfig(allowed_metals=["ZN"])
    rec_prep = ReceptorPreparer(metal_config=metal_config)
    receptor, metal_centers = rec_prep.prepare_from_pdb(rec_path)
    print(f"Receptor: {len(receptor.atoms)} atoms, {len(metal_centers)} metal center(s)")

    # 2. Prepare Partner protein
    partner = PartnerPreparer.prepare_partner_from_pdb(partner_path)
    print(f"Partner: {len(partner.atoms)} atoms")

    # 3. Configure interface search space around receptor
    r_center = receptor.center_of_mass
    grid = GridBox(
        center_x=r_center[0], center_y=r_center[1], center_z=r_center[2],
        size_x=25.0, size_y=25.0, size_z=25.0
    )

    config = DockingConfig(
        grid=grid,
        metal_config=metal_config,
        num_modes=5,
        rigid_body_rotations=60,
        rigid_body_translations=30,
        exhaustiveness=4,
        seed=42,
    )

    # 4. Run Protein-Protein docking
    print("Running 6D rigid-body exploration & Vina interface scoring...")
    engine = ProteinProteinEngine(receptor=receptor, partner=partner, config=config, metal_centers=metal_centers)
    result = engine.dock()

    # 5. Display results table
    print("\n" + DockingReporter.generate_table(result))

    # 6. Interaction profiling
    top_pose = result.get_top_pose()
    if top_pose:
        docked_partner = top_pose.apply_to_molecule(partner)
        profiler = InteractionProfiler()
        contacts = profiler.profile_interactions(receptor, docked_partner)
        print(f"\nInterface Contacts in Top Pose: {len(contacts)} contacts found")
        for c in contacts[:8]:
            print(f"  [{c.interaction_type.upper()}] Rec: {c.receptor_res}{c.receptor_res_seq}:{c.receptor_atom_name} "
                  f"<---> Partner: {c.query_atom_name} | Dist: {c.distance:.2f} Å")

    print("\nProtein-Protein docking completed successfully.")

if __name__ == "__main__":
    run_example()
