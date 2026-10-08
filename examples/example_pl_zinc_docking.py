"""
Example 1: Protein-Ligand Docking with Metal Co-factor Coordination (Zn2+)
Demonstrates loading receptor, auto-detecting zinc cofactor, preparing ligand,
running Monte Carlo global search with AutoDock Vina + Metal scoring,
and analyzing interactions (H-bonds, metal coordination bonds).
"""

import os
import numpy as np
from moldock import (
    ReceptorPreparer,
    LigandPreparer,
    GridBox,
    MetalConfig,
    DockingConfig,
    ProteinLigandEngine,
    DockingReporter,
    InteractionProfiler,
)

def run_example():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rec_path = os.path.join(base_dir, "sample_data", "receptor_zinc.pdb")
    lig_path = os.path.join(base_dir, "sample_data", "ligand_inhibitor.pdb")

    print("=== MolDock Example 1: Protein-Ligand Docking with Zn2+ Cofactor ===")
    
    # 1. Prepare Receptor and detect metal centers
    metal_config = MetalConfig(allowed_metals=["ZN", "MG", "CA"])
    rec_preparer = ReceptorPreparer(metal_config=metal_config)
    receptor, metal_centers = rec_preparer.prepare_from_pdb(rec_path)
    
    print(f"Receptor prepared: {len(receptor.atoms)} atoms")
    print(f"Metal cofactors detected: {len(metal_centers)}")
    for mc in metal_centers:
        print(f"  - {mc.properties.name} at {mc.metal_atom.coord.round(2)}, "
              f"Current coordination: {mc.coordination_number}, Available valency: {mc.available_valency}")

    # 2. Prepare Ligand (detect rotatable bonds, partial charges)
    ligand = LigandPreparer.prepare_ligand(lig_path)
    print(f"Ligand prepared: {len(ligand.atoms)} atoms, {ligand.metadata['num_rotatable_bonds']} rotatable bonds")

    # 3. Define search grid centered on the zinc active site
    metal_coords = np.array([mc.metal_atom.coord for mc in metal_centers])
    grid = GridBox.from_metal_ions(metal_coords, box_radius=8.0)
    print(f"Grid box centered at: ({grid.center_x:.2f}, {grid.center_y:.2f}, {grid.center_z:.2f})")

    # 4. Configure and run docking
    config = DockingConfig(
        grid=grid,
        metal_config=metal_config,
        num_modes=5,
        exhaustiveness=6,
        seed=123,
    )
    
    engine = ProteinLigandEngine(receptor=receptor, ligand=ligand, config=config, metal_centers=metal_centers)
    result = engine.dock()

    # 5. Display Vina-style results table
    print("\n" + DockingReporter.generate_table(result))

    # 6. Analyze intermolecular contacts for top pose
    top_pose = result.get_top_pose()
    if top_pose:
        docked_lig = top_pose.apply_to_molecule(ligand)
        profiler = InteractionProfiler()
        contacts = profiler.profile_interactions(receptor, docked_lig)
        
        print("\nIntermolecular Interactions in Top Pose:")
        for c in contacts:
            print(f"  [{c.interaction_type.upper()}] Rec: {c.receptor_res}{c.receptor_res_seq} ({c.receptor_atom_name}) "
                  f"<---> Lig: {c.query_atom_name} ({c.query_element}) | Distance: {c.distance:.2f} Å")

    print("\nDocking completed successfully.")

if __name__ == "__main__":
    run_example()
