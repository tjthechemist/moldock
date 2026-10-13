"""
Unified Command-Line Interface (CLI) for MolDock:
Protein-Ligand, Protein-Protein docking, Metal Cofactor detection, and Scoring.
"""

import argparse
import sys
import os
import numpy as np

from moldock.core.config import GridBox, MetalConfig, DockingConfig, ScoringWeights
from moldock.core.pdb_io import PDBIO, PDBQTIO
from moldock.core.sdf_io import SDFIO
from moldock.preparation.receptor_prep import ReceptorPreparer
from moldock.preparation.ligand_prep import LigandPreparer
from moldock.preparation.partner_prep import PartnerPreparer
from moldock.engines.protein_ligand import ProteinLigandEngine
from moldock.engines.protein_protein import ProteinProteinEngine
from moldock.analysis.reporter import DockingReporter
from moldock.analysis.interactions import InteractionProfiler
from moldock.analysis.visualizer import InteractionVisualizer
from moldock.scoring.composite_scoring import CompositeScoringFunction
from moldock.screening.virtual_screener import VirtualScreener
from moldock.screening.screening_report import ScreeningReporter


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moldock",
        description="MolDock: Molecular Docking in Python with AutoDock Vina Scoring & Metal Co-factor Coordination",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # 1. pl-dock (Protein-Ligand)
    pl_parser = subparsers.add_parser("pl-dock", help="Protein-Ligand docking")
    pl_parser.add_argument("--receptor", "-r", required=True, help="Receptor PDB file")
    pl_parser.add_argument("--ligand", "-l", required=True, help="Ligand file (.sdf, .mol, .pdb, .pdbqt)")
    pl_parser.add_argument("--out", "-o", default="docked_ligand.pdbqt", help="Output poses file (.pdbqt or .pdb)")
    pl_parser.add_argument("--complex-out", default=None, help="Save top docked pose merged with receptor (.pdb)")
    pl_parser.add_argument("--csv-out", default=None, help="Save score breakdown table to CSV")
    pl_parser.add_argument("--plot-out", default=None, help="Save interaction diagram to PNG image")


    # Grid box parameters
    pl_parser.add_argument("--center-x", type=float, default=None, help="Grid box center X")
    pl_parser.add_argument("--center-y", type=float, default=None, help="Grid box center Y")
    pl_parser.add_argument("--center-z", type=float, default=None, help="Grid box center Z")
    pl_parser.add_argument("--size-x", type=float, default=20.0, help="Grid box size X (Å)")
    pl_parser.add_argument("--size-y", type=float, default=20.0, help="Grid box size Y (Å)")
    pl_parser.add_argument("--size-z", type=float, default=20.0, help="Grid box size Z (Å)")
    pl_parser.add_argument("--center-on-metal", action="store_true", help="Auto-center grid box on detected metal cofactor(s)")

    # Search and metal parameters
    pl_parser.add_argument("--exhaustiveness", "-e", type=int, default=8, help="Exhaustiveness of search (default: 8)")
    pl_parser.add_argument("--num-modes", "-n", type=int, default=9, help="Max number of binding modes (default: 9)")
    pl_parser.add_argument("--metals", default="ZN,MG,CA,MN,FE,CU,CO,NI", help="Comma-separated allowed metal cofactors")
    pl_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # 2. pp-dock (Protein-Protein)
    pp_parser = subparsers.add_parser("pp-dock", help="Protein-Protein docking from PDB")
    pp_parser.add_argument("--receptor", "-r", required=True, help="Receptor PDB file")
    pp_parser.add_argument("--partner", "-p", required=True, help="Partner protein PDB file")
    pp_parser.add_argument("--out", "-o", default="docked_partner.pdb", help="Output poses file (.pdb)")
    pp_parser.add_argument("--complex-out", default=None, help="Save top docked complex merged into single PDB")
    pp_parser.add_argument("--csv-out", default=None, help="Save score breakdown table to CSV")
    pp_parser.add_argument("--plot-out", default=None, help="Save interface interaction diagram to PNG image")

    # Grid box parameters for interface
    pp_parser.add_argument("--center-x", type=float, default=None, help="Interface search center X")
    pp_parser.add_argument("--center-y", type=float, default=None, help="Interface search center Y")
    pp_parser.add_argument("--center-z", type=float, default=None, help="Interface search center Z")
    pp_parser.add_argument("--size-x", type=float, default=30.0, help="Search size X (Å)")
    pp_parser.add_argument("--size-y", type=float, default=30.0, help="Search size Y (Å)")
    pp_parser.add_argument("--size-z", type=float, default=30.0, help="Search size Z (Å)")
    pp_parser.add_argument("--center-on-metal", action="store_true", help="Auto-center grid box on detected metal cofactor(s)")

    pp_parser.add_argument("--rotations", type=int, default=120, help="Number of rotational orientations to sample")
    pp_parser.add_argument("--translations", type=int, default=40, help="Number of translational positions to probe")
    pp_parser.add_argument("--exhaustiveness", "-e", type=int, default=8, help="Exhaustiveness (default: 8)")
    pp_parser.add_argument("--num-modes", "-n", type=int, default=9, help="Max number of binding modes (default: 9)")
    pp_parser.add_argument("--metals", default="ZN,MG,CA,MN,FE,CU,CO,NI", help="Comma-separated allowed metal cofactors")
    pp_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # 3. detect-metals
    dm_parser = subparsers.add_parser("detect-metals", help="Scan and inspect metal cofactors in PDB")
    dm_parser.add_argument("--receptor", "-r", required=True, help="Receptor PDB file")
    dm_parser.add_argument("--metals", default="ZN,MG,CA,MN,FE,CU,CO,NI,NA,K", help="Allowed metal elements")

    # 4. score
    sc_parser = subparsers.add_parser("score", help="Score an existing complex using AutoDock Vina + Metal scoring")
    sc_parser.add_argument("--receptor", "-r", required=True, help="Receptor PDB file")
    sc_parser.add_argument("--query", "-q", required=True, help="Ligand or Partner file (.sdf, .pdb, .pdbqt)")

    # 5. visualize
    vis_parser = subparsers.add_parser("visualize", help="Render PNG interaction diagram of a complex")
    vis_parser.add_argument("--receptor", "-r", required=True, help="Receptor PDB file")
    vis_parser.add_argument("--query", "-q", required=True, help="Docked Ligand or Partner file (.sdf, .pdb, .pdbqt)")
    vis_parser.add_argument("--out", "-o", default="interactions.png", help="Output PNG file path")
    vis_parser.add_argument("--mode", choices=["2d", "3d"], default="2d", help="Visualization mode (2d network or 3d pocket)")

    # 6. screen (Virtual Screening)
    vs_parser = subparsers.add_parser(
        "screen",
        help="Virtual screening: dock a library of ligands against a single receptor",
    )
    vs_parser.add_argument("--receptor", "-r", required=True, help="Receptor PDB file")
    vs_parser.add_argument(
        "--library", "-l", required=True,
        help="Ligand library: path to multi-molecule .sdf, a directory of ligand files, "
             "or a single .sdf/.pdb/.pdbqt file",
    )
    vs_parser.add_argument("--out-dir", "-o", default="screening_results",
                           help="Output directory for results (default: screening_results/)")
    vs_parser.add_argument("--top-n", type=int, default=20,
                           help="Number of top hits to include in reports (default: 20)")
    vs_parser.add_argument("--score-cutoff", type=float, default=None,
                           help="Discard hits with score above this value (kcal/mol)")
    vs_parser.add_argument("--workers", type=int, default=1,
                           help="Number of parallel docking workers (default: 1)")
    vs_parser.add_argument("--exhaustiveness", type=int, default=8,
                           help="Exhaustiveness per ligand (default: 8)")
    vs_parser.add_argument("--num-modes", type=int, default=9,
                           help="Max docking poses per ligand (default: 9)")
    vs_parser.add_argument("--center-x", type=float, default=None, help="Grid box center X")
    vs_parser.add_argument("--center-y", type=float, default=None, help="Grid box center Y")
    vs_parser.add_argument("--center-z", type=float, default=None, help="Grid box center Z")
    vs_parser.add_argument("--size-x", type=float, default=20.0, help="Grid box size X (Å)")
    vs_parser.add_argument("--size-y", type=float, default=20.0, help="Grid box size Y (Å)")
    vs_parser.add_argument("--size-z", type=float, default=20.0, help="Grid box size Z (Å)")
    vs_parser.add_argument("--center-on-metal", action="store_true",
                           help="Auto-center grid on detected metal cofactors")
    vs_parser.add_argument("--metals", default="ZN,MG,CA,MN,FE,CU,CO,NI,NA,K",
                           help="Allowed metal elements")
    vs_parser.add_argument("--no-dashboard", action="store_true",
                           help="Skip rendering the PNG dashboard (faster)")
    vs_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser



def run_detect_metals(args) -> int:
    metals_list = [m.strip().upper() for m in args.metals.split(",") if m.strip()]
    m_cfg = MetalConfig(allowed_metals=metals_list)
    prep = ReceptorPreparer(metal_config=m_cfg)
    _, centers = prep.prepare_from_pdb(args.receptor)

    print("=" * 70)
    print(f"  DETECTED METAL COFACTORS IN: {args.receptor}")
    print("=" * 70)
    if not centers:
        print("  No metal ions found.")
        print("=" * 70)
        return 0

    for idx, c in enumerate(centers, start=1):
        m = c.metal_atom
        print(f"[{idx}] Metal: {c.properties.name} ({c.properties.element}), Atom ID: {m.index}, Chain: {m.chain_id}, Res: {m.res_name} {m.res_seq}")
        print(f"    Coordinates: ({m.coord[0]:.2f}, {m.coord[1]:.2f}, {m.coord[2]:.2f}) Å")
        print(f"    Oxidation state: +{c.properties.oxidation_state}, Preferred coordination: {c.properties.preferred_coord_num}")
        print(f"    Current coordinating protein residues: {c.coordination_number}")
        for cr in c.coordinating_atoms:
            print(f"      - {cr.res_name}{cr.res_seq} ({cr.atom_name}) at {cr.distance:.2f} Å")
        print(f"    Available coordination slots for ligand/partner: {c.available_valency}")
        print("-" * 70)

    print("=" * 70)
    return 0


def run_pl_dock(args) -> int:
    metals_list = [m.strip().upper() for m in args.metals.split(",") if m.strip()]
    m_cfg = MetalConfig(allowed_metals=metals_list)

    print(f">> Preparing receptor: {args.receptor}")
    prep = ReceptorPreparer(metal_config=m_cfg)
    rec, centers = prep.prepare_from_pdb(args.receptor)
    print(f"   Receptor atoms: {len(rec.atoms)}, Metal cofactors detected: {len(centers)}")

    print(f">> Preparing ligand: {args.ligand}")
    lig = LigandPreparer.prepare_ligand(args.ligand)
    print(f"   Ligand atoms: {len(lig.atoms)}, Rotatable bonds: {lig.metadata.get('num_rotatable_bonds', 0)}")

    # Define grid box
    if args.center_on_metal and centers:
        metal_coords = np.array([c.metal_atom.coord for c in centers])
        grid = GridBox.from_metal_ions(metal_coords, box_radius=args.size_x / 2.0)
        print(f"   Auto-centered grid on metal cofactors: center=({grid.center_x:.2f}, {grid.center_y:.2f}, {grid.center_z:.2f})")
    elif args.center_x is not None and args.center_y is not None and args.center_z is not None:
        grid = GridBox(
            center_x=args.center_x, center_y=args.center_y, center_z=args.center_z,
            size_x=args.size_x, size_y=args.size_y, size_z=args.size_z
        )
    else:
        # Default: center on receptor or ligand
        if centers:
            metal_coords = np.array([c.metal_atom.coord for c in centers])
            grid = GridBox.from_metal_ions(metal_coords, box_radius=args.size_x / 2.0)
            print(f"   Centered grid on detected metal cofactors: ({grid.center_x:.2f}, {grid.center_y:.2f}, {grid.center_z:.2f})")
        else:
            center = lig.center_of_mass
            grid = GridBox(center_x=center[0], center_y=center[1], center_z=center[2], size_x=args.size_x, size_y=args.size_y, size_z=args.size_z)
            print(f"   Centered grid on ligand center: ({grid.center_x:.2f}, {grid.center_y:.2f}, {grid.center_z:.2f})")

    cfg = DockingConfig(
        grid=grid,
        metal_config=m_cfg,
        num_modes=args.num_modes,
        exhaustiveness=args.exhaustiveness,
        seed=args.seed,
    )

    print(">> Executing Protein-Ligand docking...")
    engine = ProteinLigandEngine(receptor=rec, ligand=lig, config=cfg, metal_centers=centers)
    res = engine.dock()

    # Output table
    print(DockingReporter.generate_table(res))

    # Save poses
    ext = os.path.splitext(args.out)[1].lower()
    if ext == ".pdb":
        res.export_poses_pdb(args.out)
    else:
        res.export_poses_pdbqt(args.out)
    print(f">> Poses saved to: {args.out}")

    if args.complex_out and res.poses:
        DockingReporter.save_complex_pdb(rec, res.poses[0], lig, args.complex_out)
        print(f">> Top pose complex saved to: {args.complex_out}")

    if args.csv_out:
        DockingReporter.export_csv(res, args.csv_out)
        print(f">> CSV summary written to: {args.csv_out}")

    if args.plot_out and res.poses:
        top_mol = res.poses[0].apply_to_molecule(lig)
        profiler = InteractionProfiler()
        contacts = profiler.profile_interactions(rec, top_mol)
        InteractionVisualizer.render_interaction_diagram(
            receptor=rec,
            query=top_mol,
            contacts=contacts,
            output_png_path=args.plot_out,
            title="Protein-Ligand Interaction Map",
            score_info={"total_score": res.poses[0].score, "metal_score": res.poses[0].score_breakdown.get("metal_score", 0.0)},
        )
        print(f">> Interaction diagram PNG saved to: {args.plot_out}")

    return 0



def run_pp_dock(args) -> int:
    metals_list = [m.strip().upper() for m in args.metals.split(",") if m.strip()]
    m_cfg = MetalConfig(allowed_metals=metals_list)

    print(f">> Preparing receptor: {args.receptor}")
    prep = ReceptorPreparer(metal_config=m_cfg)
    rec, centers = prep.prepare_from_pdb(args.receptor)
    print(f"   Receptor atoms: {len(rec.atoms)}, Metal cofactors detected: {len(centers)}")

    print(f">> Preparing partner protein: {args.partner}")
    partner = PartnerPreparer.prepare_partner_from_pdb(args.partner)
    print(f"   Partner atoms: {len(partner.atoms)}")

    # Define interface search region
    if args.center_on_metal and centers:
        metal_coords = np.array([c.metal_atom.coord for c in centers])
        grid = GridBox.from_metal_ions(metal_coords, box_radius=args.size_x / 2.0)
    elif args.center_x is not None and args.center_y is not None and args.center_z is not None:
        grid = GridBox(
            center_x=args.center_x, center_y=args.center_y, center_z=args.center_z,
            size_x=args.size_x, size_y=args.size_y, size_z=args.size_z
        )
    else:
        # Default: center on receptor or metal
        if centers:
            metal_coords = np.array([c.metal_atom.coord for c in centers])
            grid = GridBox.from_metal_ions(metal_coords, box_radius=args.size_x / 2.0)
        else:
            center = rec.center_of_mass
            grid = GridBox(center_x=center[0], center_y=center[1], center_z=center[2], size_x=args.size_x, size_y=args.size_y, size_z=args.size_z)

    cfg = DockingConfig(
        grid=grid,
        metal_config=m_cfg,
        num_modes=args.num_modes,
        exhaustiveness=args.exhaustiveness,
        rigid_body_rotations=args.rotations,
        rigid_body_translations=args.translations,
        seed=args.seed,
    )

    print(">> Executing Protein-Protein docking with Vina interface scoring...")
    engine = ProteinProteinEngine(receptor=rec, partner=partner, config=cfg, metal_centers=centers)
    res = engine.dock()

    print(DockingReporter.generate_table(res))

    res.export_poses_pdb(args.out)
    print(f">> Poses saved to: {args.out}")

    if args.complex_out and res.poses:
        DockingReporter.save_complex_pdb(rec, res.poses[0], partner, args.complex_out)
        print(f">> Docked complex saved to: {args.complex_out}")

    if args.csv_out:
        DockingReporter.export_csv(res, args.csv_out)
        print(f">> CSV summary written to: {args.csv_out}")

    if args.plot_out and res.poses:
        top_mol = res.poses[0].apply_to_molecule(partner)
        profiler = InteractionProfiler()
        contacts = profiler.profile_interactions(rec, top_mol)
        InteractionVisualizer.render_interaction_diagram(
            receptor=rec,
            query=top_mol,
            contacts=contacts,
            output_png_path=args.plot_out,
            title="Protein-Protein Interface Interaction Map",
            score_info={"total_score": res.poses[0].score, "metal_score": res.poses[0].score_breakdown.get("metal_score", 0.0)},
        )
        print(f">> Interface interaction diagram PNG saved to: {args.plot_out}")

    return 0


def run_score(args) -> int:
    rec_prep = ReceptorPreparer()
    rec, centers = rec_prep.prepare_from_pdb(args.receptor)

    ext = os.path.splitext(args.query)[1].lower()
    if ext == ".pdbqt":
        query = PDBQTIO.read_pdbqt(args.query)
    elif ext in (".sdf", ".mol"):
        query = SDFIO.read_sdf(args.query)[0]
    else:
        query = PDBIO.read_pdb(args.query)

    scorer = CompositeScoringFunction(metal_centers=centers)
    bd = scorer.score_breakdown(rec, query)

    print("=" * 60)
    print("  COMPLEX SCORING BREAKDOWN")
    print("=" * 60)
    print(f"  Total Affinity:         {bd['total_score']:.3f} kcal/mol")
    print(f"  AutoDock Vina Score:    {bd['vina_score']:.3f} kcal/mol")
    print(f"    - Gauss1:             {bd['vina_breakdown']['gauss1']:.3f}")
    print(f"    - Gauss2:             {bd['vina_breakdown']['gauss2']:.3f}")
    print(f"    - Repulsion:          {bd['vina_breakdown']['repulsion']:.3f}")
    print(f"    - Hydrophobic:        {bd['vina_breakdown']['hydrophobic']:.3f}")
    print(f"    - Hydrogen Bonding:   {bd['vina_breakdown']['hbond']:.3f}")
    print(f"  Metal Ion Score:        {bd['metal_score']:.3f} kcal/mol")
    print(f"    - Coordination Well:  {bd['metal_coordination_energy']:.3f}")
    print(f"    - Steric Penalty:     {bd['metal_steric_penalty']:.3f}")
    print(f"    - Polarization:       {bd['metal_polarization']:.3f}")
    print(f"  Electrostatic Score:    {bd['electrostatic_score']:.3f} kcal/mol")
    print("=" * 60)
    return 0


def run_visualize(args) -> int:
    rec_prep = ReceptorPreparer()
    rec, centers = rec_prep.prepare_from_pdb(args.receptor)

    ext = os.path.splitext(args.query)[1].lower()
    if ext == ".pdbqt":
        query = PDBQTIO.read_pdbqt(args.query)
    elif ext in (".sdf", ".mol"):
        query = SDFIO.read_sdf(args.query)[0]
    else:
        query = PDBIO.read_pdb(args.query)

    profiler = InteractionProfiler()
    contacts = profiler.profile_interactions(rec, query)

    scorer = CompositeScoringFunction(metal_centers=centers)
    bd = scorer.score_breakdown(rec, query)

    if args.mode == "3d":
        InteractionVisualizer.render_3d_pocket_projection(
            receptor=rec,
            query=query,
            contacts=contacts,
            metal_centers=centers,
            output_png_path=args.out,
            title="3D Binding Pocket & Metal Coordination",
        )
    else:
        InteractionVisualizer.render_interaction_diagram(
            receptor=rec,
            query=query,
            contacts=contacts,
            output_png_path=args.out,
            title="Molecular Docking Interaction Map",
            score_info=bd,
        )

    print(f">> Interaction visualization saved to: {args.out} (Found {len(contacts)} contacts)")
    return 0


def run_screen(args) -> int:
    """Execute a virtual screening campaign."""
    import os

    metals_list = [m.strip().upper() for m in args.metals.split(",") if m.strip()]
    m_cfg = MetalConfig(allowed_metals=metals_list)

    # Build grid box
    from moldock.preparation.receptor_prep import ReceptorPreparer as _RP
    _rec_tmp, _centers_tmp = _RP(metal_config=m_cfg).prepare_from_pdb(args.receptor)

    if args.center_on_metal and _centers_tmp:
        metal_coords = np.array([c.metal_atom.coord for c in _centers_tmp])
        grid = GridBox.from_metal_ions(metal_coords, box_radius=args.size_x / 2.0)
    elif args.center_x is not None:
        grid = GridBox(
            center_x=args.center_x, center_y=args.center_y, center_z=args.center_z,
            size_x=args.size_x, size_y=args.size_y, size_z=args.size_z,
        )
    elif _centers_tmp:
        metal_coords = np.array([c.metal_atom.coord for c in _centers_tmp])
        grid = GridBox.from_metal_ions(metal_coords, box_radius=args.size_x / 2.0)
    else:
        ctr = _rec_tmp.center_of_mass
        grid = GridBox(center_x=ctr[0], center_y=ctr[1], center_z=ctr[2],
                       size_x=args.size_x, size_y=args.size_y, size_z=args.size_z)

    cfg = DockingConfig(
        grid=grid,
        metal_config=m_cfg,
        exhaustiveness=args.exhaustiveness,
        num_modes=args.num_modes,
        seed=args.seed,
    )

    screener = VirtualScreener(
        receptor_pdb=args.receptor,
        config=cfg,
        metal_config=m_cfg,
        score_cutoff=args.score_cutoff,
        max_workers=args.workers,
        use_threads=True,
    )

    campaign = screener.screen(args.library)

    # Output directory
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Print table
    print(ScreeningReporter.generate_table(campaign, top_n=args.top_n, score_cutoff=args.score_cutoff))

    # CSV
    csv_path = os.path.join(out_dir, "screening_results.csv")
    ScreeningReporter.export_csv(campaign, csv_path)
    print(f">> CSV results written to: {csv_path}")

    # Top-hits SDF
    sdf_path = os.path.join(out_dir, "top_hits.sdf")
    ScreeningReporter.export_hits_sdf(campaign, sdf_path, top_n=args.top_n, score_cutoff=args.score_cutoff)
    print(f">> Top-{args.top_n} docked poses exported to SDF: {sdf_path}")

    # PNG dashboard
    if not args.no_dashboard:
        dashboard_path = os.path.join(out_dir, "dashboard.png")
        ScreeningReporter.render_dashboard(
            campaign,
            dashboard_path,
            top_n=args.top_n,
            score_cutoff=args.score_cutoff,
            title=f"Virtual Screening — {os.path.basename(args.receptor)}",
        )
        print(f">> Dashboard PNG saved to: {dashboard_path}")

    return 0


def main():

    parser = setup_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if args.command == "pl-dock":
        sys.exit(run_pl_dock(args))
    elif args.command == "pp-dock":
        sys.exit(run_pp_dock(args))
    elif args.command == "detect-metals":
        sys.exit(run_detect_metals(args))
    elif args.command == "score":
        sys.exit(run_score(args))
    elif args.command == "visualize":
        sys.exit(run_visualize(args))
    elif args.command == "screen":
        sys.exit(run_screen(args))
    else:
        parser.print_help()
        sys.exit(1)



if __name__ == "__main__":
    main()

