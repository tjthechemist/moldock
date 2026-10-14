# MolDock: Python Molecular Docking & Virtual Screening Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

**MolDock** is a modular, high-performance molecular docking and virtual screening framework implemented in pure Python and NumPy. It requires **zero compiled external binaries** and provides:

1. **Protein-Ligand Docking**: Global Monte Carlo Simulated Annealing with torsional flexibility.
2. **Virtual Screening**: High-throughput multi-ligand library screening (multi-molecule `.sdf`, directories) with multi-threading, ranking, and publication-ready dashboard generation.
3. **Protein-Protein Docking (PPD)**: Macromolecular rigid-body $SO(3)$ surface search directly from PDB structures.
4. **AutoDock Vina Empirical Scoring**: Native vectorized implementation of Vina scoring potentials (Gauss1, Gauss2, Repulsion, Hydrophobic, Hydrogen bonding, and torsion count penalty).
5. **Multi-Metal Co-factor Coordination**: Explicit parameterization and scoring for catalytic and structural metal ions ($\text{Zn}^{2+}, \text{Mg}^{2+}, \text{Ca}^{2+}, \text{Mn}^{2+}, \text{Fe}^{2+/3+}, \text{Cu}^{2+}, \text{Co}^{2+}, \text{Ni}^{2+}$, etc.) using coordination wells and 12-6-4 polarization potentials.
6. **2D & 3D Visual Analysis**: Automatic PNG generation featuring **complete 2D ligand chemical structures** with highlighted contact sites, distance badges, and 3D orthographic active-site pocket projections.

---

## Key Features

- **Zero Heavy Binary Dependencies**: Pure Python and NumPy; runs out of the box on Linux, macOS, and Windows without compiling C/C++ binaries.
- **Full SDF & PDBQT Support**: Native reader and writer for MDL `.sdf` / `.mol` (V2000 & V3000) with formal charges, bond orders, and property tags, alongside standard `.pdb` and `.pdbqt` formats.
- **Complete Ligand 2D Depictions**: Unlike tools that show only disconnected contacting atoms, MolDock's 2D interaction visualizer renders the full chemical graph (all atoms, bonds, aromatic rings) and rings the interacting atoms in gold.
- **High-Throughput Virtual Screening**: Dock libraries of compounds in parallel, filter with score cutoffs, and export ranked CSVs, multi-model SDF poses, and PNG distribution dashboards.
- **Metal Co-factor Valency & Retention**: Automatically detects coordination spheres of metal cofactors in receptor PDBs, preserves catalytic ions, and computes coordination energy.
- **CLI & Python API**: Unified command-line interface (`moldock`) and clean modular Python classes.

---

## Directory Structure

```
moldock/
├── pyproject.toml              # Build & dependency metadata
├── setup.py                    # Package installer
├── requirements.txt            # Core dependencies (numpy, pillow)
├── README.md                   # Project overview & documentation
├── docs/                       # Technical architecture & math formulation
│   ├── architecture.md         # System design & module breakdown
│   └── metal_ions.md           # Metal coordination physics & 12-6-4 potentials
├── moldock/                    # Main Python package
│   ├── __init__.py             # Top-level exports
│   ├── core/                   # Core molecular representations & I/O
│   │   ├── atom.py             # Atom class, VdW radii, AutoDock atom types
│   │   ├── molecule.py         # Molecule container & 3D transformations
│   │   ├── pdb_io.py           # PDB & PDBQT parser/writer
│   │   ├── sdf_io.py           # Pure-Python SDF/MOL (V2000/V3000) parser/writer
│   │   ├── grid.py             # Search grid & spatial partitioning
│   │   └── config.py           # DockingConfig, GridBox, MetalConfig
│   ├── metals/                 # Metal co-factor subsystem
│   │   ├── parameters.py       # Metal database (Zn, Mg, Ca, Fe, Cu, etc.)
│   │   ├── detector.py         # MetalDetector & coordination sphere mapper
│   │   └── scoring.py          # 12-6-4 and coordination well scoring
│   ├── scoring/                # Scoring functions
│   │   ├── base.py             # BaseScoringFunction interface
│   │   ├── vina_scoring.py     # Pure-Python AutoDock Vina empirical terms
│   │   ├── composite_scoring.py# Vina + Metal + Electrostatics scorer
│   │   └── external_vina.py    # Wrapper for official Vina library/binary
│   ├── preparation/            # Structure preparation
│   │   ├── receptor_prep.py    # Receptor cleaning, metal retention, typing
│   │   ├── ligand_prep.py      # Ligand rotatable bond detection & batch prep
│   │   └── partner_prep.py     # Protein partner preparation for PPD
│   ├── engines/                # Docking search algorithms
│   │   ├── base.py             # DockingPose, DockingResult, BaseDockingEngine
│   │   ├── protein_ligand.py   # MC simulated annealing & torsion search
│   │   └── protein_protein.py  # 6D SO(3) rigid-body exploration & refinement
│   ├── screening/              # Virtual screening subsystem
│   │   ├── virtual_screener.py # Multi-threaded library screening engine
│   │   └── screening_report.py # Leaderboard tables, CSV, SDF & PNG dashboard
│   ├── analysis/               # Results analysis & visualization
│   │   ├── interactions.py     # Profiler for H-bonds, salt bridges, metal bonds
│   │   ├── bond_orders.py      # Bond order and aromaticity perception
│   │   ├── ligand_layout.py    # 2D planar chemical layout algorithms
│   │   ├── visualizer.py       # 2D complete-ligand maps & 3D pocket projections
│   │   ├── rmsd.py             # RMSD & Kabsch superposition
│   │   └── reporter.py         # Tables, CSV exports, complex PDBs
│   └── cli/                    # Command-line interface
│       └── main.py             # CLI entry points (pl-dock, screen, pp-dock, etc.)
├── examples/                   # Runnable examples
│   ├── example_pl_zinc_docking.py  # Protein-ligand docking with Zn2+
│   ├── example_pp_docking.py       # Protein-protein docking with Vina score
│   └── sample_data/                # Clean benchmark PDB models
└── tests/                      # Automated unit test suite
    ├── test_pdb_io.py          # PDB/PDBQT I/O tests
    ├── test_metal_detector.py  # Metal cofactor detection tests
    ├── test_vina_scoring.py    # Vina empirical scoring tests
    ├── test_protein_ligand.py  # Protein-ligand docking tests
    ├── test_protein_protein.py # Protein-protein docking tests
    ├── test_bond_orders.py     # Bond perception tests
    ├── test_ligand_layout.py   # 2D layout tests
    └── test_visualizer.py      # Visualizer rendering tests
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/tjthechemist/moldock.git
cd moldock

# Install in editable mode
pip install -e .
```

*Requirements:* Python 3.9+ with `numpy>=1.20.0` and `pillow>=9.0.0`.

---

## Command-Line Usage

### 1. Detect Metal Co-factors
Inspect a receptor PDB for metal cofactors and their coordination spheres:
```bash
moldock detect-metals --receptor examples/sample_data/receptor_zinc.pdb
```

### 2. Protein-Ligand Docking (SDF or PDB/PDBQT)
Dock a small molecule (from `.sdf`, `.mol`, `.pdbqt`, or `.pdb`) against a metalloprotein, auto-centering on the zinc ion, and render a complete 2D interaction diagram:
```bash
moldock pl-dock \
  --receptor examples/sample_data/receptor_zinc.pdb \
  --ligand ligand.sdf \
  --center-on-metal \
  --exhaustiveness 8 \
  --num-modes 9 \
  --out docked_poses.pdbqt \
  --complex-out complex.pdb \
  --csv-out results.csv \
  --plot-out pl_interaction_map.png
```

### 3. Virtual Screening (Multiple Ligands)
Screen an entire compound library (multi-molecule `.sdf` or directory of ligand files) against a target receptor in parallel:
```bash
moldock screen \
  --receptor examples/sample_data/receptor_zinc.pdb \
  --library compound_library.sdf \
  --out-dir screening_results/ \
  --center-on-metal \
  --workers 4 \
  --top-n 20 \
  --score-cutoff -5.0
```
This automatically produces:
- `screening_results.csv`: Complete ranked table of all docked ligands with energy breakdown.
- `top_hits.sdf`: Docked 3D conformations of top hits with embedded `MOLDOCK_SCORE` metadata tags.
- `dashboard.png`: Multi-panel PNG dashboard with score distributions, hit leaderboards, and cumulative affinity curves.

### 4. Protein-Protein Docking
Dock a partner protein against a receptor directly from PDB files:
```bash
moldock pp-dock \
  --receptor examples/sample_data/receptor_zinc.pdb \
  --partner examples/sample_data/protein_partner.pdb \
  --rotations 120 \
  --translations 40 \
  --out docked_partner.pdb \
  --complex-out complex_pp.pdb \
  --plot-out pp_interface_map.png
```

### 5. Score an Existing Complex
Evaluate the AutoDock Vina score + metal coordination breakdown of any complex:
```bash
moldock score \
  --receptor examples/sample_data/receptor_zinc.pdb \
  --query docked_poses.pdbqt
```

### 6. Render 2D / 3D Visual Results to PNG
```bash
# 2D Interaction Diagram (complete ligand structure + receptor contact nodes)
moldock visualize \
  --receptor examples/sample_data/receptor_zinc.pdb \
  --query docked_poses.pdbqt \
  --mode 2d \
  --out interaction_2d.png

# 3D Binding Pocket & Metal Coordination Projection
moldock visualize \
  --receptor examples/sample_data/receptor_zinc.pdb \
  --query docked_poses.pdbqt \
  --mode 3d \
  --out binding_pocket_3d.png
```

---

## Python API Usage

### Protein-Ligand Docking with SDF Ligand & 2D Interaction Diagram

```python
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
    InteractionVisualizer,
)

# 1. Prepare Receptor and detect metal centers
metal_config = MetalConfig(allowed_metals=["ZN", "MG", "CA"])
rec_preparer = ReceptorPreparer(metal_config=metal_config)
receptor, metal_centers = rec_preparer.prepare_from_pdb("receptor.pdb")

# 2. Prepare Ligand (.sdf, .mol, .pdb, or .pdbqt)
ligand = LigandPreparer.prepare_ligand("inhibitor.sdf")

# 3. Auto-center search grid on metal cofactor
metal_coords = np.array([mc.metal_atom.coord for mc in metal_centers])
grid = GridBox.from_metal_ions(metal_coords, box_radius=10.0)

# 4. Configure & run docking
config = DockingConfig(grid=grid, exhaustiveness=8, num_modes=9)
engine = ProteinLigandEngine(receptor, ligand, config, metal_centers=metal_centers)
result = engine.dock()

# 5. Print results & export
print(DockingReporter.generate_table(result))
result.export_poses_pdbqt("docked_poses.pdbqt")

# 6. Profile intermolecular interactions and render diagram
top_pose = result.get_top_pose()
docked_lig = top_pose.apply_to_molecule(ligand)

contacts = InteractionProfiler().profile_interactions(receptor, docked_lig)
InteractionVisualizer.render_interaction_diagram(
    receptor=receptor,
    query=docked_lig,
    contacts=contacts,
    output_png_path="interaction_map.png",
    title="Protein-Ligand Interaction Map",
    score_info={"total_score": top_pose.score, "metal_score": top_pose.score_breakdown.get("metal_score", 0.0)},
)
```

### High-Throughput Virtual Screening

```python
from moldock import VirtualScreener, ScreeningReporter, DockingConfig, GridBox

# 1. Configure binding site
grid = GridBox(center_x=12.5, center_y=8.0, center_z=15.2, size_x=20.0, size_y=20.0, size_z=20.0)
config = DockingConfig(grid=grid, exhaustiveness=4, num_modes=3)

# 2. Initialize screener with 4 parallel worker threads
screener = VirtualScreener(
    receptor_pdb="receptor.pdb",
    config=config,
    score_cutoff=-5.0,
    max_workers=4,
)

# 3. Screen multi-molecule SDF library
campaign = screener.screen("library.sdf")

# 4. Inspect leaderboard and export reports
print(ScreeningReporter.generate_table(campaign, top_n=10))
ScreeningReporter.export_csv(campaign, "results.csv")
ScreeningReporter.export_hits_sdf(campaign, "top_hits.sdf", top_n=10)
ScreeningReporter.render_dashboard(campaign, "dashboard.png")
```

### Protein-Protein Docking

```python
from moldock import (
    ReceptorPreparer,
    PartnerPreparer,
    GridBox,
    DockingConfig,
    ProteinProteinEngine,
    DockingReporter,
)

rec_prep = ReceptorPreparer()
receptor, metal_centers = rec_prep.prepare_from_pdb("receptor.pdb")
partner = PartnerPreparer.prepare_partner_from_pdb("partner.pdb")

grid = GridBox.from_coordinates(receptor.coords, padding=4.0)
config = DockingConfig(grid=grid, rigid_body_rotations=120, num_modes=5)

engine = ProteinProteinEngine(receptor, partner, config, metal_centers=metal_centers)
result = engine.dock()

print(DockingReporter.generate_table(result))
result.export_poses_pdb("docked_partner_poses.pdb")
```

---

## Running the Test Suite

```bash
python3 -m unittest discover -s tests -v
```

The test suite validates:
- PDB, PDBQT, and SDF/MOL parsing and formatting
- Metal co-factor detection and valency assignment
- AutoDock Vina scoring potentials and coordination well calculations
- Rotatable bond detection and planar 2D layout relaxation
- Protein-ligand simulated annealing and virtual screening pipelines
- 2D complete-ligand interaction image and 3D pocket projection generation

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
