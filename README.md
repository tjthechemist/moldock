# MolDock: Python Molecular Docking Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

**MolDock** is a modular, high-performance Python framework for molecular docking. It provides:
1. **Protein-Ligand Docking**: Global Monte Carlo Simulated Annealing with torsional flexibility.
2. **Protein-Protein Docking (PPD)**: Macromolecular rigid-body orientation exploration from PDB files.
3. **AutoDock Vina Scoring**: Vectorized implementation of the empirical AutoDock Vina scoring function (Gauss1, Gauss2, Repulsion, Hydrophobic, Hydrogen bonding, torsion penalties).
4. **Multi-Metal Co-factor Coordination**: Explicit support for catalytic and structural metal ions ($\text{Zn}^{2+}, \text{Mg}^{2+}, \text{Ca}^{2+}, \text{Mn}^{2+}, \text{Fe}^{2+/3+}, \text{Cu}^{2+}, \text{Co}^{2+}, \text{Ni}^{2+}$) using coordination wells and 12-6-4 polarization potentials.

---

## Key Features

- **Zero Heavy Binary Dependencies**: Native pure-Python/NumPy implementation works out of the box with zero external compilation needed.
- **Metal Co-factor Retention & Valency**: Automatically scans receptor PDB files for metal ions, maps coordinating protein residues, assigns charges, and scores metal-ligand coordinate bonds.
- **AutoDock Vina Compatibility**: Full support for `.pdbqt` and `.pdb` formats, compatible with standard AutoDock Vina workflows and poses.
- **Protein-Protein Docking**: Load receptor and partner proteins directly from PDB files, perform 6D $SO(3)$ surface sampling, and evaluate interfaces using Vina potentials.
- **CLI & Python API**: Intuitive command-line interface (`moldock`) alongside a clean Python API.

---

## Directory Structure

```
molecular-docking-python/
├── pyproject.toml              # Build & dependency metadata
├── setup.py                    # Package installer
├── requirements.txt            # Core dependencies (numpy)
├── README.md                   # Project overview & documentation
├── docs/                       # Technical architecture & math formulation
│   ├── architecture.md         # System design & module breakdown
│   └── metal_ions.md           # Metal coordination physics & 12-6-4 potentials
├── moldock/                    # Main Python package
│   ├── __init__.py             # Top-level API imports
│   ├── core/                   # Core representations
│   │   ├── atom.py             # Atom class, VdW radii, AutoDock atom types
│   │   ├── molecule.py         # Molecule container & 3D transformations
│   │   ├── pdb_io.py           # PDB & PDBQT parser/writer
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
│   │   ├── receptor_prep.py    # Receptor cleaning, metal retention, PDBQT
│   │   ├── ligand_prep.py      # Ligand rotatable bond detection & charges
│   │   └── partner_prep.py     # Protein partner preparation for PPD
│   ├── engines/                # Docking search algorithms
│   │   ├── base.py             # DockingPose, DockingResult, BaseDockingEngine
│   │   ├── protein_ligand.py   # MC simulated annealing & torsion search
│   │   └── protein_protein.py  # 6D SO(3) rigid-body exploration & refinement
│   ├── analysis/               # Results analysis & profiling
│   │   ├── interactions.py     # Hydrogen bonds, salt bridges, metal bonds
│   │   ├── rmsd.py             # RMSD & Kabsch superposition
│   │   └── reporter.py         # Tables, CSV exports, complex PDBs
│   └── cli/                    # Command-line interface
│       └── main.py             # Entry points for pl-dock, pp-dock, detect-metals
├── examples/                   # Runnable examples
│   ├── example_pl_zinc_docking.py  # Protein-ligand docking with Zn2+
│   ├── example_pp_docking.py       # Protein-protein docking with Vina score
│   └── sample_data/                # Clean benchmark PDB models
└── tests/                      # Automated unit test suite
    ├── test_pdb_io.py          # PDB/PDBQT I/O tests
    ├── test_metal_detector.py  # Metal cofactor detection tests
    ├── test_vina_scoring.py    # Vina empirical scoring tests
    ├── test_protein_ligand.py  # Protein-ligand docking tests
    └── test_protein_protein.py # Protein-protein docking tests
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

*Requirements:* Python 3.9+ and `numpy>=1.20.0`.

---

## Command-Line Usage

### 1. Detect Metal Co-factors
Inspect a receptor PDB for metal cofactors and their coordinating residues:
```bash
moldock detect-metals --receptor examples/sample_data/receptor_zinc.pdb
```

### 2. Protein-Ligand Docking (with PNG Interaction Diagram)
Dock a small molecule ligand against a metalloprotein, auto-centering on the zinc ion, and export a PNG interaction diagram:
```bash
moldock pl-dock \
  --receptor examples/sample_data/receptor_zinc.pdb \
  --ligand examples/sample_data/ligand_inhibitor.pdb \
  --center-on-metal \
  --exhaustiveness 8 \
  --num-modes 9 \
  --out docked_poses.pdbqt \
  --complex-out complex.pdb \
  --csv-out results.csv \
  --plot-out pl_interaction_map.png
```

### 3. Protein-Protein Docking (with PNG Interface Diagram)
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

### 4. Score an Existing Complex
Evaluate the AutoDock Vina score + metal coordination breakdown of a complex:
```bash
moldock score \
  --receptor examples/sample_data/receptor_zinc.pdb \
  --query docked_poses.pdbqt
```

### 5. Render 2D / 3D Visual Results to PNG
Render a 2D interaction network or 3D active site projection from any docked pose or complex:
```bash
# 2D Interaction Diagram (showing H-bonds, metal bonds, distances)
moldock visualize \
  --receptor examples/sample_data/receptor_zinc.pdb \
  --query docked_poses.pdbqt \
  --mode 2d \
  --out interaction_2d.png

# 3D Active Site & Metal Coordination Pocket Projection
moldock visualize \
  --receptor examples/sample_data/receptor_zinc.pdb \
  --query docked_poses.pdbqt \
  --mode 3d \
  --out binding_pocket_3d.png
```


---

## Python API Usage

### Protein-Ligand Docking with Metal Cofactor

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
)

# 1. Prepare Receptor and detect metal centers
metal_config = MetalConfig(allowed_metals=["ZN", "MG", "CA"])
rec_preparer = ReceptorPreparer(metal_config=metal_config)
receptor, metal_centers = rec_preparer.prepare_from_pdb("receptor.pdb")

# 2. Prepare Ligand (identifies torsions and Lewis donors)
ligand = LigandPreparer.prepare_ligand("ligand.pdb")

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

# 6. Profile intermolecular interactions
top_pose = result.get_top_pose()
docked_lig = top_pose.apply_to_molecule(ligand)
contacts = InteractionProfiler().profile_interactions(receptor, docked_lig)
for c in contacts:
    print(f"[{c.interaction_type}] {c.receptor_res}:{c.receptor_atom_name} <-> {c.query_atom_name} ({c.distance:.2f} Å)")
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

All unit tests run in <1 second and validate PDB/PDBQT parsing, metal detection, Vina empirical scoring terms, and end-to-end docking pipelines.

