# MolDock Software Architecture

`MolDock` is a modular, high-performance Python framework for molecular docking that integrates the AutoDock Vina empirical scoring function with explicit coordination potentials for metal ion co-factors. It supports both **Protein-Ligand** and **Protein-Protein** docking directly from standard PDB/PDBQT structures.

---

## 1. System Architecture Overview

```
                                      +--------------------------------+
                                      |          User / CLI            |
                                      +--------------------------------+
                                                      |
                    +---------------------------------+---------------------------------+
                    |                                                                   |
        +-----------------------+                                           +-----------------------+
        |   ReceptorPreparer    |                                           | Ligand/PartnerPrep    |
        +-----------------------+                                           +-----------------------+
                    |                                                                   |
                    v                                                                   v
        +-----------------------+                                           +-----------------------+
        |  Receptor (Molecule)  |                                           | Query (Molecule)      |
        +-----------------------+                                           +-----------------------+
                    |                                                                   |
                    +-----------------+                       +-------------------------+
                                      |                       |
                                      v                       v
                        +---------------------------------------------------+
                        |             Docking Search Engine                 |
                        |  - ProteinLigandEngine (MC Annealing + Torsions)  |
                        |  - ProteinProteinEngine (6D SO(3) Rigid Body)     |
                        +---------------------------------------------------+
                                                  |
                                                  v
                        +---------------------------------------------------+
                        |             CompositeScoringFunction              |
                        |  - VinaScoringFunction (Gauss1, Gauss2, Rep, HB)  |
                        |  - MetalCoordinationScorer (12-6-4 + Wells)       |
                        |  - Electrostatic Coulomb Potential                |
                        +---------------------------------------------------+
                                                  |
                                                  v
                        +---------------------------------------------------+
                        |             Analysis & Reporting                  |
                        |  - InteractionProfiler (HBond, Salt, Metal)       |
                        |  - DockingReporter (PDB/PDBQT/CSV/Tables)         |
                        +---------------------------------------------------+
```

---

## 2. Core Modules

### 2.1 `moldock.core`
- **`Atom`** ([`atom.py`](file:///home/tawatchai/Desktop/Projects/molecular-docking-python/moldock/core/atom.py)): Models 3D coordinates, AutoDock atom typing (`C`, `A`, `OA`, `NA`, `SA`, `HD`, etc.), Van der Waals radii, partial charges, and metal identification.
- **`Molecule`** ([`molecule.py`](file:///home/tawatchai/Desktop/Projects/molecular-docking-python/moldock/core/molecule.py)): Container for atoms providing 3D matrix transformations (translation, rotation, center of mass) and chain/residue filtering.
- **`PDBIO` & `PDBQTIO`** ([`pdb_io.py`](file:///home/tawatchai/Desktop/Projects/molecular-docking-python/moldock/core/pdb_io.py)): Robust parsing and writing of PDB and AutoDock PDBQT formats.
- **`GridBox` & `SpatialGrid`** ([`grid.py`](file:///home/tawatchai/Desktop/Projects/molecular-docking-python/moldock/core/grid.py)): Bounding search volume calculations and spatial cell partitioning for fast neighbor pair discovery.

### 2.2 `moldock.metals`
- **`MetalProperties` & `METAL_DATABASE`** ([`parameters.py`](file:///home/tawatchai/Desktop/Projects/molecular-docking-python/moldock/metals/parameters.py)): Biophysical parameters for transition metals and alkaline earth ions ($\text{Zn}^{2+}, \text{Mg}^{2+}, \text{Ca}^{2+}, \text{Mn}^{2+}, \text{Fe}^{2+/3+}, \text{Cu}^{2+}, \text{Co}^{2+}, \text{Ni}^{2+}, \text{Na}^+, \text{K}^+$).
- **`MetalDetector`** ([`detector.py`](file:///home/tawatchai/Desktop/Projects/molecular-docking-python/moldock/metals/detector.py)): Inspects macromolecular structures, detects co-factor metal ions, maps coordinating protein residues, groups multi-metal clusters, and computes remaining binding valency.
- **`MetalCoordinationScorer`** ([`scoring.py`](file:///home/tawatchai/Desktop/Projects/molecular-docking-python/moldock/metals/scoring.py)): Computes Gaussian coordination wells, steric repulsion, and 12-6-4 polarization terms for single or multiple metal cofactors.

### 2.3 `moldock.scoring`
- **`VinaScoringFunction`** ([`vina_scoring.py`](file:///home/tawatchai/Desktop/Projects/molecular-docking-python/moldock/scoring/vina_scoring.py)): Fully vectorized pure-Python implementation of the standard AutoDock Vina empirical scoring terms (Trott & Olson 2010):
  - $\text{Gauss}_1$ and $\text{Gauss}_2$ attractive dispersion
  - Quadratic steric repulsion ($d < 0$)
  - Hydrophobic-hydrophobic pair ramp
  - Hydrogen bonding donor-acceptor ramp
  - Rotatable bond torsion normalization factor: $(1 + w_{\text{rot}} \cdot N_{\text{rot}})^{-1}$
- **`CompositeScoringFunction`** ([`composite_scoring.py`](file:///home/tawatchai/Desktop/Projects/molecular-docking-python/moldock/scoring/composite_scoring.py)): Unifies Vina scoring, metal coordination, and distance-dependent electrostatics into a single binding energy metric.

### 2.4 `moldock.engines`
- **`ProteinLigandEngine`** ([`protein_ligand.py`](file:///home/tawatchai/Desktop/Projects/molecular-docking-python/moldock/engines/protein_ligand.py)): Monte Carlo Simulated Annealing exploring 6D rigid body degrees of freedom plus internal dihedral torsion angles around rotatable bonds.
- **`ProteinProteinEngine`** ([`protein_protein.py`](file:///home/tawatchai/Desktop/Projects/molecular-docking-python/moldock/engines/protein_protein.py)): Rigid-body macromolecular docking from PDB files using Fibonacci SO(3) rotational orientation grids, surface contact filtering, and Vina interface score refinement.

### 2.5 `moldock.analysis`
- **`InteractionProfiler`** ([`interactions.py`](file:///home/tawatchai/Desktop/Projects/molecular-docking-python/moldock/analysis/interactions.py)): Analyzes docked complexes for metal coordination, hydrogen bonds, salt bridges, and hydrophobic contacts.
- **`DockingReporter`** ([`reporter.py`](file:///home/tawatchai/Desktop/Projects/molecular-docking-python/moldock/analysis/reporter.py)): Generates Vina-style terminal tables, CSV exports, and merged complex PDB files.
