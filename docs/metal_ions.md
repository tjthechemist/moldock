# Metal Ion Co-factor Handling & Extended Scoring

In standard AutoDock Vina, metal ions in PDBQT files are either omitted, treated as inert spheres, or cause runtime failures due to missing parameters for transition metals. In biology, however, over **one-third of all proteins require metal ion cofactors** for catalytic activity or structural integrity (e.g. Zinc metalloproteases, Magnesium kinases, Calcium signaling proteins, Iron cytochromes).

`MolDock` provides first-class support for metal ions as co-factors, both when preparing receptors and when evaluating binding affinity.

---

## 1. Supported Metal Co-factors

`MolDock` includes calibrated biophysical parameters for:
- **$\text{Zn}^{2+}$ (Zinc)**: Tetrahedral/octahedral, ideal donor distances: $\text{O}$ (2.05 Å), $\text{N}$ (2.08 Å), $\text{S}$ (2.32 Å).
- **$\text{Mg}^{2+}$ (Magnesium)**: Octahedral, ideal donor distances: $\text{O}$ (2.08 Å), $\text{N}$ (2.18 Å).
- **$\text{Ca}^{2+}$ (Calcium)**: Pentagonal bipyramidal / octahedral, ideal donor distances: $\text{O}$ (2.40 Å).
- **$\text{Mn}^{2+}$ (Manganese)**: Octahedral, ideal donor distances: $\text{O}$ (2.15 Å), $\text{N}$ (2.20 Å), $\text{S}$ (2.40 Å).
- **$\text{Fe}^{2+}/\text{Fe}^{3+}$ (Iron)**: Octahedral, ideal donor distances: $\text{O}$ (2.10 Å), $\text{N}$ (2.15 Å), $\text{S}$ (2.30 Å).
- **$\text{Cu}^{2+}$ (Copper)**: Square planar / tetrahedral, ideal donor distances: $\text{O}$ (1.98 Å), $\text{N}$ (2.01 Å).
- **$\text{Co}^{2+}$ (Cobalt)**: Octahedral, ideal donor distances: $\text{O}$ (2.08 Å), $\text{N}$ (2.12 Å).
- **$\text{Ni}^{2+}$ (Nickel)**: Octahedral, ideal donor distances: $\text{O}$ (2.05 Å), $\text{N}$ (2.08 Å).
- **$\text{Na}^+, \text{K}^+$ (Alkali ions)**: Monovalent coordination spheres.

---

## 2. Multi-Metal Coordination Scoring Function

The composite scoring function computes:
$$E_{\text{total}} = E_{\text{vina}} + E_{\text{metal}} + E_{\text{elec}}$$

Where the metal term $E_{\text{metal}}$ sums across all detected metal co-factors $m \in M$ and Lewis base donor atoms $d \in D$ (oxygen, nitrogen, sulfur):
$$E_{\text{metal}} = \sum_{m \in M} \sum_{d \in D} \left( E_{\text{coord}}(r_{md}) + E_{\text{rep}}(r_{md}) + E_{\text{pol}}(r_{md}) \right)$$

### 2.1 Gaussian Coordination Attractive Well
$$E_{\text{coord}}(r_{md}) = -w_{\text{metal}} \cdot \exp\left( -\frac{(r_{md} - r_0)^2}{2\sigma^2} \right)$$
- $r_{md}$: Interatomic distance between metal $m$ and donor $d$.
- $r_0$: Ideal coordination bond distance specific to the metal element and donor element (e.g. 2.05 Å for $\text{Zn}-\text{O}$, 2.08 Å for $\text{Zn}-\text{N}$).
- $\sigma$: Dispersion parameter (0.35 Å), creating an attractive potential well around the coordination sphere.

### 2.2 Short-Range Steric Repulsion
If the incoming atom is closer than the physical coordination distance:
$$E_{\text{rep}}(r_{md}) = 
\begin{cases} 
k_{\text{rep}} \cdot (r_{\text{onset}} - r_{md})^2 & \text{if } r_{md} < r_{\text{onset}} \\
0 & \text{otherwise}
\end{cases}$$
where $r_{\text{onset}} = r_0 - 0.35\text{ \AA}$ and $k_{\text{rep}} = 25.0\text{ kcal/mol}\cdot\text{\AA}^{-2}$.

### 2.3 12-6-4 Polarization Potential
Standard 12-6 Lennard-Jones models fail to capture the charge-induced dipole polarization that occurs when divalent cations interact with polar donor atoms. `MolDock` includes the $C_4 / r^4$ polarization term:
$$E_{\text{pol}}(r_{md}) = -\frac{C_4}{r_{md}^4}$$
calibrated for each metal ion (e.g. $C_4 = 115.0$ for $\text{Zn}^{2+}$, $95.0$ for $\text{Mg}^{2+}$).

### 2.4 Valency & Multi-Metal Clustering
- **Valency awareness**: If a metal ion is already coordinated by 3 histidine residues in a catalytic site (preferred coordination number 4), its remaining valency is 1. If a ligand attempts to coordinate with more donor atoms than available valency, diminishing returns/penalty is applied.
- **Multi-metal sites**: For dinuclear active sites (e.g. di-zinc or di-iron enzymes within 4.5 Å of each other), `MetalDetector` groups them into a cluster, allowing ligands to bridge between both metals.
