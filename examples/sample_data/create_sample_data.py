"""
Generate benchmark test PDB structures for:
1. Metalloprotein receptor with active site Zn2+ ion and coordinating residues (His94, His96, His119).
2. Small molecule ligand with zinc-binding group (sulfonamide/carboxylate).
3. Partner protein for Protein-Protein docking.
"""

import os


def generate_zinc_receptor_pdb(filepath: str):
    """
    Generate model metalloprotein with Zn2+ catalytic center
    coordinated by His94, His96, His119 and surrounding residues.
    """
    content = """REMARK   1 METALLOPROTEIN MODEL WITH ZN2+ COFACTOR
ATOM      1  N   HIS A  94       1.850  -2.100   0.000  1.00 20.00           N  
ATOM      2  CA  HIS A  94       2.450  -3.250   0.650  1.00 20.00           C  
ATOM      3  C   HIS A  94       3.950  -3.150   0.450  1.00 20.00           C  
ATOM      4  O   HIS A  94       4.450  -2.150   0.000  1.00 20.00           O  
ATOM      5  CB  HIS A  94       1.950  -4.550   0.050  1.00 20.00           C  
ATOM      6  CG  HIS A  94       1.650  -4.550  -1.400  1.00 20.00           C  
ATOM      7  ND1 HIS A  94       1.950  -3.450  -2.150  1.00 20.00           N  
ATOM      8  CD2 HIS A  94       1.050  -5.500  -2.200  1.00 20.00           C  
ATOM      9  CE1 HIS A  94       1.500  -3.750  -3.350  1.00 20.00           C  
ATOM     10  NE2 HIS A  94       0.950  -4.950  -3.450  1.00 20.00           N  
ATOM     11  N   HIS A  96      -1.850  -2.100   0.000  1.00 20.00           N  
ATOM     12  CA  HIS A  96      -2.450  -3.250   0.650  1.00 20.00           C  
ATOM     13  C   HIS A  96      -3.950  -3.150   0.450  1.00 20.00           C  
ATOM     14  O   HIS A  96      -4.450  -2.150   0.000  1.00 20.00           O  
ATOM     15  CB  HIS A  96      -1.950  -4.550   0.050  1.00 20.00           C  
ATOM     16  CG  HIS A  96      -1.650  -4.550  -1.400  1.00 20.00           C  
ATOM     17  ND1 HIS A  96      -1.950  -3.450  -2.150  1.00 20.00           N  
ATOM     18  CD2 HIS A  96      -1.050  -5.500  -2.200  1.00 20.00           C  
ATOM     19  CE1 HIS A  96      -1.500  -3.750  -3.350  1.00 20.00           C  
ATOM     20  NE2 HIS A  96      -0.950  -4.950  -3.450  1.00 20.00           N  
ATOM     21  N   HIS A 119       0.000   1.850  -2.100  1.00 20.00           N  
ATOM     22  CA  HIS A 119       0.650   2.450  -3.250  1.00 20.00           C  
ATOM     23  C   HIS A 119       0.450   3.950  -3.150  1.00 20.00           C  
ATOM     24  O   HIS A 119       0.000   4.450  -2.150  1.00 20.00           O  
ATOM     25  CB  HIS A 119       0.050   1.950  -4.550  1.00 20.00           C  
ATOM     26  CG  HIS A 119      -1.400   1.650  -4.550  1.00 20.00           C  
ATOM     27  ND1 HIS A 119      -2.150   1.950  -3.450  1.00 20.00           N  
ATOM     28  CD2 HIS A 119      -2.200   1.050  -5.500  1.00 20.00           C  
ATOM     29  CE1 HIS A 119      -3.350   1.500  -3.750  1.00 20.00           C  
ATOM     30  NE2 HIS A 119      -3.450   0.950  -4.950  1.00 20.00           N  
ATOM     31  N   GLU A 106       2.500   2.500   1.500  1.00 20.00           N  
ATOM     32  CA  GLU A 106       2.800   3.500   2.500  1.00 20.00           C  
ATOM     33  C   GLU A 106       4.100   3.200   3.100  1.00 20.00           C  
ATOM     34  O   GLU A 106       4.300   2.100   3.600  1.00 20.00           O  
ATOM     35  CB  GLU A 106       1.700   3.700   3.500  1.00 20.00           C  
ATOM     36  CG  GLU A 106       0.300   4.000   2.900  1.00 20.00           C  
ATOM     37  CD  GLU A 106      -0.700   4.100   4.000  1.00 20.00           C  
ATOM     38  OE1 GLU A 106      -0.400   3.600   5.100  1.00 20.00           O  
ATOM     39  OE2 GLU A 106      -1.800   4.600   3.800  1.00 20.00           O  
HETATM   40 ZN    ZN A 262       0.000  -1.500  -1.800  1.00 15.00          ZN  
END
"""
    with open(filepath, "w") as f:
        f.write(content.strip() + "\n")


def generate_ligand_pdb(filepath: str):
    """
    Generate small molecule inhibitor with a zinc-binding Lewis base oxygen/nitrogen.
    """
    content = """REMARK   1 SMALL MOLECULE INHIBITOR
HETATM    1  C1  LIG A   1       0.000   0.500   0.000  1.00 10.00           C  
HETATM    2  C2  LIG A   1       1.200   1.200   0.000  1.00 10.00           C  
HETATM    3  C3  LIG A   1       1.200   2.600   0.000  1.00 10.00           C  
HETATM    4  C4  LIG A   1       0.000   3.300   0.000  1.00 10.00           C  
HETATM    5  C5  LIG A   1      -1.200   2.600   0.000  1.00 10.00           C  
HETATM    6  C6  LIG A   1      -1.200   1.200   0.000  1.00 10.00           C  
HETATM    7  S1  LIG A   1       0.000  -1.200   0.000  1.00 10.00           S  
HETATM    8  O1  LIG A   1       1.250  -1.600   0.550  1.00 10.00           O  
HETATM    9  O2  LIG A   1      -1.250  -1.600   0.550  1.00 10.00           O  
HETATM   10  N1  LIG A   1       0.000  -1.500  -1.500  1.00 10.00           N  
END
"""
    with open(filepath, "w") as f:
        f.write(content.strip() + "\n")


def generate_partner_protein_pdb(filepath: str):
    """
    Generate partner protein model for protein-protein docking.
    """
    content = """REMARK   1 PARTNER PROTEIN (BINDING DOMAIN)
ATOM      1  N   ALA B   1       0.000   5.000   0.000  1.00 20.00           N  
ATOM      2  CA  ALA B   1       0.500   6.200   0.600  1.00 20.00           C  
ATOM      3  C   ALA B   1       2.000   6.100   0.800  1.00 20.00           C  
ATOM      4  O   ALA B   1       2.500   5.100   1.200  1.00 20.00           O  
ATOM      5  CB  ALA B   1      -0.200   6.500   1.900  1.00 20.00           C  
ATOM      6  N   HIS B   2       2.700   7.200   0.500  1.00 20.00           N  
ATOM      7  CA  HIS B   2       4.150   7.300   0.700  1.00 20.00           C  
ATOM      8  C   HIS B   2       4.700   8.500  -0.100  1.00 20.00           C  
ATOM      9  O   HIS B   2       5.900   8.600  -0.200  1.00 20.00           O  
ATOM     10  CB  HIS B   2       4.800   6.000   0.200  1.00 20.00           C  
ATOM     11  CG  HIS B   2       4.400   4.800   1.000  1.00 20.00           C  
ATOM     12  ND1 HIS B   2       3.200   4.700   1.650  1.00 20.00           N  
ATOM     13  CD2 HIS B   2       5.050   3.650   1.250  1.00 20.00           C  
ATOM     14  CE1 HIS B   2       3.150   3.550   2.250  1.00 20.00           C  
ATOM     15  NE2 HIS B   2       4.250   2.900   2.050  1.00 20.00           N  
ATOM     16  N   ASP B   3       3.850   9.350  -0.650  1.00 20.00           N  
ATOM     17  CA  ASP B   3       4.250  10.500  -1.450  1.00 20.00           C  
ATOM     18  C   ASP B   3       4.400  11.750  -0.600  1.00 20.00           C  
ATOM     19  O   ASP B   3       5.250  11.850   0.250  1.00 20.00           O  
ATOM     20  CB  ASP B   3       3.250  10.800  -2.550  1.00 20.00           C  
ATOM     21  CG  ASP B   3       3.500  12.150  -3.200  1.00 20.00           C  
ATOM     22  OD1 ASP B   3       4.450  12.850  -2.800  1.00 20.00           O  
ATOM     23  OD2 ASP B   3       2.750  12.500  -4.100  1.00 20.00           O  
END
"""
    with open(filepath, "w") as f:
        f.write(content.strip() + "\n")


if __name__ == "__main__":
    outdir = os.path.dirname(os.path.abspath(__file__))
    generate_zinc_receptor_pdb(os.path.join(outdir, "receptor_zinc.pdb"))
    generate_ligand_pdb(os.path.join(outdir, "ligand_inhibitor.pdb"))
    generate_partner_protein_pdb(os.path.join(outdir, "protein_partner.pdb"))
    print("Benchmark PDB files generated successfully.")

