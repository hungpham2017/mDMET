# pmDMET
Density matrix embedding theory for periodic/molecular systems

## Overview:
### 1. Notes: 
- The molecular DMET (mdmet) uses some functions modified from those in the QC-DMET <Copyright (C) 2015 Sebastian Wouters>
### 2. Features:
- Using overlap matrix to do Smith decomposition
- Using symmetry and multiple solvers
- Single-embedding DMET (similar to CASCI=True in QC-DMET)
- Quantum chemical solvers: RHF, CASCI, CASSCF

### 3. On progress:
- Smith decomposition for a UHF wavefunction
- Smith decomposition from a Kohn-Sham 1RDM
- Maximally localized Wannier functions for pyscf/pbc class
