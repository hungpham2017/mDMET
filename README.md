# pydmet
Density matrix embedding theory for periodic/molecular systems

NOTES:
- The molecular DMET (mdmet) uses some of the functions in the QC-DMET <Copyright (C) 2015 Sebastian Wouters>
- New Features:
-- Using overlap matrix to do Smith decomposition (testing)
-- Using symmetry and multiple solvers
-- single-embedding DMET (similar to CASCI=True in QC-DMET)
-- New solvers: UHF, CASSCF

TODO:
- Smith decomposition for a UHF wavefunction
- Smith decomposition from a Kohn-Sham 1RDM
