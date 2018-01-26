'''
Testing lattice models
'''

import sys
from pyscf import gto, scf, mcscf
import numpy as np
import pytest
from mdmet import orthobasis, schmidtbasis, qcsolvers, dmet
from mdmet.latticeHamiltonian import hubbard_1D, hubbard_2D_rectangular
import time

def test_hubbard1D_FCI():
	
	num_sites = 4
	filling = 0.5
	t = -1.0
	U = 8.0
	boundary_conditions = 'open'
	site_energy = [0.3, 0, 0.3, 0]
	no_hopping = [[1,2]] #[1]
	mf_hubbard = hubbard_1D(num_sites, filling, t, U, boundary_conditions, site_energy, no_hopping)
	
	mc = mcscf.CASCI(mf_hubbard, 4,4)
	EFCI = mc.kernel()[0]
	#assert np.isclose(np.round(EFCI/4,decimals=3), −0.086) #J. Chem. Phys. 143, 024107 (2015)	
	return EFCI, mf_hubbard
	
def test_hubbard2D():
	
	num_sites = [8,8]
	filling = 0.25
	t = -1.0
	U = 1.0
	mf_hubbard = hubbard_2D_rectangular(num_sites, filling, t, U, boundary_conditions = 'antipbc', site_energy = None, no_hopping = None)
	return mf_hubbard
	
'''def test_hubbard1D_DMET():
	EFCI, mf_hubbard = test_hubbard1D_FCI()	
	impClusters = [[1, 1, 0, 0],[0, 0, 1, 1]]
	symmetry = None
	solverlist = 'DMRG'
	runDMET = dmet.DMET(mf_hubbard, impClusters, symmetry, orthogonalize_method = 'lattice', schmidt_decomposition_method = 'OED', OEH_type = 'FOCK', SC_CFtype = 'FB', solver = solverlist)	
	runDMET.self_consistent()
	assert np.isclose(runDMET.Energy_total, EFCI)'''
	
