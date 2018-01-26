'''
Testing the self_consistent DMET
'''

import sys, os
from pyscf import gto, scf
import numpy as np
import pytest
from mdmet import orthobasis, schmidtbasis, qcsolvers, dmet
import time

def test_makemole1():
	bondlength = 1.0
	nat = 10
	mol = gto.Mole()
	mol.atom = []
	r = 0.5 * bondlength / np.sin(np.pi/nat)
	for i in range(nat):
		theta = i * (2*np.pi/nat)
		mol.atom.append(('H', (r*np.cos(theta), r*np.sin(theta), 0)))

	mol.basis = 'sto-6g'
	mol.build(verbose=0)

	mf = scf.RHF(mol)
	mf.scf()

	atoms_per_imp = 2 # Impurity size = 1 atom
	Norbs = mol.nao_nr()
	assert ( nat % atoms_per_imp == 0 )
	orbs_per_imp = int(Norbs * atoms_per_imp // nat)

	impClusters = []
	for cluster in range(nat // atoms_per_imp):
		impurities = np.zeros([Norbs], dtype=int)
		for orb in range( orbs_per_imp ):
			impurities[orbs_per_imp*cluster + orb] = 1
		impClusters.append(impurities)

	return mol, mf, impClusters 

def test_makemole2():
	bondlength = 1.0
	mol = gto.M(atom='He 0 0.5 0; He -0.5 0 0; Be 1 0 0; Be 2 0 0; He 3 0.5 0; He 3 -0.5 0', basis='sto-6g')
	mol.build(verbose=0)
	mf = scf.RHF(mol)
	mf.scf()
	
	unit_sizes = np.array([ 2, 5, 5, 2])
	impClusters = []
	for frag in range(4):
		impurity_orbitals = np.zeros( [mol.nao_nr()], dtype = int)
		start = unit_sizes[:frag].sum()
		impurity_orbitals[start:(start + unit_sizes[frag])] = 1
		impClusters.append(impurity_orbitals)	
	return mol, mf, impClusters 
def test_solvers():
	#mpDMET
	mol, mf, impClusters  = test_makemole1()
	symmetry = 'Translation'  #or [0]*5, takes longer time
	
	time1 = time.time()
	solverlist = 'CASCI' #['RHF', 'CASCI', 'CASCI', 'CASCI', 'CASCI']
	runDMET = dmet.DMET(mf, impClusters, symmetry, orthogonalize_method = 'overlap', schmidt_decomposition_method = 'OED', OEH_type = 'FOCK', SC_CFtype = 'F', solver = solverlist)
	runDMET.one_shot()
	time2 = time.time()
	time_FCI_1 = time2 - time1
	E_FCI_1 = runDMET.Energy_total
	
	time1 = time.time()
	solverlist = 'DMRG-CASCI-B' #['RHF', 'CASCI', 'CASCI', 'CASCI', 'CASCI']
	runDMET = dmet.DMET(mf, impClusters, symmetry, orthogonalize_method = 'overlap', schmidt_decomposition_method = 'OED', OEH_type = 'FOCK', SC_CFtype = 'F', solver = solverlist)
	runDMET.one_shot()
	time2 = time.time()
	time_FCI_2 = time2 - time1
	E_FCI_2 = runDMET.Energy_total

	time1 = time.time()
	solverlist = 'DMRG-CASCI-C' #['RHF', 'CASCI', 'CASCI', 'CASCI', 'CASCI']
	runDMET = dmet.DMET(mf, impClusters, symmetry, orthogonalize_method = 'overlap', schmidt_decomposition_method = 'OED', OEH_type = 'FOCK', SC_CFtype = 'F', solver = solverlist)
	runDMET.one_shot()
	time2 = time.time()
	time_FCI_3 = time2 - time1
	E_FCI_3 = runDMET.Energy_total
	
	assert np.isclose(E_FCI_2, E_FCI_1)	
	assert np.isclose(E_FCI_3, E_FCI_1)
	
	#Remove DMRG temp directories
	for i in range(10):
		os.system('rm -rf ' + str(i) + '*')


