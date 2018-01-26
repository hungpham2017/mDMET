'''
Testing the self_consistent DMET
'''

import sys
import pyscf
from pyscf import gto, scf, dft, ao2mo
import numpy as np
import pytest
from mdmet import orthobasis, schmidtbasis, qcsolvers, dmet
from functools import reduce
import scipy as scipy
sys.path.append('/panfs/roc/groups/6/gagliard/phamx494/QC-DMET/src')
import localintegrals, qcdmet_paths
import dmet as qc_dmet
sys.path.append('./lib/build')
import libdmet
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
	mydft = dft.RKS(mol)
	mydft.xc = 'b3lyp'
	mydft.kernel()

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

	return mol, mf, mydft, impClusters 

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

def test_self_consistent():
	#pmDMET
	mol, mf, mydft, impClusters  = test_makemole1()
	symmetry = None  #or [0]*5, takes longer time
	solverlist = 'CASCI' #['RHF', 'CASCI', 'CASCI', 'CASCI', 'CASCI']
	runDMET = dmet.DMET(mf, impClusters, symmetry, orthogonalize_method = 'overlap', schmidt_decomposition_method = 'OED', OEH_type = 'FOCK', SC_CFtype = 'F', solver = solverlist)
	#runDMET.CAS = [[4,4]]
	runDMET.SC_canonical = True
	time1 = time.time()
	runDMET.self_consistent()
	time2 = time.time()
	time_pmDMET = time2 - time1
	
	#QC-DMET	
	myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
	myInts.TI_OK = False
	method = 'CASSCF'
	SCmethod = 'BFGS' #Don't do it self-consistently
	TI = False
	theDMET = qc_dmet.dmet( myInts, impClusters, TI, method, SCmethod )	
	theDMET.impCAS = (4,4)
	time1 = time.time()
	theDMET.doselfconsistent()
	time2 = time.time()
	time_QCDMET = time2 - time1	
	
	return runDMET.Energy_total, time_pmDMET, time_QCDMET	