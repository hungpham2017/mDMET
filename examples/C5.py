'''
Multipurpose Density Matrix Embedding theory (mp-DMET)
Copyright (C) 2015 Hung Q. Pham
Author: Hung Q. Pham, Unviversity of Minnesota
email: phamx494@umn.edu
'''

import sys
import pyscf
from pyscf import gto, scf, dft, ao2mo
import numpy as np
sys.path.append('/panfs/roc/groups/6/gagliard/phamx494/mpdmet')
from mdmet import orthobasis, schmidtbasis, qcsolvers, dmet
from functools import reduce
import scipy as scipy
sys.path.append('/panfs/roc/groups/6/gagliard/phamx494/QC-DMET/src')
import localintegrals, qcdmet_paths
import dmet as qc_dmet
sys.path.append('./lib/build')
import libdmet
import time

def test_makemole():
	mol = gto.Mole()
	mol.atom = '''
	C                 -3.23834213    0.20725388    0.00000000
	H                 -2.88168770   -0.80155612    0.00000000
	H                 -2.88166929    0.71165207   -0.87365150
	H                 -4.30834213    0.20726707    0.00000000
	C                 -2.72499991    0.93321016    1.25740497
	H                 -3.08326902    0.42994172    2.13105486
	H                 -1.65500172    0.93150347    1.25838372
	C                 -3.23601559    2.38595273    1.25599807
	H                 -4.30601337    2.38765891    1.25464312
	H                 -2.87743949    2.88932897    0.38253625
	C                 -2.72311570    3.11175368    2.51367319
	H                 -1.65311809    3.10997445    2.51506932
	H                 -3.08175976    2.60842591    3.38713503
	C                 -3.23403330    4.56453069    2.51220642
	H                 -4.30403090    4.56630901    2.51080057
	H                 -2.87767531    5.06881831    3.38605022
	H                 -2.87538173    5.06786134    1.63874933
	'''

	mol.basis = 'sto-3g'
	mol.build(verbose=0)

	mf = scf.RHF(mol)
	mf.scf()
	A1=np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
	A2=np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
	A3=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
	A4=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0])
	A5=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
	impClusters = [A1, A2, A3, A4, A5]

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

for bond in np.arange(0.8, 2.0, 0.2): 
	mol, mf, impClusters  = test_makemole(bond)
	symmetry = 'Translation'  #or [0]*5, takes longer time
	solverlist = 'CASCI' #['RHF', 'CASCI', 'CASCI', 'CASCI', 'CASCI']
	runDMET = dmet.DMET(mf, impClusters, symmetry, orthogonalize_method = 'overlap', schmidt_decomposition_method = 'OED', OEH_type = 'FOCK', SC_CFtype = 'F', solver = solverlist)
	#runDMET.CAS = [[4,4]]
	time1 = time.time()
	runDMET.self_consistent()
	time2 = time.time()
	time_mpDMET = time2 - time1
	print('Total energy + Time:', runDMET.Energy_total, time_mpDMET)
	
	'''#QC-DMET	
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
	time_QCDMET = time2 - time1'''	