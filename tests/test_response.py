'''
Testing the implementation of libdmet (in C++).
Author: Hung Q. Pham, Unviversity of Minnesota
email: phamx494@umn.edu
'''

import sys
import pyscf
from pyscf import gto, scf
import numpy as np
import pytest
from mdmet import orthobasis, schmidtbasis, qcsolvers, dmet
from functools import reduce
sys.path.append('/panfs/roc/groups/6/gagliard/phamx494/QC-DMET/src')
import localintegrals, qcdmet_paths
import dmet as qc_dmet
sys.path.append('./lib/build')
import libdmet
import ctypes

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
	nat = 10
	mol = gto.Mole()
	mol.atom = []
	r = 0.5 * bondlength / np.sin(np.pi/nat)
	for i in range(nat):
		theta = i * (2*np.pi/nat)
		if i%3 == 0: 
			element = 'He'
		else:
			element = 'Be'
		mol.atom.append((element, (r*np.cos(theta), r*np.sin(theta), 0)))

	mol.basis = 'sto-3g'
	mol.build(verbose=0)

	mf = scf.RHF(mol)
	mf.scf()
	atoms_per_imp = 2 # Impurity size = 1 atom
	Norbs = mol.nao_nr()
	assert ( nat % atoms_per_imp == 0 )

	#Parition: (He-Be)-(BeHe)-(BeBe)-(HeBe)-(BeHe) = 6-6-10-6-6 orb
	impClusters = []
	for cluster in range(nat // atoms_per_imp):
		if cluster == 2:
			impurities = np.zeros([Norbs], dtype=int)
			impurities[12:22] = 1
		elif cluster == 3:
			impurities = np.zeros([Norbs], dtype=int)
			impurities[22:28] = 1	
		elif cluster == 4:
			impurities = np.zeros([Norbs], dtype=int)
			impurities[28:34] = 1			
		else:
			impurities = np.zeros([Norbs], dtype=int)
			impurities[(6*cluster):(6*(cluster+1))] = 1		
		impClusters.append(impurities)
	assert (sum(impClusters).sum() == Norbs)
	return mol, mf, impClusters 
		
def test_1rdm_response():
	mol, mf, impClusters  = test_makemole1()
	#symmetry = [0, 1, 2, 1, 0]
	#runDMET = dmet.DMET(mf, impClusters, symmetry, orthogonalize_method = 'overlap', schmidt_decomposition_method = 'OED', OEH_type = 'FOCK', SC_CFtype = 'FB', solver = 'RHF')
	#runDMET.one_shot()
	umat = np.zeros((mol.nao_nr(), mol.nao_nr()))	

	#QC-DMET
	myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
	myInts.TI_OK = True
	method = 'ED'
	SCmethod = 'LSTSQ' #Don't do it self-consistently
	TI = True
	theDMET = qc_dmet.dmet( myInts, impClusters, TI, method, SCmethod )	
	RDMderivs_rot = theDMET.helper.construct1RDM_response( False, umat, None )
	
	Norb = theDMET.helper.locints.Norbs
	Nterms = theDMET.helper.Nterms
	numPairs = theDMET.helper.numPairs
	
	inH1start = theDMET.helper.H1start
	inH1row = theDMET.helper.H1row
	inH1col = theDMET.helper.H1col
	inH0 = theDMET.helper.locints.loc_rhf_fock() #+ umat
	RDMderivs_libdmet = libdmet.rhf_response(Norb, Nterms, numPairs, inH1start, inH1row, inH1col, inH0)
	inH0 = np.array( inH0.reshape( (10 * 10) ), dtype=ctypes.c_double )
	assert np.abs((RDMderivs_rot - RDMderivs_libdmet)).sum() < 1e-8
	return RDMderivs_rot, RDMderivs_libdmet
	
def test_make_H1():

	#pmDMET
	mol, mf, impClusters  = test_makemole1()
	symmetry = [0]*5  #or 'Translation'
	runDMET = dmet.DMET(mf, impClusters, symmetry, orthogonalize_method = 'overlap', schmidt_decomposition_method = 'OED', OEH_type = 'FOCK', SC_CFtype = 'FB', solver = 'RHF')
	H1start, H1row, H1col = runDMET.make_H1()[1:]
	
	#QC-DMET
	myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
	myInts.TI_OK = True
	method = 'ED'
	SCmethod = 'LSTSQ' #Don't do it self-consistently
	TI = True
	theDMET = qc_dmet.dmet( myInts, impClusters, TI, method, SCmethod )	
	
	H1start = theDMET.helper.H1start - H1start
	H1row = theDMET.helper.H1row - H1row
	H1col = theDMET.helper.H1col - H1col
	assert H1start.sum() == 0
	assert H1row.sum() == 0
	assert H1col.sum() == 0	
	
def test_1RDM_response():
	#pmDMET
	mol, mf, impClusters  = test_makemole1()
	symmetry = [0]*5  #or 'Translation'
	runDMET = dmet.DMET(mf, impClusters, symmetry, orthogonalize_method = 'meta_lowdin', schmidt_decomposition_method = 'OED', OEH_type = 'FOCK', SC_CFtype = 'FB', solver = 'RHF')

	#QC-DMET
	myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
	myInts.TI_OK = True
	method = 'ED'
	SCmethod = 'LSTSQ' #Don't do it self-consistently
	TI = True
	theDMET = qc_dmet.dmet( myInts, impClusters, TI, method, SCmethod )	
	
	
	uvec_size = runDMET.uvec.size
	uvec = np.zeros(uvec_size)
	umat = runDMET.uvec2umat(uvec)	
	
	RDMderivs_QCDMET = theDMET.helper.construct1RDM_response( False, umat, None )
	RDMderivs_pDMET = runDMET.construct_1RDM_response(uvec)	
	diff = np.abs((RDMderivs_QCDMET - RDMderivs_pDMET)).sum()
	assert np.isclose(RDMderivs_QCDMET.size, RDMderivs_pDMET.size)
	assert np.isclose(diff , 0)	
	
def test_costfunction():
	#pmDMET
	mol, mf, impClusters  = test_makemole1()
	symmetry = None  #or [0]*5, takes longer time
	runDMET = dmet.DMET(mf, impClusters, symmetry, orthogonalize_method = 'meta_lowdin', schmidt_decomposition_method = 'OED', OEH_type = 'FOCK', SC_CFtype = 'FB', solver = 'RHF')
	runDMET.one_shot()
	
	#QC-DMET
	myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
	myInts.TI_OK = False
	method = 'RHF'
	SCmethod = 'NONE' #Don't do it self-consistently
	TI = False
	theDMET = qc_dmet.dmet( myInts, impClusters, TI, method, SCmethod )	
	theDMET.doselfconsistent()
	
	uvec_size = runDMET.uvec.size
	uvec = np.random.rand(uvec_size)
	umat = runDMET.uvec2umat(uvec)
	
	CF_pmDMET = runDMET.costfunction(uvec)
	CF_QCDMET = theDMET.costfunction(uvec)
	CF_deriv_pmDMET = runDMET.costfunction_gradient(uvec)
	CF_deriv_QCDMET = theDMET.costfunction_derivative(uvec)

	assert np.isclose((CF_QCDMET - CF_pmDMET).sum(), 0)
	assert np.isclose((CF_deriv_QCDMET - CF_deriv_pmDMET).sum(), 0)
	