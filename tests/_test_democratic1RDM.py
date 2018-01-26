'''
Testing the self_consistent DMET
'''

import sys
import pyscf
from pyscf import gto, scf, ao2mo
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

def test_makemole():
	bondlength = 1.0
	nat = 10
	mol = gto.Mole()
	mol.atom = []
	r = 0.5 * bondlength / np.sin(np.pi/nat)
	for i in range(nat):
		theta = i * (2*np.pi/nat)
		mol.atom.append(('H', (r*np.cos(theta), r*np.sin(theta), 0)))

	mol.basis = 'sto-3g'
	mol.build(verbose=0)

	mf = scf.RHF(mol)
	mf.scf()
	impOrbs = np.zeros(mol.nao_nr())
	impOrbs[-numBathOrbs:] = 1

	return mol, mf, impOrbs 

def test_democratic_1RDM():
	numBathOrbs = 2
	mol, mf, impOrbs  = test_makemole()
	nocc = mol.nelectron // 2
	
	umat = np.zeros((mol.nao_nr(), mol.nao_nr()))
	ortho = orthobasis.Orthobasis(mf, method = 'overlap')
	orthoOED = ortho.construct_orthoOED(umat, OEH_type = 'FOCK')
	schmidt = schmidtbasis.RHF_decomposition(mf, impOrbs, numBathOrbs, orthoOED)	

	
	schmidt.method = 'OED'	
	BathOrbs, FBEorbs, core_eigenvals = schmidt.baths()
	
	OEIortho = ortho.orthoOEI
	TEIortho = ortho.orthoTEI
	Norb = ortho.Norbs
	numAct = 2*numBathOrbs
	
	orthoOED_core = reduce(np.dot, (FBEorbs, np.diag(core_eigenvals), FBEorbs.T))
	JKcore = np.einsum('pqrs,rs->pq', TEIortho, orthoOED_core) - 0.5*np.einsum('prqs,rs->pq', TEIortho, orthoOED_core)	
	
	Etest = 0.5* (orthoOED_core * (2*OEIortho + JKcore)).sum()


	JKdmet = reduce(np.dot,(FBEorbs[:,:numAct].T, JKcore, FBEorbs[:,:numAct]))
	OEIdmet = reduce(np.dot,(FBEorbs[:,:numAct].T, OEIortho, FBEorbs[:,:numAct]))
	TEIdmet = ao2mo.incore.full(ao2mo.restore(8, TEIortho, Norb), FBEorbs[:,:numAct], compact=False).reshape(numAct, numAct, numAct, numAct)
	
	Nel, Nimp = 4, 2
	chempot= 0.0
	DMguess = None
	solver = qcsolvers.QCsolvers(OEIdmet, TEIdmet, JKdmet, DMguess, numAct, Nel, Nimp, chempot)
	Eimp, Eemb, OED = solver.RHF()
	
'''
numBathOrbs = 2
mol, mf, impOrbs  = test_makemole()
nocc = mol.nelectron // 2

umat = np.zeros((mol.nao_nr(), mol.nao_nr()))
ortho = orthobasis.Orthobasis(mf, method = 'overlap')
orthoOED = ortho.construct_orthoOED(umat, OEH_type = 'FOCK')
schmidt = schmidtbasis.RHF_decomposition(mf, impOrbs, numBathOrbs, orthoOED)


schmidt.method = 'OED'
BathOrbs, FBEorbs, core_eigenvals = schmidt.baths()

OEIortho = ortho.orthoOEI
TEIortho = ortho.orthoTEI
Norb = ortho.Norbs
numAct = 2*numBathOrbs

orthoOED_core = reduce(np.dot, (FBEorbs, np.diag(core_eigenvals), FBEorbs.T))
JKcore = np.einsum('pqrs,rs->pq', TEIortho, orthoOED_core) - 0.5*np.einsum('prqs,rs->pq', TEIortho, orthoOED_core)

Etest = 0.5* (orthoOED_core * (2*OEIortho + JKcore)).sum()


JKdmet = reduce(np.dot,(FBEorbs[:,:numAct].T, JKcore, FBEorbs[:,:numAct]))
OEIdmet = reduce(np.dot,(FBEorbs[:,:numAct].T, OEIortho, FBEorbs[:,:numAct]))
TEIdmet = ao2mo.incore.full(ao2mo.restore(8, TEIortho, Norb), FBEorbs[:,:numAct], compact=False).reshape(numAct, numAct, numAct, numAct)

Nel, Nimp = 4, 2
chempot= 0.0
DMguess = None
solver = qcsolvers.QCsolvers(OEIdmet, TEIdmet, JKdmet, DMguess, numAct, Nel, Nimp, chempot)
Eimp, Eemb, OED = solver.RHF()
Ecal = Etest +  Eemb

embOED = reduce(np.dot,(FBEorbs[:,:numAct], OED, FBEorbs[:,:numAct].T))
impOED = reduce(np.dot,(FBEorbs[:,:2], OED[:2,:2], FBEorbs[:,:2].T))
bathOED = reduce(np.dot,(FBEorbs[:,2:numAct], OED[2:,2:], FBEorbs[:,2:numAct].T))
totalOED = embOED + orthoOED_core		#it should be equal to D in orthogonal basis
mfD = mf.make_rdm1().dot(mf.get_ovlp())
impD = reduce(np.dot,(FBEorbs[:,:2].T, mfD, FBEorbs[:,:2]))
bathD = reduce(np.dot,(FBEorbs[:,2:4].T, mfD, FBEorbs[:,2:4]))
envD = reduce(np.dot,(FBEorbs[:,4:].T, mfD, FBEorbs[:,4:]))
'''
