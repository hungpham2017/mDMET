'''
Molecular Density Matrix Embedding theory
ref: 
J. Chem. Theory Comput. 2016, 12, 2706−2719
PHYSICAL REVIEW B 89, 035140 (2014)
Author: Hung Q. Pham, Unviversity of Minnesota
email: phamx494@umn.edu
'''

import numpy as np
import scipy as scipy
from functools import reduce
import pyscf
from pyscf import gto, scf, ao2mo
from pyscf.tools import rhf_newtonraphson

class QCsolvers:
	def __init__(self, OEI, TEI, JK, DMguess, Norb, Nel, Nimp, chempot = 0.0):
		self.OEI = OEI
		self.TEI = TEI
		self.FOCK = OEI + JK
		self.DMguess = DMguess
		self.Norb = Norb
		self.Nel = Nel
		self.Nimp = Nimp
		self.chempot = chempot
		
	def RHF(self):
		'''
		Restricted Hartree-Fock
		'''		
		FOCK = self.FOCK.copy()

		if (self.chempot != 0.0):
			for orb in range(self.Nimp):
				FOCK[orb, orb] -= self.chempot	
		
		mol = gto.Mole()
		mol.build(verbose = 0)
		mol.atom.append(('C', (0, 0, 0)))
		mol.nelectron = self.Nel
		mol.incore_anyway = True
		mf = scf.RHF( mol )
		mf.get_hcore = lambda *args: FOCK
		mf.get_ovlp = lambda *args: np.eye(self.Norb)
		mf._eri = ao2mo.restore(8, self.TEI, self.Norb)
		mf.scf(self.DMguess)
		DMloc = np.dot(np.dot(mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T)
		if ( mf.converged == False ):
			mf = rhf_newtonraphson.solve( mf, dm_guess=DMloc)
			DMloc = np.dot(np.dot( mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
		
		ERHF = mf.e_tot
		RDM1 = mf.make_rdm1()
		jk   = mf.get_veff(None, dm=RDM1)
	 
		# To calculate the impurity energy, rescale the JK matrix with a factor 0.5 to avoid double counting: 0.5 * ( OEI + FOCK ) = OEI + 0.5 * JK
		ImpurityEnergy = 0.5*np.einsum('ij,ij->', RDM1[:self.Nimp,:], self.OEI[:self.Nimp,:] + FOCK[:self.Nimp,:]) \
						+ 0.5*np.einsum('ij,ij->', RDM1[:self.Nimp,:], jk[:self.Nimp,:])
						
		return (ImpurityEnergy, ERHF, RDM1)
		
	def UHF(self):
		'''
		Unrestricted Hartree-Fock
		'''		
		pass		
		
	def CASSCF(self):
		'''
		Complete-active Space Self-consistent Field (CASSCF)
		'''		
		pass

	def CCSD(self):
		'''
		Couple-cluster Singly-Doubly 
		'''		
		pass			
		
	def DMRG(self):
		'''
		Density Matrix Renormalization Group
		'''		
		pass	

	def FCI(self):
		'''
		Full Configuration Interaction (FCI)
		'''		
		pass