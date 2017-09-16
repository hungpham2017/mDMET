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
from pyscf import gto, scf, mcscf, ao2mo
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
		
		Nimp = self.Nimp
		FOCK = self.FOCK.copy()
		
		if (self.chempot != 0.0):
			for orb in range(Nimp):
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
		DMloc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
		if ( mf.converged == False ):
			mf = rhf_newtonraphson.solve( mf, dm_guess=DMloc)
			DMloc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
		
		ERHF = mf.e_tot
		RDM1 = mf.make_rdm1()
		jk   = mf.get_veff(None, dm=RDM1)
	 
		# To calculate the impurity energy, rescale the JK matrix with a factor 0.5 to avoid double counting: 0.5 * ( OEI + FOCK ) = OEI + 0.5 * JK
		ImpurityEnergy = 0.5*np.einsum('ij,ij->', RDM1[:Nimp,:], self.OEI[:Nimp,:] + FOCK[:Nimp,:]) \
						+ 0.5*np.einsum('ij,ij->', RDM1[:Nimp,:], jk[:Nimp,:])
						
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

	def CAS(self, CAS, CAS_MO, Orbital_optimization = False):
		'''
		Complete Active Space Configuration Interaction (CASCI) or Complete Active Space Self-Consisten Field (CASSCF)
		Orbital_optimization = False : CASCI
		Orbital_optimization = True  : CASSCF
		'''		
		
		Norb = self.Norb
		Nimp = self.Nimp
		FOCK = self.FOCK.copy()
		
		if (self.chempot != 0.0):
			for orb in range(Nimp):
				FOCK[orb, orb] -= self.chempot	
		
		mol = gto.Mole()
		mol.build(verbose = 0)
		mol.atom.append(('C', (0, 0, 0)))
		mol.nelectron = self.Nel
		mol.incore_anyway = True
		mf = scf.RHF( mol )
		mf.get_hcore = lambda *args: FOCK
		mf.get_ovlp = lambda *args: np.eye(Norb)
		mf._eri = ao2mo.restore(8, self.TEI, Norb)
		mf.scf(self.DMguess)
		DMloc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
		if ( mf.converged == False ):
			mf = rhf_newtonraphson.solve( mf, dm_guess=DMloc)
			DMloc = np.dot(np.dot(mf.mo_coeff, np.diag(mf.mo_occ)), mf.mo_coeff.T)
		if CAS == None:
			CAS_nelec = self.Nel
			CAS_norb = Norb
			CAS = 'full'
		else:
			CAS_nelec = CAS[0]
			CAS_norb = CAS[1]
		print("     Active space: ", CAS)
		
		if Orbital_optimization == True: 
			mc = mcscf.CASSCF(mf, CAS_norb, CAS_nelec)	
		else:
			mc = mcscf.CASCI(mf, CAS_norb, CAS_nelec)
		if CAS_MO is not None: 
			print("     Active space MOs: ", CAS_MO)
			mo = mc.sort_mo(CAS_MO)
			ECASSCF = mc.kernel(mo)[0]
		else:
			ECASSCF = mc.kernel()[0]	
		
		###### Get RDM1 + RDM2 #####
		CAS_norb = mc.ncas
		core_norb = mc.ncore
		CAS_nelec = mc.nelecas	
		core_MO = mc.mo_coeff[:,:core_norb]
		CAS_MO = mc.mo_coeff[:,core_norb:core_norb+CAS_norb]

	
		casdm1 = mc.fcisolver.make_rdm1(mc.ci, CAS_norb, CAS_nelec) #in CAS space
		# Transform the casdm1 (in CAS space) to casdm1ortho (orthonormal space).     
		casdm1ortho = np.einsum('ap,pq->aq', CAS_MO, casdm1)
		casdm1ortho = np.einsum('bq,aq->ab', CAS_MO, casdm1ortho)
		coredm1 = np.dot(core_MO, core_MO.T) * 2 #in localized space
		RDM1 = coredm1 + casdm1ortho	

		casdm2 = mc.fcisolver.make_rdm2(mc.ci, CAS_norb, CAS_nelec) #in CAS space
		# Transform the casdm2 (in CAS space) to casdm2ortho (orthonormal space). 
		casdm2ortho = np.einsum('ap,pqrs->aqrs', CAS_MO, casdm2)
		casdm2ortho = np.einsum('bq,aqrs->abrs', CAS_MO, casdm2ortho)
		casdm2ortho = np.einsum('cr,abrs->abcs', CAS_MO, casdm2ortho)
		casdm2ortho = np.einsum('ds,abcs->abcd', CAS_MO, casdm2ortho)	
	
		coredm2 = np.zeros([Norb, Norb, Norb, Norb]) #in AO
		coredm2 += np.einsum('pq,rs-> pqrs',coredm1,coredm1)
		coredm2 -= 0.5*np.einsum('ps,rq-> pqrs',coredm1,coredm1)
	
		effdm2 = np.zeros([Norb, Norb, Norb, Norb]) #in AO
		effdm2 += 2*np.einsum('pq,rs-> pqrs',casdm1ortho,coredm1)
		effdm2 -= np.einsum('ps,rq-> pqrs',casdm1ortho,coredm1)				
					
		RDM2 = coredm2 + casdm2ortho + effdm2

		ImpurityEnergy = 0.50  * np.einsum('ij,ij->',     RDM1[:Nimp,:],     FOCK[:Nimp,:] + self.OEI[:Nimp,:]) \
                       + 0.125 * np.einsum('ijkl,ijkl->', RDM2[:Nimp,:,:,:], self.TEI[:Nimp,:,:,:]) \
                       + 0.125 * np.einsum('ijkl,ijkl->', RDM2[:,:Nimp,:,:], self.TEI[:,:Nimp,:,:]) \
                       + 0.125 * np.einsum('ijkl,ijkl->', RDM2[:,:,:Nimp,:], self.TEI[:,:,:Nimp,:]) \
                       + 0.125 * np.einsum('ijkl,ijkl->', RDM2[:,:,:,:Nimp], self.TEI[:,:,:,:Nimp])		
	
		return (ImpurityEnergy, ECASSCF, RDM1)