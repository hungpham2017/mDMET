'''
Molecular Density Matrix Embedding theory
Copyright (C) 2015 Hung Q. Pham
Author: Hung Q. Pham, Unviversity of Minnesota
email: phamx494@umn.edu
'''

import numpy as np
import scipy as scipy
from functools import reduce
from pyscf.tools import localizer
from pyscf.lo import nao, orth
from pyscf import ao2mo

class Orthobasis:
	def __init__(self, mf, method = 'overlap'):
		'''
		Prepare the orthonormal/localized set of orbitals for DMET
		Args:
			mf		: a mean-field wf
			method	: overlap/boys/lowdin/meta_lowdin
					
		Return:
			U		: Tranformation matrix to the orthonormal basis
		'''	
		
		self.mol = mf.mol 	
		self.mf = mf
		self.Nelecs = mf.mol.nelectron		
		self.Norbs = mf.mol.nao_nr()
		self.U = None		
		self.orthoOEI = None 		
		self.orthoTEI = None
		self.S = mf.get_ovlp()	
		
		if method == 'overlap':
			self.U = scipy.linalg.fractional_matrix_power(self.S, -0.5)
		elif method == 'boys':
			self.U = self.mf.mo_coeff
			loc = localizer.localizer( self.mol, self.U, method, use_full_hessian = True )
			loc.verbose = 0			
			self.U = loc.optimize( threshold = 1.e-8 )
		elif method == 'lowdin':
			ovlp = self.mol.intor('cint1e_ovlp_sph')
			ovlp_eigs, ovlp_vecs = np.linalg.eigh( ovlp )
			assert ( np.linalg.norm( np.dot( np.dot( ovlp_vecs, np.diag( ovlp_eigs ) ), ovlp_vecs.T ) - ovlp ) < 1e-10 )
			self.U = np.dot( np.dot( ovlp_vecs, np.diag( np.power( ovlp_eigs, -0.5 ) ) ), ovlp_vecs.T )
		elif method == 'meta_lowdin':
			nao.AOSHELL[4] = ['1s0p0d0f', '2s1p0d0f'] #redefine the valence shell for Be	
			self.U = orth.orth_ao( self.mol, 'meta_lowdin' )			

		#Tranform One-Electron Integral to orthonormal basis
		OEI = self.mf.get_hcore()
		self.orthoOEI = reduce(np.dot, (self.U.T, OEI, self.U))

		#Tranform Two-Electron Integral to orthonormal basis	
		Norb = self.mol.nao_nr()		
		g_format = self.mol.intor('cint2e_sph')
		TEI = np.zeros([Norb, Norb, Norb, Norb]) 
		for cn1 in range(Norb):
			for cn2 in range(Norb):
				for cn3 in range(Norb):
					for cn4 in range(Norb):
						TEI[cn1][cn2][cn3][cn4] = g_format[cn1*Norb+cn2][cn3*Norb+cn4]		
		self.orthoTEI = ao2mo.incore.full(ao2mo.restore(8, TEI, Norb), self.U, compact=False).reshape(self.Norbs, self.Norbs, self.Norbs, self.Norbs)	

		#Tranform Fock to orthonormal basis
		dm = self.mf.make_rdm1()
		vhf = self.mf.get_veff()
		FOCK = self.mf.get_fock(OEI , self.S, vhf, dm)  
		self.orthoFOCK = reduce(np.dot, (self.U.T, FOCK, self.U))
		
	def construct_orthoOED(self, umat, OEH_type):
		'''
		Construct MOs/one-electron density matrix in orthonormal basis
		with a certain correlation potential umat
		'''	
		
		#Two choices for the one-electron Hamiltonian
		if OEH_type == 'OEI':
			OEH = self.orthoOEI + umat
		elif OEH_type == 'FOCK':
			OEH = self.orthoFOCK + umat
		else:
			raise Exception('the current one-electron Hamiltonian type is not supported')

		eigenvals, eigenvecs = np.linalg.eigh(OEH)
		idx = eigenvals.argsort()
		eigenvals = eigenvals[idx]
		eigenvecs = eigenvecs[:,idx]
		nelec_pairs = self.Nelecs // 2 
		orthoOED = 2 * np.dot(eigenvecs[:,:nelec_pairs], eigenvecs[:,:nelec_pairs].T)	
		
		
		return (eigenvecs, orthoOED)
		
	def dmet_oei(self, FBEorbs, Norb_in_imp):
		oei = reduce(np.dot,(FBEorbs[:,:Norb_in_imp].T, self.orthoOEI, FBEorbs[:,:Norb_in_imp]))		
		return oei

	def dmet_tei(self, FBEorbs, Norb_in_imp):
		tei = ao2mo.incore.full(ao2mo.restore(8, self.orthoTEI, self.Norbs), FBEorbs[:,:Norb_in_imp], compact=False)
		tei = tei.reshape(Norb_in_imp, Norb_in_imp, Norb_in_imp, Norb_in_imp)
		return tei			

	def dmet_corejk(self, FBEorbs, Norb_in_imp, core1RDM_ortho):
		J = np.einsum('pqrs,rs->pq', self.orthoTEI, core1RDM_ortho)
		K = np.einsum('prqs,rs', self.orthoTEI, core1RDM_ortho)	
		jk = reduce(np.dot,(FBEorbs[:,:Norb_in_imp].T, J -0.5*K, FBEorbs[:,:Norb_in_imp]))		
		return jk
