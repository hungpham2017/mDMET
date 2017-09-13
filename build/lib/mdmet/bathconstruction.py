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
from pyscf.tools import localizer
from pyscf import ao2mo

class Baths:
	def __init__(self, mol, mf, impOrbs, orthonormalize = 'overlap', localization_threshold = 1.e-6):
		self.mol = mol 	
		self.mf = mf 
		self.Norbs = mol.nao_nr()
		self.impOrbs = impOrbs
		self.U = None	
		self.orthoMO = None		
		self.orthoOEI = None 		
		self.orthoTEI = None

		
		'''---------------------------------------------------------------------------------
		Prepare the orthonormal set of orbitals for bath construction
		in:
			mf		: a mean-field wf
			method	: overlap/boys/lowdin
					
		out:
			U		: Tranformation matrix to the orthonormal basis
		---------------------------------------------------------------------------------'''
		
		if orthonormalize == 'overlap':
			S = self.mf.get_ovlp()
			self.U = scipy.linalg.fractional_matrix_power(S, -0.5)
			
		elif orthonormalize == 'boys':
			self.U = self.mf.mo_coeff
			loc = localizer.localizer( self.mol, self.U, orthonormalize, use_full_hessian = True )
			self.U = loc.optimize( threshold = localization_threshold )
			
		elif orthonormalize == 'lowdin':
			ovlp = self.mol.intor('cint1e_ovlp_sph')
			ovlp_eigs, ovlp_vecs = np.linalg.eigh( ovlp )
			assert ( np.linalg.norm( np.dot( np.dot( ovlp_vecs, np.diag( ovlp_eigs ) ), ovlp_vecs.T ) - ovlp ) < 1e-10 )
			self.U = np.dot( np.dot( ovlp_vecs, np.diag( np.power( ovlp_eigs, -0.5 ) ) ), ovlp_vecs.T )

		#Tranform MO coefficients to orthogonal space
		MO = self.mf.mo_coeff
		self.orthoMO = np.linalg.inv(self.U).dot(MO)	

		#Tranform One-Electron Integral to orthogonal space
		OEI = self.mf.get_hcore()
		self.orthoOEI = reduce(np.dot, (self.U.T, OEI, self.U))

		#Tranform Two-Electron Integral to orthogonal space	
		Norb = self.mol.nao_nr()		
		g_format = self.mol.intor('cint2e_sph')
		TEI = np.zeros([Norb, Norb, Norb, Norb]) 
		for cn1 in range(Norb):
			for cn2 in range(Norb):
				for cn3 in range(Norb):
					for cn4 in range(Norb):
						TEI[cn1][cn2][cn3][cn4] = g_format[cn1*Norb+cn2][cn3*Norb+cn4]		
		self.orthoTEI = ao2mo.incore.full(ao2mo.restore(8, TEI, Norb), self.U, compact=False).reshape(self.Norbs, self.Norbs, self.Norbs, self.Norbs)		
		
		
	def RHFbath(self, numBathOrbs, threshold = 1e-6):
		'''
		Construct the bath using a RHF wf
		ref: PHYSICAL REVIEW B 89, 035140 (2014)
		'''

		# Build the projectors for fragment and bath
		S = self.mf.get_ovlp() 
		X = scipy.linalg.sqrtm(S)
		Forbs = X[:, self.impOrbs == 1]
		SFF_inv = np.linalg.inv(S[:,self.impOrbs == 1][self.impOrbs == 1,:])
		P_F = reduce(np.dot,(Forbs, SFF_inv,Forbs.T))
		P_B = np.identity(P_F.shape[0])- P_F
		
		
		
		# Build the overlap matrix between hole states and fragment orbs
		UOcc = self.orthoMO[:, self.mf.mo_occ!=2]		
		Occ = self.orthoMO[:, self.mf.mo_occ==2]
		M = reduce(np.dot,(Occ.T, P_F,Occ))
		d, V = np.linalg.eigh(M) 
		idx = np.maximum( -d, d - 1.0 ).argsort() #the eigenvalues close to 1 come first
		d, V = d[idx], V[:, idx] 
		tokeep = np.sum( -np.maximum( -d, d - 1.0 ) > threshold )
		if ( tokeep < numBathOrbs ):
			print ("BATH CONSTRUCTION: using only ", tokeep, " orbitals which are within ", threshold, " of 0 or 1")
		numBathOrbs = min(tokeep, numBathOrbs)
		
		Forbs = np.einsum('pi,up->ui',V[:,:numBathOrbs], np.dot(P_F, Occ))/np.sqrt(d[:numBathOrbs])
		Borbs = np.einsum('pi,up->ui',V[:,:numBathOrbs], np.dot(P_B, Occ))/np.sqrt(1 - d[:numBathOrbs])
		FBorbs = np.hstack((Forbs, Borbs))
		
		Smith_orbs = np.einsum('pi,up->ui', V, Occ)
		Smith_orbs = np.hstack((Smith_orbs, UOcc))		
		Envorbs = Smith_orbs[:,numBathOrbs:]
		Psmith = FBorbs.dot(FBorbs.T)
			
		return FBorbs, Smith_orbs
		
	def UHFbath(self):
		'''
		Construct the bath using a UHF wf
		'''
		pass	
		
