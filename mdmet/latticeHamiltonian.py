'''
Multipurpose Density Matrix Embedding theory (mp-DMET)
Copyright (C) 2015 Hung Q. Pham
Author: Hung Q. Pham, Unviversity of Minnesota
email: phamx494@umn.edu
'''

import numpy as np
from functools import reduce
from pyscf import gto, scf, ao2mo
			
		
def hubbard_1D(num_sites, filling = 0.5, t = 1.0, U = 4.0, boundary_conditions = 'pbc', site_energy = None, no_hopping = None):
	'''
	One-dimensional Hubbard Hamiltonian
		H^{hat} = - Sum_{i,j,sig}(h_i,j * a^{hat,dagger}_i,sig * a^{hat}_j,sig) + Sum_{i}(U * n^{hat}_i,up * n^{hat}_i,down)
	
			o - o - o - o - o - o - o
	
	Args:
		num_sites			: numbers of sites
		boundary_conditions	: open/pbc/antipbc
		site_energy			: a list used to add a an energy to each site.
		no_hopping			: a list of pairs of site where the t = 0 (no hopping). 
	Return:
		mf					: a mean-field object
	'''
	
	if boundary_conditions == 'antipbc':
		assert (num_sites > 2)
		
	#Construct hopping part:
	h = np.zeros([num_sites,num_sites], dtype=float)
	for site in range(num_sites - 1):
		h[site, site+1] = -t
		h[site+1, site] = -t	
		
	if no_hopping != None:
		assert boundary_conditions == 'open'	
		for pair in no_hopping:
			h[pair[0], pair[1]] = 0
			h[pair[1], pair[0]] = 0				
	
	if site_energy != None: 
		assert boundary_conditions == 'open'	
		assert len(site_energy) == num_sites
		for site in range(num_sites):
			h[site, site] = site_energy[site]
		
	if	boundary_conditions == 'pbc':
		h[0, num_sites-1] = -1.0
		h[num_sites-1, 0] = -1.0
	elif boundary_conditions == 'antipbc':
		h[0, num_sites-1] = 1.0
		h[num_sites-1, 0] = 1.0
	else:
		if no_hopping != None:
			for pair in no_hopping:
				h[pair[0], pair[1]] = 0
				h[pair[1], pair[0]] = 0				
		
		if site_energy != None: 
			assert len(site_energy) == num_sites
			for site in range(num_sites):
				h[site, site] = site_energy[site]		
	
	#Construct on-site repulsion part:
	eri = np.zeros((num_sites,num_sites,num_sites,num_sites), dtype=float)
	for site in range(num_sites):
		eri[site,site,site,site] = U

	#Construct a mf object:
	mol = gto.M()
	mol.nelectron = int(2 * num_sites * filling)
	mol.nao_nr = lambda *args: num_sites
	assert( mol.nelectron % 2 == 0)		
	mf = scf.RHF(mol)
	mf.get_hcore = lambda *args: h
	mf.get_ovlp = lambda *args: np.eye(num_sites)
	mf._eri = ao2mo.restore(8, eri, num_sites)
	numPairs = mol.nelectron // 2
	
	if site_energy != None:					#For heterogeneous lattice
		mf.max_cycle = 200
		mf.kernel()
		if mf.conv_check == False: print('SCF is not converged')		
		assert(mf.mo_energy[numPairs] - mf.mo_energy[numPairs-1] > 1e-8)
	else:									#For homogeneous lattice
		eigvals, eigvecs = np.linalg.eigh(h)
		idx = eigvals.argsort()
		eigvals = eigvals[idx]
		eigvecs = eigvecs[:, idx]
		assert( eigvals[numPairs] - eigvals[numPairs-1] > 1e-8 )	#Make sure this is a gapped system
		RDM1  = 2*np.dot(eigvecs[:,:numPairs], eigvecs[:,:numPairs].T)
		JK  = np.zeros( [Lattice_size,Lattice_size], dtype=float )
		JK_value = 0.5 * U * mol.nelectron/Lattice_size
		for site in range(Lattice_size):
			JK[site,site] = JK_value
		mf.get_veff = lambda *args: JK
		mf.mo_coeff = eigvecs
		mf.mo_energy = eigvals + JK_value
	
	return mf
		
def hubbard_2D_rectangular(num_sites, filling = 0.5, t = 1.0, U = 4.0, boundary_conditions = 'pbc', site_energy = None, no_hopping = None):
	'''
	Rectangular (M,N) two-dimensional Hubbard Hamiltonian
		H^{hat} = - Sum_{i,j,sig}(h_i,j * a^{hat,dagger}_i,sig * a^{hat}_j,sig) + Sum_{i}(U * n^{hat}_i,up * n^{hat}_i,down)
		
			o - o - o - o
			|	|	|   | 
			o - o - o - o		
			|	|	|   | 	
			o - o - o - o	
	
	Args:
		num_sites			: the size of the rectangular [M,N]
		boundary_conditions	: open/pbc/antipbc
		site_energy			: a list of site index, used to add a an energy to each site.
							  Note that this is the 1D index converted from 2D index (i,j) by using (i*N + j)  
		no_hopping			: a list of pairs of site where the t = 0 (no hopping). 
	Return:
		mf					: a mean-field object
	'''
	
	#Construct hopping part:
	Nrow, Ncol = num_sites[0], num_sites[1]
	if boundary_conditions == 'antipbc':
		assert (Nrow > 2) and (Ncol>2)
	
	Lattice_size = Nrow * Ncol
	h = np.zeros([Lattice_size,Lattice_size], dtype=float)
	for x in range(Nrow):
		for y in range(Ncol):
			up 	  = x - 1 
			down  = x + 1
			left  = y - 1
			right = y + 1
			
			row = x*Ncol + y
			if up < 0: 
				if	boundary_conditions == 'pbc':
					up = up + Nrow
					h[row, up*Ncol + y] = t
				elif boundary_conditions == 'antipbc':
					up = up + Nrow
					h[row, up*Ncol + y] = -t	
			else:
				h[row, up*Ncol + y] = t
					
			if down >= Nrow: 
				if	boundary_conditions == 'pbc':
					down = down - Nrow
					h[row, down*Ncol + y] = t
				elif boundary_conditions == 'antipbc':
					down = down - Nrow
					h[row, down*Ncol + y] = -t	
			else:
				h[row, down*Ncol + y] = t	
				
			if left < 0: 
				if	boundary_conditions == 'pbc':
					left = left + Ncol
					h[row, x*Ncol + left] = t
				elif boundary_conditions == 'antipbc':
					left = left + Ncol
					h[row, x*Ncol + left] = -t	
			else:
				h[row, x*Ncol + left] = t				

			if right >= Ncol: 
				if	boundary_conditions == 'pbc':
					right = right - Ncol
					h[row, x*Ncol + right] = t
				elif boundary_conditions == 'antipbc':
					right = right - Ncol
					h[row, x*Ncol + right] = -t	
			else:
				h[row, x*Ncol + right] = t
	
	if boundary_conditions == 'open':	
		if no_hopping != None:
			for pair in no_hopping:
				h[pair[0], pair[1]] = 0
				h[pair[1], pair[0]] = 0				
		
		if site_energy != None: 
			assert len(site_energy) == Lattice_size
			for site in range(Lattice_size):
				h[site, site] = site_energy[site]
	
	#Construct on-site repulsion part:
	eri = np.zeros((Lattice_size,Lattice_size,Lattice_size,Lattice_size), dtype=float)
	for site in range(Lattice_size):
		eri[site,site,site,site] = U

	#Construct a mf object:
	mol = gto.M()
	mol.nelectron = int(2 * Lattice_size * filling)
	mol.nao_nr = lambda *args: Lattice_size
	assert( mol.nelectron % 2 == 0)		
	mf = scf.RHF(mol)
	mf.get_hcore = lambda *args: h
	mf.get_ovlp = lambda *args: np.eye(Lattice_size)
	mf._eri = ao2mo.restore(8, eri, Lattice_size)
	numPairs = mol.nelectron // 2
	
	
	if site_energy != None:					#For heterogeneous lattice
		mf.max_cycle = 200
		mf.kernel()
		if mf.conv_check == False: print('SCF is not converged')
		assert(mf.mo_energy[numPairs] - mf.mo_energy[numPairs-1] > 1e-8)
	else:									#For homogeneous lattice
		eigvals, eigvecs = np.linalg.eigh(h)
		idx = eigvals.argsort()
		eigvals = eigvals[idx]
		eigvecs = eigvecs[:, idx]
		assert( eigvals[numPairs] - eigvals[numPairs-1] > 1e-8 )	#Make sure this is a gapped system
		RDM1  = 2*np.dot(eigvecs[:,:numPairs], eigvecs[:,:numPairs].T)
		JK  = np.zeros( [Lattice_size,Lattice_size], dtype=float )
		JK_value = 0.5 * U * mol.nelectron/Lattice_size
		for site in range(Lattice_size):
			JK[site,site] = JK_value
		mf.get_veff = lambda *args: JK
		mf.mo_coeff = eigvecs
		mf.mo_energy = eigvals + JK_value
	
	return mf
	
def hubbard_2D_honeycomb(num_sites, filling = 0.5, t = 1.0, U = 4.0, boundary_conditions = 'pbc', site_energy = None, no_hopping = None):
	'''
	Honeycomb two-dimensional Hubbard Hamiltonian
		H^{hat} = - Sum_{i,j,sig}(h_i,j * a^{hat,dagger}_i,sig * a^{hat}_j,sig) + Sum_{i}(U * n^{hat}_i,up * n^{hat}_i,down)
		
			o - o - o - o
			|	|	|   | 
			o - o - o - o		
			|	|	|   | 	
			o - o - o - o	
	
	Args:
		num_sites			: the size of the rectangular [M,N]
		boundary_conditions	: open/pbc/antipbc
		site_energy			: a list of site index, used to add a an energy to each site.
							  Note that this is the 1D index converted from 2D index (i,j) by using (i*N + j)  
		no_hopping			: a list of pairs of site where the t = 0 (no hopping). 
	Return:
		mf					: a mean-field object
	'''
	
	pass