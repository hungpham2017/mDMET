'''
Molecular Density Matrix Embedding theory
ref: J. Chem. Theory Comput. 2016, 12, 2706−2719
Author: Hung Q. Pham, Unviversity of Minnesota
email: phamx494@umn.edu
'''

import numpy as np
import scipy as sp

class Bathconstruction:
	def __init__(self, mf, ImpOrbs):
		self.mf = mf
		self.ImpOrbs = ImpOrbs
		
	def RHFbath(self):
		'''
		Construct the bath using a RHF wf
		'''
		
		return 'Correct', self.ImpOrbs
		
	def UHFbath(self):
		'''
		Construct the bath using a UHF wf
		'''
		pass	
		
