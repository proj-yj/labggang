import numpy as np
from ase import Atoms
from ase.io import read,write
import sys
import torch

def back_to_10_cell(scaled_pos,n_fe,n_o):
	cell = np.identity(3)*15
	atoms = Atoms('Fe'+str(n_fe)+'O'+str(n_o))
	atoms.set_cell(cell)
	atoms.set_scaled_positions(scaled_pos)
	pos = atoms.get_positions()	

	cell = np.identity(3)*10
	pos = pos - np.array([2.5,2.5,2.5])
	atoms = Atoms('Fe'+str(n_fe)+'O'+str(n_o))
	atoms.set_cell(cell)
	atoms.set_positions(pos)
	scaled_poss = atoms.get_scaled_positions()
	return scaled_poss

def back_to_real_cell(scaled_pos, real_cell, n_fe,n_o):
	atoms = Atoms('Fe'+str(n_fe)+'O'+str(n_o))
	atoms.set_cell(real_cell)
	atoms.set_scaled_positions(scaled_pos)
	return atoms

def remove_zero_padding(pos):
    criteria = 0.4
    fe_pos = pos[:40,:]
    o_pos = pos[40:,:]
    
    fe = np.sum(fe_pos, axis=1)
    o = np.sum(o_pos, axis=1)
    fe_index = np.where(fe > criteria)
    o_index = np.where(o > criteria)
    
    n_fe = len(fe_index[0])
    n_o = len(o_index[0])
    fe_pos = fe_pos[fe_index]
    o_pos = o_pos[o_index]
	
    if n_fe == 0:
        fe_pos = np.array([0.1667,0.1667,0.1667]).reshape(1,3)
        n_fe = 1
        
    if n_o == 0:
        o_pos = np.array([0.1667,0.1667,0.1667]).reshape(1,3)
        n_o = 1
        
    pos = np.vstack((fe_pos,o_pos))
    
    return pos, n_fe,n_o

def view_atoms(image, view = True):
	x = image
	x = x.reshape(-1,3)
	l = x[0,:]*30
	a = x[1,:]*180
	cell = np.hstack((l,a))
	pos=x[2:,:]
	pos,n_fe, n_o = remove_zero_padding(pos)
	scaled_pos = back_to_10_cell(pos,n_fe,n_o)
	atoms = back_to_real_cell(scaled_pos, cell, n_fe,n_o)
	atoms.set_pbc([1,1,1])
	if view:
		atoms.edit()
	return atoms, x

def view_atoms_classifier(image,fe_label,o_label, view=True):
	x= image.reshape(-1,3)
	fe = x[2:42,:]
	o = x[42:,:]

	l = x[0,:]*30
	a = x[1,:]*180
	c = np.hstack((l,a))
	# print(c)
	atoms = Atoms('H')
	try:
		atoms.set_cell(c)
		cell = atoms.get_cell()
		t = np.isnan(cell)
		tt = np.sum(t)
		isnan = False
		if not tt == 0:
			isnan = True
			# print(cell)
			# print(l)
			# print(a)
		# _,fe_index = torch.max(fe_label,dim=1)
		# _,o_index = torch.max(o_label,dim=1)
		
		# fe_index = fe_index.reshape(10,).detach().cpu().numpy()
		# o_index = o_index.reshape(10,).detach().cpu().numpy()
		
		z=0
		fe_pos=[]
		while z<len(fe_label):
			if fe_label[z]==1:
				fe_pos.append(fe[z,:])
			else: pass
			z+=1
		fe_pos=np.array(fe_pos)
		
		z=0
		o_pos=[]  #np.zeros_like(o)
		while z<len(o_label):
			if o_label[z]==1:
				o_pos.append(o[z,:])
			else: pass
			z+=1
		o_pos=np.array(o_pos)



		# print(np.shape(fe))
		# fe_pos = fe[np.where(fe_label==1)]
		# print(np.shape(fe_pos))
		# o_pos = o[np.where(o_label==1)]
		
		print(fe_label)
		n_fe = int(np.sum(fe_label))
		n_o = int(np.sum(o_label))
		print(n_fe)
		print(n_o)
		
		# if n_fe == 0:
		# 	fe_pos = np.array([0.1667,0.1667,0.1667]).reshape(1,3)
		# 	n_fe = 1
		# if n_o == 0:
		# 	o_pos = np.array([0.1667,0.1667,0.1667]).reshape(1,3)
		# 	n_o = 1
		if n_fe == 0 or n_o ==0:
			isnan = True
		
		pos = np.vstack((fe_pos,o_pos))
		print(np.shape(pos))
		scaled_pos = back_to_10_cell(pos,n_fe,n_o)
		atoms = back_to_real_cell(scaled_pos, cell, n_fe,n_o)
		atoms.set_pbc([1,1,1])
		if view :
			atoms.edit()
	except AssertionError:
		isnan = True
		atoms, x, c = [], [], []
			
	return atoms, x, isnan, c

# if '__name__' == '__main__':
#         pass
# else:
#         print("import")


