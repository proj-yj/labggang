import os
import numpy as np
from ase.io import  read, write
from ase import Atoms,Atom
import glob
from collections import Counter
from tqdm import tqdm
#lists = glob.glob("*_ca")

#lists = glob.glob("/qcfs/juhwan/project/COD/materials_project_low-mpid/cif_by_type/2/v_o_copy/all_vxoy_from_dataset/*.vasp")
#lists =glob.glob("/qcfs/juhwan/project/COD/materials_project_low-mpid/cif_by_type/2/v_o_copy/slerp_redo_22/combine/v5o8/finished_k05/m*")
#lists = glob.glob("cal/vo*")
cell_max = 38
#pick_comp = ['1_1_3', '4_4_12', '2_4_8', '4_2_8' , '2_2_6', '2_2_4', '1_1_2', '4_4_8', '4_4_10', '1_4_8' , '2_4_6', '4_2_6', '2_1_4']
def make_condition(n,n_class):
    temp = np.zeros((n_class,1))
    temp[n-1,0] = 1
    return temp

def read_poscar(poscar_path, isatoms=False, atoms=None):

    if not isatoms:
        atoms = read(poscar_path)
    
    else:
        atoms = atoms
#    print(atoms)
    cell = atoms.get_cell()
    temp = atoms.get_cell_lengths_and_angles()
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_scaled_positions()
    
    return atoms, cell, symbols, pos, temp[:3], temp[3:]


def go_to_10_cell(scaled_pos,n_fe,n_o):
    cell = np.array([[10,0,0],[0,10,0],[0,0,10]]).astype(float)
    atoms = Atoms('Fe'+str(n_fe)+'O'+str(n_o))
    atoms.set_cell(cell)
    atoms.set_scaled_positions(scaled_pos)
    pos = atoms.get_positions()
    return pos

def go_to_15_cell(pos_10,n_fe,n_o):
    cell = np.array([[15,0,0],[0,15,0],[0,0,15]]).astype(float)
    pos = pos_10 + np.array([2.5,2.5,2.5])
    atoms = Atoms('Fe'+str(n_fe)+'O'+str(n_o))
    atoms.set_cell(cell)
    atoms.set_positions(pos)
    scaled_pos = atoms.get_scaled_positions()
    return scaled_pos

def make_onehot(n_class,e_pos):
    temp = np.zeros((n_class,3))
    for i,p in enumerate(e_pos):
        temp[i,:] = p
    return temp

def do_feature(atoms):

    atoms, cell, symbols, pos , lengths, angles = read_poscar(poscar_path = None, isatoms = True, atoms = atoms)
    l = lengths/30
    l = l.reshape(1,3)
    a = angles/180
    a = a.reshape(1,3)
    cell = np.vstack((l,a))
    
    n_fe = symbols.count('Fe')
    n_o = symbols.count('O')
    pos_10 = go_to_10_cell(pos,n_fe,n_o)
    scaled_pos_15 = go_to_15_cell(pos_10,n_fe,n_o)
    fe_pos = scaled_pos_15[:n_fe,:]
    o_pos = scaled_pos_15[n_fe:n_fe+n_o,:]
    fe_pos_onehot = make_onehot(40,fe_pos)
    o_pos_onehot = make_onehot(40,o_pos)
    pos_onehot = np.vstack((fe_pos_onehot,o_pos_onehot))
    temp = np.vstack((cell,pos_onehot))
    inp = temp.reshape(-1,3)
    return inp


#print Counter(symbolss)
#print max(numbers)
#print np.max(cells)
#print "n_v max is ", max(n_v_list)
#print "n_o max is ", max(n_o_list)
#np.save('gan_input', input_data)
if __name__ == "__main__":
    pass
