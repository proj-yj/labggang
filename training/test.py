import numpy as np
from view_atoms_gan import *
import ase
from tqdm import tqdm

data_path = './gen_image_cwgan_mgmno'
dat = np.load(data_path+'/gen_images_500.npy')
print(dat)
print(np.shape(dat))

def check(image):
	# pos = image[2:,:]
	ga = image[2:12,:]
	n = image[12:,:]
	gaga = np.sum(ga,axis=1)
	gagaga = np.zeros((10,1)) + 1
	# gagaga = np.zeros_like(gaga) + 1
	gagaga[gaga<0.4] = 0
	nn = np.sum(n,axis=1)
	nnn = np.zeros((10,1)) +1
	# nnn = np.zeros_like(nn) +1
	nnn[nn<0.4] = 0
	label = np.vstack((gagaga,nnn))
	print(label.shape)
	return label

m = dat.shape[0]
output = []
for i in tqdm(range(m)):
	x, = dat[i]
	label = check(x)
	# print(x)
	# print(label)
	new_input = (x,label)

	output.append(new_input)
np.save("label_gen" ,output)

'''
image = dat[0]
ax, alabel=output[0]
ga_label, n_label=alabel[:10], alabel[10:]
print(np.shape(ga_label))
print(np.shape(n_label))
view_atoms_classifier(image, torch.Tensor(ga_label), torch.Tensor(n_label), 0)
'''

z=0
while z<900:
	print(z)
	image = dat[z]
	ax, alabel=output[z]
	ga_label, n_label=alabel[:10], alabel[10:]
	# position =view_atoms_classifier(image, torch.Tensor(ga_label), torch.Tensor(n_label), 0)
	position =view_atoms_classifier(image, ga_label, n_label, 0)

	print(position)
	if position[-2] == 0:
		ase.io.write(f"POSCARS/POSCAR_{z}",position[0], "vasp")

	z+=1
    