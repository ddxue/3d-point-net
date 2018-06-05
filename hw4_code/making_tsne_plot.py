'''
CS233 Stanford, HW4

Note: Code tested with Python 2.7.

Written by Panos Achlioptas, 2018.
'''

import numpy as np
import os.path as osp 
from plt_utils import plot_2d_embedding_in_grid_greedy_way
from sklearn.manifold import TSNE

random_seed = 42 # Students use this seed if you use sklearn's TSNE with default parameters.
vanilla_ae_emb_file = '../data/out/Neural_nets/vanilla_ae/latent_codes.npz'

in_d = np.load(vanilla_ae_emb_file)    # Students: this assumes that you used np.savez 
latent_codes = in_d['l_codes']         # in the end of establishing_aes.py
test_names = in_d['test_names']

# Students: Compute 2D TSNE
tsne_lcodes = TSNE(n_components=2, random_state=random_seed).fit_transform(latent_codes)

im_files = []
top_im_dir = '../data/in/images/'
for name in test_names:
    print(name[2:-1])
    print(top_im_dir)
    im_files.append(osp.join(top_im_dir, str(name[2:-1]) + '.png'))
    
plot_2d_embedding_in_grid_greedy_way(tsne_lcodes, im_files, big_dim=1500, small_dim=70, save_file='../data/out/Neural_nets/vanilla_ae/test_pc_tsne.png', transparent=True);