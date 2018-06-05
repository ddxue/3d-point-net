'''
CS233 Stanford, HW4

Note: Code tested with Python 2.7.

Written by Panos Achlioptas, 2018.
'''

import numpy as np
import os.path as osp
from collections import defaultdict

def find_nearest_neighbors(X, k=10):
    # Students.
    pass
    
    
vanilla_ae_emb_file = '../data/out/Neural_nets/vanilla_ae/latent_codes.npz'
part_ae_emb_file = '../data/out/Neural_nets/pc_aware_ae/latent_codes.npz'
golden_part_dist_file = '../data/in/golden_dists.npz'

golden_data = np.load(golden_part_dist_file)
golden_part_dist = golden_data['golden_part_dist']
golden_names = golden_data['golden_names']

# To load vanilla-AE-embeddings (if False will open those of the 2-branch AE).
vanilla = True

#  Students. Extract shape-net model ids of golden, map them to their parts.
sn_id_to_parts = defaultdict(list)


# Students. Map shape-net model id and part_id to location in distance matrix.
id_to_part_loc = dict()


    

if vanilla:
    in_d = np.load(vanilla_ae_emb_file)    # Students: this assumes that you used np.savez 
    latent_codes = in_d['l_codes']         # in the end of establishing_aes.py to save the
    test_names = in_d['test_names']        # latent_codes/test_names.
else:
    in_d = np.load(part_ae_emb_file)
    latent_codes = in_d['l_codes']
    test_names = in_d['test_names']


# Students. Implement Question [g].


print('Average Holistic distance:', 0)
print('Average Aggrement: ', 0)
print('Average Cumulative Dists:', 0)