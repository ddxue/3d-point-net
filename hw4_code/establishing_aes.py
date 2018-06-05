'''
Standford CS233, HW4.

Note: Code tested with Python 2.7, adapted to work with Python 3.0 too.

Written by Panos Achlioptas, 2018.
'''

import tensorflow as tf
import numpy as np
import os.path as osp
import matplotlib.pylab as plt
import six

import sys
sys.path.insert(0, "/home/davidxue/notebooks/tensorflow/cs233_hw4_2018")
print(sys.path)

from hm4_code.numpy_dataset import NumpyDataset
from hm4_code.cs233_point_auto_encoder import cs233PointAutoEncoder
from hm4_code.neural_net import Neural_Net_Conf
from hm4_code.encoders_decoders import encoder_with_convs_and_symmetry, decoder_with_fc_only
from hm4_code.in_out_utils import unpickle_data, pickle_data, create_dir




# Students: If you run your code with Python3 instead of Python2 (like in Azzure) set python_2to3=True
python_2to3 = False


# Students: Default options for splits. Do NOT change.
split_loads = [0.75, 0.15, 0.10]
random_seed = 42
verbose = True
n_total_shapes = 1000

# Training options.
do_training = True
batch_size = 50        # Students: do NOT change this one.
held_out_step = 10
n_epochs = 400

# Loading DATA/employing train/val/test splits.
generator = unpickle_data('data/in/part_labeled_point_clouds.pkl', python_2to3)
hw4_data = six.next(generator)
sids = six.next(unpickle_data('data/in/randomized_ids.pkl', python_2to3))
hw4_data.part_masks = hw4_data.part_masks.astype(np.int32)

hw4_data = hw4_data.subsample(n_total_shapes, replace=False, seed=random_seed)
net_data = {}
for s in sids:
    idx = sids[s]
    net_data[s] = hw4_data.extract(idx)
test_data = net_data['test'].freeze()


# Define Configuration of the Point-AE.
pc_ae_conf = Neural_Net_Conf()

n_pc_per_model = net_data['train'].pcs.shape[-2]
bneck = 128

pc_ae_conf.encoder = encoder_with_convs_and_symmetry
pc_ae_conf.decoder = decoder_with_fc_only

pc_ae_conf.n_points = n_pc_per_model

pc_ae_conf.encoder_args = {'n_filters':5, 'filter_sizes':[32, 64, 64, 128, 128],
                          'verbose':True} # Students add your encoder's options.

pc_ae_conf.decoder_args = {'layer_sizes': [256, 256, n_pc_per_model * 3],                           
                           'verbose': False}

pc_ae_conf.learning_rate = 0.0009
pc_ae_conf.saver_max_to_keep = 1
pc_ae_conf.allow_gpu_growth = True


# Students: Will it predict part-segmentation too? If so, set to true. 
pc_ae_conf.use_parts = False
pc_ae_conf.n_parts = 4
pc_ae_conf.part_pred_with_one_layer = True


# How much is the relative importance of part-prediction vs. pc-reconstruction.
pc_ae_conf.part_weight = 0.005 #Students: leave this option unchanged for the (non-bonus) questions.

if pc_ae_conf.use_parts:
    if pc_ae_conf.part_pred_with_one_layer:
        pc_ae_conf.name = 'pc_aware_ae'
    else:
        pc_ae_conf.name = 'pc_aware_ae_bonus'
    n_losses = 2
else:
    pc_ae_conf.name = 'vanilla_ae'
    n_losses = 1


# Establish tensor-flow graph.
ae = cs233PointAutoEncoder(pc_ae_conf.name, pc_ae_conf)


# if do_training:
#     save_dir = create_dir(osp.join('../data/out/Neural_nets', pc_ae_conf.name))    
#     tf.set_random_seed(random_seed)
#     np.random.seed(random_seed)
#     with open(osp.join(save_dir, 'net_stats.txt'), 'w') as file_out:            
#         train_loss, val_loss, test_loss = ae.train_model(net_data, n_epochs, batch_size, save_dir,\
#                                                          held_out_step, fout=file_out)


# Done Training? Congrats!

# If you want to resume training it is possible, but be-careful to NOT over-write 
# the saved pre-trained model!



# # Load model on optimal (per validation) epoch.
# epoch_to_restore = np.argmin(val_loss) # Student 
# ae.restore_model(save_dir, epoch_to_restore, verbose=True)

# # Students: Save-plot reconstructions.
# n_plots = 5
# in_pc = test_data.pcs[:n_plots]
# in_names = test_data.model_names[:n_plots]
# in_masks = test_data.part_masks[:n_plots]

# # Students: Save-plot masks of predictions, report accuracy.
# if pc_ae_conf.use_parts:
#     feed = ae.prepare_feed(test_data, test_data.n_examples)
#     test_accuracy = ae.sess.run(ae.acc_by_parts, feed_dict=feed)
#     print(test_accuracy)
    
# # Extract and save latent codes of test chairs.
# test_pcs = test_data.pcs
# test_names = test_data.model_names
# # l_codes = # Students: compute this.
# np.savez(osp.join(save_dir, 'latent_codes'), l_codes=l_codes, test_names=test_names)