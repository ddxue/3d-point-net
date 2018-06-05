'''
CS 233, HW4

Written by Panos Achlioptas, 2018.
'''

import tensorflow as tf
import numpy as np
import os.path as osp
from six import string_types

from tf_util import conv1d
from tf_util import fully_connected as fc

def expand_scope_by_name(scope, name):
    """ expand_scope_by_name.
    """

    if isinstance(scope, string_types):
        scope += osp.sep + name
        return scope

    if scope is not None:
        return scope.name + osp.sep + name
    else:
        return name


def replicate_parameter_for_all_layers(parameter, n_layers):
    if parameter is not None and len(parameter) != n_layers:
        if len(parameter) != 1:
            raise ValueError()
        parameter = np.array(parameter)
        parameter = parameter.repeat(n_layers).tolist()
    return parameter


def encoder_with_convs_and_symmetry(in_layer, n_filters, filter_sizes, scope=None, verbose=False):
    # Student's TODO
    if verbose:
        print ('Building Encoder')    

        
    for i in range(n_filters):
        name = 'encoder_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        if i == 0:
            layer = in_layer
        layer = conv1d(layer, filter_sizes[i], kernel_size=1, scope=scope_i)

        if i == 2:
            keep_layer = layer

    latent_vec = tf.reduce_max(layer, axis=1)

    n_pc_per_model = keep_layer.get_shape()[1]
    dim_latent_vec = latent_vec.get_shape()[1]

    latent_vec_repeated = tf.reshape(tf.tile(latent_vec, [1, n_pc_per_model]), [-1, n_pc_per_model, dim_latent_vec])
    combo_latent_pt = tf.concat([keep_layer, latent_vec_repeated], axis=-1)

    return latent_vec, combo_latent_pt
        
        
        
    
def decoder_with_fc_only(latent_signal, layer_sizes=[], b_norm=False, non_linearity=tf.nn.relu,
                         weight_decay=0.0, scope=None, dropout_prob=None,
                         verbose=False):
    ''' Note:  dropout, b_norm, weight_decay are dummy input names, but can be 
    usefule in the bonus.
    '''
    
    if verbose:
        print ('Building Decoder')

    n_layers = len(layer_sizes)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)
    
    if n_layers < 2:
        raise ValueError('For an FC decoder with single a layer use simpler code.')

    for i in range(0, n_layers - 1):
        name = 'decoder_fc_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        if i == 0:
            layer = latent_signal
        
        layer = fc(layer, layer_sizes[i], scope_i, activation_fn=non_linearity)
                   
    
    name = 'decoder_fc_' + str(n_layers - 1)
    scope_i = expand_scope_by_name(scope, name)
    layer = fc(layer, layer_sizes[n_layers - 1], scope_i, activation_fn=tf.identity)
    
    return layer
