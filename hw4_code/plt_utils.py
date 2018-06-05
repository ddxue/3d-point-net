'''
Created on December 26, 2016

@author: Panos Achlioptas

Note: Code tested with Python 2.7.
'''

import numpy as np

try:
    import cv2
    from PIL import Image
except:    
    raise ValueError('Missing dependencies. To use the functionality of this module pip-install 1) opencv-python & 2) PIL. \n\n 1) pip install opencv-python \n 2) pip install image\n Atlernatively, modify the loading/saving parts of it with other libraries e.g. skimage, or matplotlib.')
    

def read_transparent_png(filename):    
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:, :, 3]
    rgb_channels = image_4channel[:, :, :3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)


def _scale_2d_embedding(two_dim_emb):
    two_dim_emb -= np.min(two_dim_emb, axis=0)  # scale x-y in [0,1]
    two_dim_emb /= np.max(two_dim_emb, axis=0)
    return two_dim_emb


def plot_2d_embedding_in_grid_greedy_way(two_dim_emb, image_files, big_dim=2500, small_dim=200, save_file=None, transparent=False):
    '''
    Input:
        two_dim_emb: (N x 2) numpy array: arbitrary 2-D embedding of data.
        image_files: (list) of strings pointing to images. Specifically image_files[i] should be an image associated with
                     the datum whose coordinates are given in two_dim_emb[i].
        big_dim:     (int) height of output 'big' grid rectangular image.
        small_dim:   (int) height to which each individual rectangular image/thumbnail will be resized.
    '''
    ceil = np.ceil
    mod = np.mod
    floor = np.floor
    x = _scale_2d_embedding(two_dim_emb)
    out_image = np.ones((big_dim, big_dim, 3), dtype='uint8')
    
    if transparent:
        occupy_val = 255
        im_loader = read_transparent_png
    else:
        occupy_val = 0
        im_loader = cv2.imread
            
    out_image *= occupy_val 
    for i, im_file in enumerate(image_files):
        #  Determine location on grid
        a = ceil(x[i, 0] * (big_dim - small_dim) + 1)
        b = ceil(x[i, 1] * (big_dim - small_dim) + 1)
        a = int(a - mod(a - 1, small_dim) - 1)
        b = int(b - mod(b - 1, small_dim) - 1)
                
        if out_image[a, b, 0] != occupy_val:
            continue    # Spot already filled (drop=>greedy).
        
        fig = im_loader(im_file)
        fig = cv2.resize(fig, (small_dim, small_dim))
        try:
            out_image[a:a + small_dim, b:b + small_dim, :] = fig
        except:
            pass

        continue

    if save_file is not None:
        im = Image.fromarray(out_image)
        im.save(save_file)
    
    return out_image