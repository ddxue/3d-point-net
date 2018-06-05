'''
Created on May 11, 2018

Author: Panos Achlioptas.
'''

import os
import os.path as osp
from six.moves import cPickle

def pickle_data(file_name, *args):
    '''Using (c)Pickle to save multiple python objects in a single file.
    '''
    myFile = open(file_name, 'wb')
    cPickle.dump(len(args), myFile, protocol=2)
    for item in args:
        cPickle.dump(item, myFile, protocol=2)
    myFile.close()


def unpickle_data(file_name, two_to_three=True):
    '''Restore data previously saved with pickle_data().
    '''
    inFile = open(file_name, 'rb')
    
    if two_to_three:
        size = cPickle.load(inFile, encoding='latin1')
    else:
        size = cPickle.load(inFile)
        
    for _ in range(size):
        if two_to_three:        
            yield cPickle.load(inFile, encoding='latin1')
        else:
            yield cPickle.load(inFile)
    inFile.close()


def create_dir(dir_path):
    ''' Creates a directory (or nested directories) if they don't exist.
    '''
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path
