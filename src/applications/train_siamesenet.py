"""
"""

import argparse
import os

import h5py
import keras.backend.tensorflow_backend as ktf
import tensorflow as tf

from applications.config import get_siamese_config
from applications.siamesenet import run_net
from core.data import build_siamese_data

# add directories in src/ to path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='')
parser.add_argument('--gpu_memory_fraction', type=float, help='gpu percentage to use', default='1.0')
parser.add_argument('--dset', type=str, help='datasett to use', default='mnist')
args = parser.parse_args()

params = get_siamese_config(args)


def load_siamese_data(data_path):
    h = h5py.File(os.path.join(data_path, '%s_siamese.hdf5' % args.dset))
    ret_dict = dict()
    keys = sorted(h.keys())
    for k in keys:
        k_i, i = k.split('-')
        l = ret_dict.get(k_i, [])
        l.append(h[k].value)
        ret_dict[k_i] = l

    return ret_dict


data = load_siamese_data(params['data_path'])

def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=False)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


ktf.set_session(get_session(args.gpu_memory_fraction))

# RUN Train
run_net(data, params)
