"""
Expected run times on a GTX 1080 GPU:
MNIST: 1 hr
Reuters: 2.5 hrs
cc: 15 min
"""

# add directories in src/ to path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import argparse

import h5py
import keras.backend.tensorflow_backend as ktf
import tensorflow as tf

from applications.config import get_spectralnet_config
from applications.spectralnet import run_net
from core.data import build_spectral_data
import os

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='')
parser.add_argument('--gpu_memory_fraction', type=float, help='gpu percentage to use', default='1.0')
parser.add_argument('--dset', type=str, help='datasett to use', default='mnist')

args = parser.parse_args()
params = get_spectralnet_config(args)


# LOAD DATA
def load_spectral_data(data_path):
    h = h5py.File(os.path.join(data_path, '%s_spectralnet.hdf5' % args.dset))
    keys = [k for k in sorted(h.keys()) if k.startswith('spectral')]
    spectral_dict = dict()
    for k in keys:
        _, k_i, i = k.split('-')
        l = spectral_dict.get(k_i, [])
        l.append(h[k].value)
        spectral_dict[k_i] = l

    ret_dict = dict()
    ret_dict["spectral"] = spectral_dict
    ret_dict["p_train"] = h["p_train"].value
    ret_dict["p_val"] = h["p_val"].value

    return ret_dict


data = load_spectral_data(params['data_path'])


def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=False)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


ktf.set_session(get_session(args.gpu_memory_fraction))

# RUN EXPERIMENT
x_spectralnet, y_spectralnet = run_net(data, params)