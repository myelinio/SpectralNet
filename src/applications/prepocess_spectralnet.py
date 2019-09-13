"""
"""

import argparse
import pickle

from applications.config import get_spectralnet_config
from core.data import build_spectral_data
import os
import h5py

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='')
parser.add_argument('--gpu_memory_fraction', type=float, help='gpu percentage to use', default='1.0')
parser.add_argument('--dset', type=str, help='dataset to use', default='mnist')
args = parser.parse_args()

params = get_spectralnet_config(args)

data_path = os.path.join(params['base_data_path'], '%s_data.pkl' % args.dset)
file = open(data_path, 'rb')
data = pickle.load(file)
file.close()

if params.get('use_code_space'):
    import keras.backend.tensorflow_backend as ktf
    import tensorflow as tf

    def get_session(gpu_fraction=0.333):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                    allow_growth=False)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    ktf.set_session(get_session(args.gpu_memory_fraction))

# LOAD DATA
data = build_spectral_data(params, data)

data_path = os.path.join(params['data_path'], '%s_spectralnet.hdf5' % args.dset)
if not os.path.exists(params['data_path']):
    os.makedirs(params['data_path'])

if os.path.exists(data_path):
    os.remove(data_path)

h = h5py.File(data_path)
for k, v in data['spectral'].items():
    for i in range(len(v)):
        h.create_dataset("spectral-%s-%s" % (k, i), data=v[i])
h.create_dataset("p_train", data=data['p_train'])
h.create_dataset("p_val", data=data['p_val'])
h.close()
