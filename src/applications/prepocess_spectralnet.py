"""
"""

import argparse
import pickle

from applications.config import get_spectralnet_config
from core.data import build_spectral_data, load_base_data
import os
import h5py

# PARSE ARGUMENTS
from core.util import get_session
import keras.backend.tensorflow_backend as ktf

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='')
parser.add_argument('--gpu_memory_fraction', type=float, help='gpu percentage to use', default='1.0')
parser.add_argument('--dset', type=str, help='dataset to use', default='mnist')
args = parser.parse_args()

params = get_spectralnet_config(args)

if params.get('use_code_space'):
    ktf.set_session(get_session(args.gpu_memory_fraction))

base_data = load_base_data(params, args.dset)
data = build_spectral_data(params, base_data)

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
