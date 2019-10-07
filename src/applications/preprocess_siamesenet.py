"""
"""

import argparse
import pickle

from applications.config import get_siamese_config
from core.data import build_siamese_data, load_base_data
import os
import h5py
from core.util import get_session

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='')
parser.add_argument('--gpu_memory_fraction', type=float, help='gpu percentage to use', default='1.0')
parser.add_argument('--dset', type=str, help='dataset to use', default='mnist')
args = parser.parse_args()

params = get_siamese_config(args)

base_data = load_base_data(params, args.dset)
# LOAD DATA
data = build_siamese_data(params, base_data)

if params.get('use_code_space'):
    import keras.backend.tensorflow_backend as ktf

    ktf.set_session(get_session(args.gpu_memory_fraction))

data_path = os.path.join(params['data_path'], '%s_siamese.hdf5' % args.dset)
if not os.path.exists(params['data_path']):
    os.makedirs(params['data_path'])

if os.path.exists(data_path):
    os.remove(data_path)

h = h5py.File(data_path)
for k, v in data.items():
    for i in range(len(v)):
        h.create_dataset("%s-%s" % (k, i), data=v[i])
h.close()
