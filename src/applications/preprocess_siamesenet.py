"""
Expected run times on a GTX 1080 GPU:
MNIST: 1 hr
Reuters: 2.5 hrs
cc: 15 min
"""

import argparse
import os

import h5py

from applications.config import get_siamese_config
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
# LOAD DATA
data = build_siamese_data(params)

data_path = os.path.join(params['data_path'], '%s_siamese.hdf5' % args.dset)
if os.path.exists(data_path):
    os.remove(data_path)

h = h5py.File(data_path)
for k, v in data.items():
    for i in range(len(v)):
        h.create_dataset("%s-%s" % (k, i), data=v[i])
