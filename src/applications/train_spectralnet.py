"""
Expected run times on a GTX 1080 GPU:
MNIST: 1 hr
Reuters: 2.5 hrs
cc: 15 min
"""

# add directories in src/ to path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import argparse

import keras.backend.tensorflow_backend as ktf
import tensorflow as tf

from applications.config import get_spectralnet_config
from applications.spectralnet import run_net
from core.data import load_spectral_data

# Parse arguments
from core.util import get_session

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='')
parser.add_argument('--gpu_memory_fraction', type=float, help='gpu percentage to use', default='1.0')
parser.add_argument('--dset', type=str, help='dataset to use', default='mnist')

args = parser.parse_args()
params = get_spectralnet_config(args)
ktf.set_session(get_session(args.gpu_memory_fraction))

# Load data
data = load_spectral_data(params['data_path'], args.dset)

# Run training
x_spectralnet, y_spectralnet = run_net(data, params)