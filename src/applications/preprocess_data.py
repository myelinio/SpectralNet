"""
"""
import argparse
import os
import pickle


from applications.config import get_common_config
from core.data import load_data

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--dset', type=str, help='dataset to use', default='mnist')
args = parser.parse_args()

params = get_common_config(args)

print("Loading data")
data = load_data(params)
print("Finsihed Loading data")

data_path = os.path.join(params['data_path'], '%s_data.pkl' % args.dset)
if not os.path.exists(params['data_path']):
    os.makedirs(params['data_path'])

if os.path.exists(data_path):
    os.remove(data_path)

file = open(data_path, 'wb')
pickle.dump(data, file)
file.close()
