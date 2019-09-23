import os
from collections import defaultdict

import myelin.admin


def get_spectralnet_config(args):
    params = get_common_config(args)
    # SELECT GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    base_data_task = myelin.admin.task(axon="spectral-net", task_name="DataPrep", namespace="myelin")
    print(base_data_task)
    params['base_data_path'] = base_data_task.data_path if base_data_task else '/tmp/data/'

    model_path = myelin.admin.model_path(default_value='/tmp/model/spectralnet/')
    params['model_path'] = model_path

    siamese_model_task = myelin.admin.task(axon="spectral-net", task_name="TrainSiameseModel", namespace="myelin")
    print(siamese_model_task)
    params['siamese_model_path'] = siamese_model_task.model_path if siamese_model_task else '/tmp/model/siamese/'

    ae_model_task = myelin.admin.task(axon="spectral-net", task_name="TrainAutoencoderModel", namespace="myelin")
    print(ae_model_task)
    params['ae_model_path'] = ae_model_task.model_path if ae_model_task else '/tmp/model/ae/'

    data_task = myelin.admin.task(axon="spectral-net", task_name="DataPrepSpectralNet", namespace="myelin")
    print(data_task)
    params['data_path'] = data_task.data_path if data_task else '/tmp/data/'

    return params


def get_siamese_config(args):
    params = get_common_config(args)
    # SELECT GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    base_data_task = myelin.admin.task(axon="spectral-net", task_name="DataPrep", namespace="myelin")
    print(base_data_task)
    params['base_data_path'] = base_data_task.data_path if base_data_task else '/tmp/data/'

    model_path = myelin.admin.model_path(default_value='/tmp/model/siamese/')
    params['model_path'] = model_path

    ae_model_task = myelin.admin.task(axon="spectral-net", task_name="TrainAutoencoderModel", namespace="myelin")
    params['ae_model_path'] = ae_model_task.model_path if ae_model_task else '/tmp/model/ae/'

    return params


def get_autoencoder_config(args):
    params = get_common_config(args)
    # SELECT GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    base_data_task = myelin.admin.task(axon="spectral-net", task_name="DataPrep", namespace="myelin")
    print(base_data_task)
    params['base_data_path'] = base_data_task.data_path if base_data_task else '/tmp/data/'

    model_path = myelin.admin.model_path(default_value='/tmp/model/ae/')
    params['model_path'] = model_path

    params['use_code_space'] = False
    return params


def get_common_config(args):

    data_path = myelin.admin.data_path(default_value='/tmp/data/')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    params = defaultdict(lambda: None)

    # SET GENERAL HYPERPARAMETERS
    general_params = {
        'dset': args.dset,  # dataset: reuters / mnist
        'val_set_fraction': 0.1,  # fraction of training set to use as validation
        'precomputedKNNPath': '',
        # path for precomputed nearest neighbors (with indices and saved as a pickle or h5py file)
        'siam_batch_size': 128,  # minibatch size for siamese net
    }
    params.update(general_params)

    # SET DATASET SPECIFIC HYPERPARAMETERS
    if args.dset == 'mnist':
        mnist_params = {
            'n_clusters': 10,  # number of clusters in data
            'use_code_space': True,  # enable / disable code space embedding
            'affinity': 'siamese',  # affinity type: siamese / knn
            'n_nbrs': 3,  # number of nonzero entries (neighbors) to use for graph Laplacian affinity matrix
            'scale_nbr': 2,  # neighbor used to determine scale of gaussian graph Laplacian; calculated by
            # taking median distance of the (scale_nbr)th neighbor, over a set of size batch_size
            # sampled from the datset

            'siam_k': 2,  # threshold where, for all k <= siam_k closest neighbors to x_i, (x_i, k) is considered
            # a 'positive' pair by siamese net

            'siam_ne': 400,  # number of training epochs for siamese net
            'spec_ne': 400,  # number of training epochs for spectral net
            'spec_ae': 400,
            'siam_lr': 1e-3,  # initial learning rate for siamese net
            'spec_lr': 1e-3,  # initial learning rate for spectral net
            'siam_patience': 10,  # early stopping patience for siamese net
            'spec_patience': 20,  # early stopping patience for spectral net
            'siam_drop': 0.1,  # learning rate scheduler decay for siamese net
            'spec_drop': 0.1,  # learning rate scheduler decay for spectral net
            'batch_size': 1024,  # batch size for spectral net
            'siam_reg': None,  # regularization parameter for siamese net
            'spec_reg': None,  # regularization parameter for spectral net
            'siam_n': None,  # subset of the dataset used to construct training pairs for siamese net
            'siamese_tot_pairs': 600000,  # total number of pairs for siamese net
            'arch': [  # network architecture. if different architectures are desired for siamese net and
                #   spectral net, 'siam_arch' and 'spec_arch' keys can be used
                {'type': 'relu', 'size': 1024},
                {'type': 'relu', 'size': 1024},
                {'type': 'relu', 'size': 512},
                {'type': 'relu', 'size': 10},
            ],
            'use_approx': False,  # enable / disable approximate nearest neighbors
            'use_all_data': True,  # enable to use all data for training (no test set)
        }
        params.update(mnist_params)
    elif args.dset == 'reuters':
        reuters_params = {
            'n_clusters': 4,
            'use_code_space': True,
            'affinity': 'siamese',
            'n_nbrs': 30,
            'scale_nbr': 10,
            'siam_k': 100,
            'siam_ne': 20,
            'spec_ne': 300,
            'spec_ae': 400,
            'siam_lr': 1e-3,
            'spec_lr': 5e-5,
            'siam_patience': 1,
            'spec_patience': 5,
            'siam_drop': 0.1,
            'spec_drop': 0.1,
            'batch_size': 2048,
            'siam_reg': 1e-2,
            'spec_reg': 5e-1,
            'siam_n': None,
            'siamese_tot_pairs': 400000,
            'arch': [
                {'type': 'relu', 'size': 512},
                {'type': 'relu', 'size': 256},
                {'type': 'relu', 'size': 128},
            ],
            'use_approx': True,
            'use_all_data': True,
        }
        params.update(reuters_params)
    elif args.dset == 'cc':
        cc_params = {
            # data generation parameters
            'train_set_fraction': 1.,  # fraction of the dataset to use for training
            'noise_sig': 0.1,  # variance of the gaussian noise applied to x
            'n': 1500,  # number of total points in dataset
            # training parameters
            'n_clusters': 2,
            'use_code_space': True,
            'affinity': 'siamese',

            'siam_k': 2,  # threshold where, for all k <= siam_k closest neighbors to x_i, (x_i, k) is considered
            # a 'positive' pair by siamese net
            'siam_ne': 20,
            'spec_ae': 400,
            'siam_lr': 1e-3,
            'siam_patience': 1,
            'siam_drop': 0.1,
            'siam_reg': 1e-2,
            'siam_n': None,
            # 'siamese_tot_pairs': 400000,

            # 'affinity': 'full',
            'n_nbrs': 2,
            'scale_nbr': 2,
            'spec_ne': 50,
            'spec_lr': 1e-3,
            'spec_patience': 30,
            'spec_drop': 0.1,
            'batch_size': 128,
            'batch_size_orthonorm': 128,
            'spec_reg': None,
            'arch': [
                {'type': 'softplus', 'size': 50},
                {'type': 'BatchNormalization'},
                {'type': 'softplus', 'size': 50},
                {'type': 'BatchNormalization'},
                {'type': 'softplus', 'size': 50},
                {'type': 'BatchNormalization'},
            ],
            'ae_reg': 1e-2,
            'ae_arch': [
                {'type': 'relu', 'size': 5},
                {'type': 'sigmoid', 'size': 2},
            ],
            'use_all_data': True,
        }
        params.update(cc_params)
    elif args.dset == 'cc_semisup':
        cc_semisup_params = {
            'spec_ae': 400,
            'dset': 'cc',  # dataset affects data loading in get_data() so we must set back to 'cc'
            # data generation parameters
            'train_set_fraction': .8,
            'noise_sig': 0.175,
            'n': 1900,
            # training parameters
            'train_labeled_fraction': 0.02,
            # fraction of the training set to provide labels for (in semisupervised experiments)
            'n_clusters': 2,
            'use_code_space': False,
            'affinity': 'full',
            'n_nbrs': 2,
            'scale_nbr': 2,
            'spec_ne': 300,
            'spec_lr': 1e-3,
            'spec_patience': 30,
            'spec_drop': 0.1,
            'batch_size': 128,
            'batch_size_orthonorm': 256,
            'spec_reg': None,
            'arch': [
                {'type': 'softplus', 'size': 50},
                {'type': 'BatchNormalization'},
                {'type': 'softplus', 'size': 50},
                {'type': 'BatchNormalization'},
                {'type': 'softplus', 'size': 50},
                {'type': 'BatchNormalization'},
            ],
            'generalization_metrics': True,  # enable to check out of set generalization error and nmi
            'use_all_data': False,
        }
        params.update(cc_semisup_params)

    params['data_path'] = data_path

    return params
