"""
Author: Dr. Arrykrishna Mootoovaloo
Email: arrykrish@gmail.com
Date: November 2022
Project: Frequentist Properties of Bayes Factor
Script: Configuration file for running the code
"""

import torch
from ml_collections.config_dict import ConfigDict


def get_config() -> ConfigDict:
    """Generates the main configuration function in a tree-like structure.

    Returns:
        ConfigDict: the main configuration file.
    """

    config = ConfigDict()
    config.logname = 'experiment'

    # important paths
    config.path = path = ConfigDict()
    path.results = 'results/'
    path.data = 'data/'
    path.logs = 'logs/'

    # for the main algorithm
    config.fid1 = torch.tensor([-1.0, 4.0])
    config.fid2 = torch.tensor([-1.0, 4.0, 0.0])
    config.p_data = torch.tensor([-0.90, 3.90, 0.05])  # the parameters used to generate the data

    # the domain
    config.xmin = 0.0
    config.xmax = torch.pi
    config.ndata = 50
    config.xgrid = torch.linspace(config.xmin, config.xmax, config.ndata)

    # the noise covariance
    config.sigma = 0.02
    config.noiseCov = config.sigma ** 2 * torch.eye(config.ndata)
    config.invNoiseCov = (1.0 / config.sigma**2) * torch.eye(config.ndata)

    # priors (centred on the fiducial points)
    config.priors = priors = ConfigDict()
    priors.cov1 = torch.eye(2)
    priors.cov2 = torch.eye(3)
    priors.precision1 = torch.linalg.inv(priors.cov1)
    priors.precision2 = torch.linalg.inv(priors.cov2)

    return config
