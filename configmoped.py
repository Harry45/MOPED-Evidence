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
    config.fid = torch.tensor([-1.0, 4.0, 0.0])
    config.sigma = 0.2
    config.xmin = 0.0
    config.xmax = torch.pi
    config.ndata = 20
    config.noiseCov = config.sigma ** 2 * torch.eye(config.ndata)
    config.invNoiseCov = (1.0 / config.sigma**2) * torch.eye(config.ndata)

    # priors
    config.priors = priors = ConfigDict()
    priors.mu1 = torch.tensor([-1.0, 4.0])
    priors.mu2 = torch.tensor([-1.0, 4.0, 0.0])  # same as fiducial point
    priors.cov1 = torch.eye(2)
    priors.cov2 = torch.eye(3)

    return config
