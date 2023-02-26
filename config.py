"""
Author: Dr. Arrykrishna Mootoovaloo
Email: arrykrish@gmail.com
Date: November 2022
Project: Frequentist Properties of Bayes Factor
Script: Configuration file for running the code
"""

import torch
from ml_collections.config_dict import ConfigDict


def get_config(experiment: str) -> ConfigDict:
    """Generates the main configuration function in a tree-like structure.

    Args:
        experiment (str): Name of the experiment

    Returns:
        ConfigDict: the main configuration file.
    """

    config = ConfigDict()
    config.logname = experiment

    # important paths
    config.path = path = ConfigDict()
    path.results = 'results/'
    path.logs = 'logs/'

    # the domain
    config.xmin = 0.0
    config.xmax = torch.pi
    config.ndata = 50

    # prior width
    config.priorwidth = 10.0

    # jitter term for numerical stability
    config.jitter = 1E-4

    # these are for the first set of models (linear in the parameters)
    config.param_data = torch.tensor([2.5, 1.0])
    config.fid1 = torch.tensor([3.0, 1.2])
    config.fid2 = torch.tensor([-1.2, 3.8])

    # these are for the second set of models (non-linear in the parameters)
    # config.param_data = torch.tensor([2.5, 1.0])
    # config.fid1 = torch.tensor([3.0, 1.2])
    # config.fid2 = torch.tensor([-1.5, 4.7])

    return config
