"""
Author: Dr. Arrykrishna Mootoovaloo
Email: arrykrish@gmail.com
Date: November 2022
Project: Frequentist Properties of Bayes Factor
Script: to generate and store the data
"""

from typing import Tuple
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from ml_collections.config_dict import ConfigDict

# our scripts and functions
from src.model import second_model
from src.model import grad_first_model, grad_second_model
from utils.helpers import save_pickle, load_pickle
from utils.logger import get_logger


def generate_data(config: ConfigDict) -> torch.Tensor:
    """Generate the data using the second model.

    Args:
        config (ConfigDict): the main configuration file.

    Returns:
        torch.Tensor: the generated data.
    """
    mean_noise = torch.zeros(config.ndata)
    noise_dist = MultivariateNormal(mean_noise, config.noiseCov)
    model_for_data = second_model(config.p_data, config.xgrid)
    noise = noise_dist.sample()
    data = model_for_data + noise
    return data


def calculate_fiducial(config: ConfigDict, save: bool) -> torch.Tensor:
    """Calculates the fiducial model and optionally store it.

    Args:
        config (ConfigDict): the main configuration file.
        save (bool): save teh fiducial model to a file if True

    Returns:
        torch.Tensor: the model evalulated at the fiducial point.
    """
    model_fid = second_model(config.fid2, config.xgrid)
    if save:
        save_pickle(model_fid, 'data', 'model_fiducial')
    return model_fid


def calculate_difference(data: torch.Tensor, model_fid: torch.Tensor,
                         save: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate the data from the extended model, y=ax^2 + bx + c.

    Args:
        data (torch.Tensor): the data generated
        model_fid (torch.Tensor): the fiducial model
        save (bool): save the outputs if set to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: the fiducial model and the difference between data and fiducial model
    """
    diff = data - model_fid
    if save:
        save_pickle(diff, 'data', 'difference')
    return diff


def compression(config: ConfigDict, save: bool) -> dict:
    """Compression of the data and generate the relevant MOPED vectors, coefficients and covariance.

    Args:
        config (ConfigDict): the main configuration file.
        save (bool): will save a dictionary if set to True.

    Returns:
        dict: a dictionary consisting of the gradients, MOPED vectors, coefficients and covariances.
    """
    logger = get_logger(config, 'compression')
    logger.info('Running the compression algorithm')

    dictionary = dict()
    diff = load_pickle('data', 'difference')

    dictionary['grad1'] = grad_first_model(config.fid1, config.xgrid)
    dictionary['grad2'] = grad_second_model(config.fid2, config.xgrid)

    dictionary['moped_vec_1'] = config.invNoiseCov @ dictionary['grad1']
    dictionary['moped_vec_2'] = config.invNoiseCov @ dictionary['grad2']

    dictionary['Lambda_1'] = dictionary['moped_vec_1'].t() @ config.noiseCov @ dictionary['moped_vec_1']
    dictionary['Lambda_2'] = dictionary['moped_vec_2'].t() @ config.noiseCov @ dictionary['moped_vec_2']

    dictionary['moped_coeff_1'] = dictionary['moped_vec_1'].t() @ diff
    dictionary['moped_coeff_2'] = dictionary['moped_vec_2'].t() @ diff

    if save:
        save_pickle(dictionary, 'data', 'compression')

    return dictionary
