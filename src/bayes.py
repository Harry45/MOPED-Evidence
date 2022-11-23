"""
Author: Dr. Arrykrishna Mootoovaloo
Email: arrykrish@gmail.com
Date: November 2022
Project: Frequentist Properties of Bayes Factor
Script: to calculate the evidence and Bayes Factor
"""
import torch
from ml_collections.config_dict import ConfigDict


def evidence_uncompressed(config: ConfigDict, diff: torch.Tensor, dictionary: dict):

    cov1 = dictionary['Lambda_1'] + config.priors.precision1
    cov2 = dictionary['Lambda_2'] + config.priors.precision2

    nparams1 = cov1.shape[0]
    nparams2 = cov2.shape[0]

    prec1 = torch.linalg.inv(cov1)
    prec2 = torch.linalg.inv(cov2)

    moped_vec_x_1 = dictionary['moped_vec_1'].t() @ diff
    moped_vec_x_2 = dictionary['moped_vec_2'].t() @ diff

    data_term = -0.5 * diff.t() @ config.invNoiseCov @ diff
    noise_term = - 0.5 * torch.logdet(2.0 * torch.pi * config.noiseCov)
    fixed = data_term + noise_term

    evi1 = fixed + 0.5 * (moped_vec_x_1.t() @ prec1 @ moped_vec_x_1)
    evi1 = evi1 - 0.5 * torch.logdet(config.priors.cov1 @ dictionary['Lambda_1'] + torch.eye(nparams1))

    evi2 = fixed + 0.5 * (moped_vec_x_2.t() @ prec2 @ moped_vec_x_2)
    evi2 = evi2 - 0.5 * torch.logdet(config.priors.cov2 @ dictionary['Lambda_2'] + torch.eye(nparams2))

    print(evi1)
    print(evi2)
