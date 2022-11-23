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

    moped_vec_x_1 = dictionary['moped_vec_1'].t() @ diff
    moped_vec_x_2 = dictionary['moped_vec_2'].t() @ diff

    print(diff.t() @ config.invNoiseCov @ diff)
    print(moped_vec_x_1.t() @ torch.linalg.inv(cov1) @ moped_vec_x_1)
    print(torch.logdet(2.0 * torch.pi * config.noiseCov))
    print(torch.logdet(config.priors.cov1 @ dictionary['Lambda_1'] + torch.eye(2)))
