"""
Author: Dr. Arrykrishna Mootoovaloo
Collaborators: Prof. Alan Heavens, Dr. Elena Sellentin
Date: July 2022
Email: arrykrish@gmail.com
Project: Frequentist properties of the Bayes Factor
Code: Investigating non-nested models.
"""

import torch
import numpy as np

# our script and functions
from src.glm import generate_design, evidence, posterior
import config as CONFIG


class BayesNested(object):
    """Calculates the Bayesian Evidence and Bayes Factor for Gaussian Linear
    Models. We are considering nested models in this case, that is,
    f(x) = ax^2 + bx + c
    g(x) = ax^2 + bx

    g(x) is nested in f(x) at c=0.

    Args:
        sigma (float): the noise level
        ndata (int): the number of data points to use. Default is found in
        config file.
    """

    def __init__(self, data: torch.Tensor):

        self.data = data
        # the number of data points
        self.ndata = CONFIG.NDATA

        self._postinit()

    def _postinit(self):

        # the covariance matrix
        self.cov = torch.from_numpy(np.diag([CONFIG.SIGMA ** 2] * self.ndata))
        self.cov_inv = torch.from_numpy(np.diag([1 / CONFIG.SIGMA ** 2] * self.ndata))

        # the x values
        self.xvalues = torch.linspace(CONFIG.XMIN, CONFIG.XMAX, CONFIG.NDATA)

        # the two design matrices
        self.phi_1 = generate_design(self.xvalues, True)
        self.phi_2 = generate_design(self.xvalues, False)

        # MOPED vectors
        self.moped_vectors_1 = self.cov_inv @ self.phi_1
        self.moped_vectors_2 = self.cov_inv @ self.phi_2

        # product of B and Phi
        self.b_phi_1 = self.moped_vectors_1.t() @ self.phi_1
        self.b_phi_2 = self.moped_vectors_2.t() @ self.phi_2

        # new covariance matrix
        self.cov_moped_1 = self.moped_vectors_1.t() @ self.cov @ self.moped_vectors_1
        self.cov_moped_2 = self.moped_vectors_2.t() @ self.cov @ self.moped_vectors_2

        # new precision matrix
        self.inv_cov_moped_1 = torch.linalg.inv(self.cov_moped_1)
        self.inv_cov_moped_2 = torch.linalg.inv(self.cov_moped_2)

    def summaries(self, order=2) -> dict:
        """Calculates the mean and covariance of the posterior distribution.
        Also stores the evidence for the compressed and uncompressed data. The
        model considered are:

        y = ax^2 + bx

        and

        y = ax^2 + bx + c
        """

        assert order in [2, 3], 'Should be either 2 or 3'

        dictionary = {}
        dictionary_moped = {}
        record = {}
        if order == 2:
            record['design'] = self.phi_1
            record['cov_prior'] = CONFIG.COV_PR_1.float()
            record['inv_cov_prior'] = CONFIG.INV_COV_PR_1.float()
            record['mean_prior'] = CONFIG.MU_PR_1
            record['moped_vectors'] = self.moped_vectors_1
            record['B_Phi'] = self.b_phi_1
            record['cov_moped'] = self.cov_moped_1
            record['inv_cov_moped'] = self.inv_cov_moped_1

        else:
            record['design'] = self.phi_2
            record['cov_prior'] = CONFIG.COV_PR_2.float()
            record['inv_cov_prior'] = CONFIG.INV_COV_PR_2.float()
            record['mean_prior'] = CONFIG.MU_PR_2
            record['moped_vectors'] = self.moped_vectors_2
            record['B_Phi'] = self.b_phi_2
            record['cov_moped'] = self.cov_moped_2
            record['inv_cov_moped'] = self.inv_cov_moped_2

        # posterior distribution for the uncompressed data
        args = (record['design'], self.cov_inv, record['inv_cov_prior'], self.data, record['mean_prior'])
        mean, cov = posterior(*args)

        # evidence for uncompressed data
        args_evi = (record['design'], self.cov, record['cov_prior'], self.data, record['mean_prior'])
        evi = evidence(*args_evi)

        dictionary['mean'] = mean
        dictionary['cov'] = cov
        dictionary['evi'] = evi

        # calculate the compressed data
        moped_data = record['moped_vectors'].t() @ self.data

        # posterior distribution for the compressed data
        args_moped = (
            record['B_Phi'],
            record['inv_cov_moped'],
            record['inv_cov_prior'],
            moped_data, record['mean_prior'])
        mu_moped, cov_moped = posterior(*args_moped)

        # evidence for the compressed data
        args_evi_moped = (
            record['B_Phi'],
            record['cov_moped'],
            record['inv_cov_moped'],
            moped_data, record['mean_prior'])
        evi_moped = evidence(*args_evi_moped)

        dictionary_moped['mean_moped'] = mu_moped
        dictionary_moped['cov_moped'] = cov_moped
        dictionary_moped['evi_moped'] = evi_moped

        return dictionary, dictionary_moped
