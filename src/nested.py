"""
Author: Dr. Arrykrishna Mootoovaloo
Collaborators: Prof. Alan Heavens, Dr. Elena Sellentin
Date: July 2022
Email: arrykrish@gmail.com
Project: Frequentist properties of the Bayes Factor
Code: Investigating non-nested models.
"""

from dataclasses import dataclass
import torch
import numpy as np

# our script and functions
import config as CONFIG
import utils.helpers as hp
from src.glm import generate_data, generate_design, evidence, posterior


@dataclass
class BayesFactor(object):
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

    # the noise level
    sigma: float

    # the number of data points
    ndata: int = CONFIG.NDATA

    def __post_init__(self):

        # the covariance matrix
        self.cov = torch.from_numpy(np.diag([self.sigma ** 2] * self.ndata))
        self.cov_inv = torch.from_numpy(np.diag([1 / self.sigma ** 2] * self.ndata))

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

    def summaries(self, nrealisation: int = 5, verbose: bool = False, save: bool = False) -> dict:
        """Calculate the posterior and evidence of the two models

        Args:
            nrealisation (int, optional): Number of realisations. Defaults to 5.
            verbose (bool, optional): Option to display the main results on the terminal. Defaults to False.
            save (bool, optional): Option to save the outputs. Defaults to False.

        Returns:
            dict: the summaries of the two models
        """

        dictionary = {}

        for i in range(nrealisation):
            _, yvals = generate_data([CONFIG.THETA_0, CONFIG.THETA_1], self.sigma, True)

            # arguments for the posterior
            args1 = (self.phi_1, self.cov_inv, CONFIG.INV_COV_PR_1, yvals, CONFIG.MU_PR_1)
            args2 = (self.phi_2, self.cov_inv, CONFIG.INV_COV_PR_2, yvals, CONFIG.MU_PR_2)

            # mean and covariance for the uncompressed data
            mu1, cov1 = posterior(*args1)
            mu2, cov2 = posterior(*args2)

            # arguments for the evidence calculation
            args_e1 = (self.phi_1, self.cov, CONFIG.COV_PR_1, yvals, CONFIG.MU_PR_1)
            args_e2 = (self.phi_2, self.cov, CONFIG.COV_PR_2, yvals, CONFIG.MU_PR_2)

            # evidence for the uncompressed data
            evi1 = evidence(*args_e1)
            evi2 = evidence(*args_e2)

            # MOPED
            moped_data_1 = self.moped_vectors_1.t() @ yvals
            moped_data_2 = self.moped_vectors_2.t() @ yvals

            args_m1 = (self.b_phi_1, self.inv_cov_moped_1, CONFIG.INV_COV_PR_1, moped_data_1, CONFIG.MU_PR_1)
            args_m2 = (self.b_phi_2, self.inv_cov_moped_2, CONFIG.INV_COV_PR_2, moped_data_2, CONFIG.MU_PR_2)

            # mean and covariance for the compressed data
            mu_moped_1, cov_moped_1 = posterior(*args_m1)
            mu_moped_2, cov_moped_2 = posterior(*args_m2)

            # arguments for the evidence calculation for the compressed data
            args_em1 = (self.b_phi_1, self.cov_moped_1, CONFIG.INV_COV_PR_1, moped_data_1, CONFIG.MU_PR_1)
            args_em2 = (self.b_phi_2, self.cov_moped_2, CONFIG.INV_COV_PR_2, moped_data_2, CONFIG.MU_PR_2)

            # evidence for the compressed data
            evi_moped_1 = evidence(*args_em1)
            evi_moped_2 = evidence(*args_em2)

            # record important quantities
            rec_1 = {}
            rec_2 = {}

            rec_1['mu'] = mu1.numpy()
            rec_1['cov'] = cov1.numpy()
            rec_1['evi'] = evi1
            rec_1['evi_moped'] = evi_moped_1

            rec_2['mu'] = mu2.numpy()
            rec_2['cov'] = cov2.numpy()
            rec_2['evi'] = evi2
            rec_2['evi_moped'] = evi_moped_2

            bf_12 = evi1 - evi2
            bf_moped_12 = evi_moped_1 - evi_moped_2

            dictionary[i] = {'model_1': rec_1, 'model_2': rec_2, 'bf_12': bf_12, 'bf_moped_12': bf_moped_12}

            if save:
                hp.save_pickle(dictionary, 'results', 'nested')

            if verbose:

                print(mu1)
                print(mu_moped_1)
                print(torch.diag(cov1))
                print(torch.diag(cov_moped_1))

                print(mu2)
                print(mu_moped_2)
                print(torch.diag(cov2))
                print(torch.diag(cov_moped_2))

                print(f'Evidence for Model 1 (Uncompressed): {evi1:.3f}')
                print(f'Evidence for Model 2 (Uncompressed): {evi2:.3f}')
                print(f'The log-BF (Uncompressed) between Model 1 and Model 2 is: {bf_12:.3f}')
                print(f'Evidence for Model 1 (Compressed): {evi_moped_1:.3f}')
                print(f'Evidence for Model 2 (Compressed): {evi_moped_2:.3f}')
                print(f'The log-BF(Compressed) between Model 1 and Model 2 is: {bf_moped_12:.3f}')
                print('-' * 10)

        return dictionary
