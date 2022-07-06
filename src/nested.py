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
from typing import Tuple
from dataclasses import dataclass, field

# our script and functions
import config as CONFIG
import utils.helpers as hp


def quadratic_two(xvalues: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    """Calculate the theoretical model, a quadratic function with two parameters only.

    Args:
        xvalues (torch.Tensor): the domain of the function.
        parameters (torch.Tensor): the parameters of the function.

    Returns:
        torch.Tensor: the output of the function.
    """

    output = parameters[0] * xvalues ** 2 + parameters[1] * xvalues
    output = output.to(torch.float64)
    return output


def quadratic_three(xvalues: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    """Calculate the theoretical model, a quadratic function with three parameters only.

    Args:
        xvalues (torch.Tensor): the domain of the function.
        parameters (torch.Tensor): the parameters of the function

    Returns:
        torch.Tensor: the output of the function.
    """

    output = parameters[0] * xvalues ** 2 + parameters[1] * xvalues + parameters[2]
    output = output.to(torch.float64)
    return output


def generate_data(params: torch.Tensor, sigma: float, two: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate the data using one of the two models.

    Args:
        params (torch.Tensor): the set of parameters to use
        sigma (float): the noise level
        two (bool): use the quadratic function with the two parameters if True

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: the domain and the function
    """

    xvalues = torch.linspace(CONFIG.XMIN, CONFIG.XMAX, CONFIG.NDATA)

    if two:
        assert len(params) == 2, "The number of parameters must be two."
        output = quadratic_two(xvalues, params)
    else:
        assert len(params) == 3, "The number of parameters must be three."
        output = quadratic_three(xvalues, params)

    # add the noise
    output += torch.randn(CONFIG.NDATA) * sigma
    output = output.type(torch.float64)

    return xvalues, output


def generate_design(xvalues: torch.Tensor, two: bool) -> torch.Tensor:
    """Generates the design matrix for the data.

    Args:
        xvalues (torch.Tensor): the domain of the function.
        two (bool): use the quadratic function with the two parameters if True

    Returns:
        torch.Tensor: the design matrix
    """

    ndata = xvalues.shape[0]

    if two:
        design = torch.zeros((ndata, 2), dtype=float)
        design[:, 0] = xvalues ** 2
        design[:, 1] = xvalues
    else:
        design = torch.zeros((ndata, 3), dtype=float)
        design[:, 0] = xvalues ** 2
        design[:, 1] = xvalues
        design[:, 2] = torch.ones(ndata)

    return design


def posterior(
        design: torch.Tensor, precision_noise: torch.Tensor, precision_prior: torch.Tensor, yvals: torch.tensor,
        mu_prior: torch.Tensor) -> torch.Tensor:
    """Calculate the posterior of the model.

    Args:
        design (torch.Tensor): the design matrix
        precision_noise (torch.Tensor): the precision matrix
        precision_prior (torch.Tensor): the precision matrix of the prior
        yvals (torch.Tensor): the data
        mu_prior (torch.Tensor): the mean of the prior

    Returns:
        torch.Tensor: the posterior of the model
    """

    # the posterior covariance matrix
    covariance = torch.linalg.inv(precision_prior + design.t().matmul(precision_noise).matmul(design))

    # the mean of the posterior
    mean = design.t() @ precision_noise @ yvals + precision_prior @ mu_prior
    mean = covariance @ mean
    return mean, covariance


def evidence(design: torch.Tensor, cov_noise: torch.Tensor, cov_prior: torch.Tensor, yvals: torch.tensor,
             mu_prior: torch.Tensor) -> float:
    """Calculate the evidence of the model.

    Args:
        design (torch.Tensor): the design matrix
        cov_noise (torch.Tensor): the covariance matrix of the noise
        cov_prior (torch.Tensor): the covariance matrix of the prior
        yvals (torch.Tensor): the data
        mu_prior (torch.Tensor): the mean of the prior

    Returns:
        float: the evidence of the data
    """

    # print(design)
    # print(mu_prior)
    design = design.float()

    mean_data = design @ mu_prior
    cov_data = cov_noise + design @ cov_prior @ design.t()
    pdf = torch.distributions.MultivariateNormal(mean_data, cov_data)

    log_evidence = pdf.log_prob(yvals).item()

    return log_evidence


@dataclass
class BayesFactor(object):

    # the noise level
    sigma: float

    # the number of data points
    ndata: int

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
