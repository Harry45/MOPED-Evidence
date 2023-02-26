"""
Project: Estimating the log Bayes-Factor via data compression.
Group: ICIC
Authors: Alan, Arrykrishna, Roberto, Elena
Date: February 2023
"""
import logging
from typing import Tuple
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from ml_collections.config_dict import ConfigDict

# our scripts and functions
from src.modelpair1 import FirstModel, SecondModel


# LOGGER = logging.getLogger(__name__)
# LOGGER.info('Now calculating the Bayes Factors.')


def marginal_likelihood(mean: torch.Tensor, design: torch.Tensor, covnoise: torch.Tensor,
                        covprior: torch.Tensor) -> MultivariateNormal:
    """Generates a multivariate normal distribution, given the mean and noise covariance, prior
    covariance and the gradient of the theoretical model.

    Args:
        mean (torch.Tensor): the mean of the multivariate normal distribution.
        design (torch.Tensor): the first derivatives of the function at the fiducial point.
        covnoise (torch.Tensor): the noise covariance matrix.
        covprior (torch.Tensor): the prior covariance matrix.

    Returns:
        MultivariateNormal: the multivariate normal object.
    """
    datacov = covnoise + design @ covprior @ design.t()
    distribution = MultivariateNormal(mean, datacov)
    return distribution


class BayesFactor:
    """Calculates the Bayes Factor given two models, M1 and M2.

    Args:
        config (ConfigDict): the main configuration file with all the settings.
        noisecov (torch.Tensor): the noise covariance matrix
    """

    def __init__(self, config: ConfigDict, noisecov: torch.Tensor):

        self.config = config
        self.domain = torch.linspace(config.xmin, config.xmax, config.ndata)
        self.model1 = FirstModel(config.fid1, self.domain)
        self.model2 = SecondModel(config.fid2, self.domain)
        self.precomputations_full_data(noisecov)
        self.precomputations_comp_data(noisecov)

    def precomputations_full_data(self, noisecov: torch.Tensor):
        """Precompute some quantities related to the calculation of the marginal likelihood for the full data.

        Args:
            noisecov (torch.Tensor): the noise covariance matrix.
        """

        # number of parameters
        self.nparam1 = len(self.model1.fid_params)
        self.nparam2 = len(self.model2.fid_params)
        self.nparam_tot = self.nparam1 + self.nparam2

        # priors
        priorcov1 = torch.eye(self.nparam1) * self.config.priorwidth**2
        priorcov2 = torch.eye(self.nparam2) * self.config.priorwidth**2

        # marginal likelihood for the full data under model 1 and 2
        self.mvn1 = marginal_likelihood(self.model1.fid_model, self.model1.fid_grad, noisecov, priorcov1)
        self.mvn2 = marginal_likelihood(self.model2.fid_model, self.model2.fid_grad, noisecov, priorcov2)

    def precomputations_comp_data(self, noisecov: torch.Tensor):
        """Precompute some quantities for the calculation of the marginal likelihood for the compressed data.

        Args:
            noisecov (torch.Tensor): the noise covariance matrix.
        """

        invnoise = torch.linalg.inv(noisecov)
        self.b1_mat = invnoise @ self.model1.fid_grad
        self.b2_mat = invnoise @ self.model2.fid_grad

        b_mat_ex = torch.cat([self.b1_mat, self.b2_mat], 1)
        lam_ex = b_mat_ex.T @ invnoise @ b_mat_ex

        # priors
        priorcov1 = torch.eye(self.nparam1) * self.config.priorwidth**2
        priorcov2 = torch.eye(self.nparam2) * self.config.priorwidth**2

        # new covariance matrices for the compressed data
        lam_1_ex = lam_ex + b_mat_ex.T @ self.model1.fid_grad @ priorcov1 @ self.model1.fid_grad.t() @ b_mat_ex
        lam_2_ex = lam_ex + b_mat_ex.T @ self.model2.fid_grad @ priorcov2 @ self.model2.fid_grad.t() @ b_mat_ex

        # difference in the models evaluated at the ficucial points
        diff_11 = self.b1_mat.t() @ (self.model1.fid_model - self.model1.fid_model)
        diff_12 = self.b2_mat.t() @ (self.model1.fid_model - self.model2.fid_model)
        diff_21 = self.b1_mat.t() @ (self.model2.fid_model - self.model1.fid_model)
        diff_22 = self.b2_mat.t() @ (self.model2.fid_model - self.model2.fid_model)
        ystar1 = torch.cat([diff_11, diff_12])
        ystar2 = torch.cat([diff_21, diff_22])

        # stable covariance matrix for the compressed data
        cov1 = lam_1_ex + self.config.jitter * torch.eye(self.nparam_tot)
        cov2 = lam_2_ex + self.config.jitter * torch.eye(self.nparam_tot)

        # the multivariate normal distributions
        self.mvn_comp_1 = MultivariateNormal(ystar1, cov1)
        self.mvn_comp_2 = MultivariateNormal(ystar2, cov2)

    def log_evi_full_data(self, data: torch.Tensor) -> Tuple[float, float, float]:
        """Calculates the marginal likelihood for the data, under two models, M1 and M2, and also the Bayes Factor.

        Args:
            data (torch.Tensor): the full data.

        Returns:
            Tuple[float, float, float]: evidence for M1, M2 and the Bayes Factor
        """
        evi1 = self.mvn1.log_prob(data).item()
        evi2 = self.mvn2.log_prob(data).item()
        return evi1, evi2, evi1 - evi2

    def log_evi_comp_data(self, data: torch.Tensor) -> Tuple[float, float, float]:
        """Calculates the marginal likelihood for the compressed data under the two models, M1 and M2 and
        also the Bayes Factor.

        Args:
            data (torch.Tensor): the full data.

        Returns:
            Tuple[float, float, float]: evidence for M1, M2 and the Bayes Factor
        """
        ycomp1 = self.b1_mat.t() @ (data - self.model1.fid_model)
        ycomp2 = self.b2_mat.t() @ (data - self.model2.fid_model)
        ycomp = torch.cat([ycomp1, ycomp2])
        evi1 = self.mvn_comp_1.log_prob(ycomp).item()
        evi2 = self.mvn_comp_2.log_prob(ycomp).item()
        return evi1, evi2, evi1 - evi2
