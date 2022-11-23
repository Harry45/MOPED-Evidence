"""
Author: Dr. Arrykrishna Mootoovaloo
Collaborators: Prof. Alan Heavens, Dr. Elena Sellentin
Date: July 2022
Email: arrykrish@gmail.com
Project: Frequentist properties of the Bayes Factor
Code: functions for Gaussian Linear Models
"""
from typing import Tuple
import torch

# our scripts and functions
import config as CONFIG


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


def posterior(design: torch.Tensor, precision_noise: torch.Tensor,
              precision_prior: torch.Tensor, yvals: torch.tensor, mu_prior: torch.Tensor) -> torch.Tensor:
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
    covariance = torch.linalg.inv(precision_prior + design.t() @ precision_noise @ design)

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
    cov_noise = cov_noise.float()
    cov_prior = cov_prior.float()

    mean_data = design @ mu_prior
    cov_data = cov_noise + design @ cov_prior @ design.t()
    pdf = torch.distributions.MultivariateNormal(mean_data.float(), cov_data.float())

    log_evidence = pdf.log_prob(yvals.float()).item()

    return log_evidence
