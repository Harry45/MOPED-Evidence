"""
Author: Arrykrishna Mootoovaloo
Collaborators: Alan Heavens, Elena Sellentin
Institution: Imperial College London
Project: Evidence calculation with non-nested models
Date: April 2022
"""

import torch


def gamma(shape: float, rate: float = 5.0):
    """

    Args:
        shape (float): _description_
        rate (float, optional): _description_. Defaults to 5.0.

    Returns:
        _type_: _description_
    """

    parameter = torch.tensor([shape], dtype=torch.float32)
    parameter.requires_grad = True

    rate = torch.tensor([rate], dtype=torch.float32)

    dist = torch.distributions.gamma.Gamma(shape, rate)

    logp = dist.log_prob(parameter)

    gradient = torch.autograd.grad(logp, parameter)

    return dist, logp.item(), gradient[0]


def loglike_gamma(data, alpha, rate: float = 5.0):

    alpha = torch.tensor([alpha], dtype=torch.float32)
    rate = torch.tensor([rate], dtype=torch.float32)

    dist = torch.distributions.gamma.Gamma(alpha, rate)

    logp = dist.log_prob(data)

    return torch.sum(logp).item()


def logprior_normal(data, mean: float = 10.0, sigma: float = 1.0):

    mean = torch.tensor([mean], dtype=torch.float32)
    sigma = torch.tensor([sigma], dtype=torch.float32)

    dist = torch.distributions.normal.Normal(mean, sigma)

    logp = dist.log_prob(data)

    return torch.sum(logp).item()


def loglike_normal(data, mean, sigma: float = 1.0):

    mean = torch.tensor([mean], dtype=torch.float32)
    sigma = torch.tensor([sigma], dtype=torch.float32)

    dist = torch.distributions.normal.Normal(mean, sigma)

    logp = dist.log_prob(data)

    return torch.sum(logp).item()
