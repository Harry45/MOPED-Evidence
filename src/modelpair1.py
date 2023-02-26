"""
Project: Estimating the log Bayes-Factor via data compression.
Group: ICIC
Authors: Alan, Arrykrishna, Roberto, Elena
Script: Involves linear models only.
Date: February 2023
"""

import logging
import torch


# LOGGER = logging.getLogger(__name__)
# LOGGER.info('Now loading the linear models.')


class FirstModel:
    """Calculates the basic quantities (exact function, gradient and approximate function) at a fiducial point.

    Args:
        fid_params (torch.Tensor): the fiducial point in parameter space.
        domain (torch.Tensor): the domain over which we want to compute the function.
    """

    def __init__(self, fid_params: torch.Tensor, domain: torch.Tensor):

        self.fid_params = fid_params
        self.fid_model = self.function(fid_params, domain)
        self.fid_grad = self.gradient(fid_params, domain)

    @staticmethod
    def function(parameters: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        """Calculates the function at that specific parameter.

        Args:
            parameters (torch.Tensor): the set of parameters.
            domain (torch.Tensor): the domain over which we want to compute the function

        Returns:
            torch.Tensor: the function evaluated at that set of parameters
        """
        target = parameters[0] + parameters[1] * torch.sin(domain)
        return target

    @staticmethod
    def gradient(parameters: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        """Calculates the gradient of the function at this parameter.

        Args:
            parameters (torch.Tensor): the set of parameters
            domain (torch.Tensor): the domain

        Returns:
            torch.Tensor: the gradient of shape N x p, where N is the number of data and p is the number of parameters.
        """
        # LOGGER.info('Calculating the gradients for M1 at %s', parameters)
        grad1 = torch.ones(len(domain))
        grad2 = torch.sin(domain)
        grad = torch.cat([grad1.view(-1, 1), grad2.view(-1, 1)], 1)
        return grad

    def taylor_function(self, parameters: torch.Tensor) -> torch.Tensor:
        """Calculates the approximate function using a first order Taylor expansion.

        Args:
            parameters (torch.Tensor): the set of parameters to use.

        Returns:
            torch.Tensor: the approximate function
        """
        approx_target = self.fid_model + self.fid_grad @ (parameters - self.fid_params)
        return approx_target


class SecondModel:
    """Calculates the basic quantities (exact function, gradient and approximate function) at a fiducial point.

    Args:
        fid_params (torch.Tensor): the fiducial point in parameter space.
        domain (torch.Tensor): the domain over which we want to compute the function.
    """

    def __init__(self, fid_params: torch.Tensor, domain: torch.Tensor):

        self.fid_params = fid_params
        self.fid_model = self.function(fid_params, domain)
        self.fid_grad = self.gradient(fid_params, domain)

    @staticmethod
    def function(parameters: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        """Calculates the function at that specific parameter.

        Args:
            parameters (torch.Tensor): the set of parameters.
            domain (torch.Tensor): the domain over which we want to compute the function

        Returns:
            torch.Tensor: the function evaluated at that set of parameters
        """
        target = parameters[0] * domain**2 + parameters[1] * domain
        return target

    @staticmethod
    def gradient(parameters: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        """Calculates the gradient of the function at this parameter.

        Args:
            parameters (torch.Tensor): the set of parameters
            domain (torch.Tensor): the domain

        Returns:
            torch.Tensor: the gradient of shape N x m, where N is the number of data and m is the number of parameters.
        """
        # LOGGER.info('Calculating the gradients for M2 at %s', parameters)
        grad1 = domain**2
        grad2 = domain
        grad = torch.cat([grad1.view(-1, 1), grad2.view(-1, 1)], 1)
        return grad

    def taylor_function(self, parameters: torch.Tensor) -> torch.Tensor:
        """Calculates the approximate function using a first order Taylor expansion.

        Args:
            parameters (torch.Tensor): the set of parameters to use.

        Returns:
            torch.Tensor: the approximate function
        """
        approx_target = self.fid_model + self.fid_grad @ (parameters - self.fid_params)
        return approx_target
