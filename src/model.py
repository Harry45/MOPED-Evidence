"""
Author: Dr. Arrykrishna Mootoovaloo
Email: arrykrish@gmail.com
Date: November 2022
Project: Frequentist Properties of Bayes Factor
Script: The model and its first derivative with respect to the parameters.
"""

from typing import Tuple
import torch


def first_model(parameters: torch.Tensor, xvalues: torch.Tensor) -> torch.Tensor:
    """The first model in this case is a quadratic model with just two parameters (a, b) such that:

    mu = a * x**2 + a * x

    The data is generated from the second model. Therefore, we will edit this function if we want to
    test a new model, for example, a sinusoidal function.

    Args:
        parameters (torch.Tensor): the set of parameters.
        xvalues (torch.Tensor): the values of x where we want to evaluate the theory.

    Returns:
        torch.Tensor: the evaluated model
    """
    yvalues = parameters[0] * xvalues**2 + parameters[1] * xvalues
    return yvalues


def second_model(parameters: torch.Tensor, xvalues: torch.Tensor) -> torch.Tensor:
    """The second model in this case is a quadratic model with three parameters (a, b, c) such that:

    mu = a * x**2 + b * x + c

    Args:
        parameters (torch.Tensor): the set of parameters.
        xvalues (torch.Tensor): the values of x where we want to evaluate the theory.

    Returns:
        torch.Tensor: the evaluated model
    """
    yvalues = parameters[0] * xvalues**2 + parameters[1] * xvalues + parameters[2]
    return yvalues


def grad_first_model(parameters: torch.Tensor, xvalues: torch.Tensor) -> torch.Tensor:
    """Evaluates the first model and its first derivative with respect to the parameters.

    Args:
        parameters (torch.Tensor): the parameters
        xvalues (torch.Tensor): the domain values of x where we want to evaluate the model.

    Returns:
        torch.Tensor: the model at that parameter and the first derivative
    """
    parameters.requires_grad = True
    grad = list()

    for xval in xvalues:
        model = first_model(parameters, xval)
        gradient = torch.autograd.grad(model, parameters)[0]
        grad.append(gradient)
    grad = torch.vstack(grad)
    parameters.requires_grad = False
    return grad


def grad_second_model(parameters: torch.Tensor, xvalues: torch.Tensor) -> torch.Tensor:
    """Evaluates the second model and its first derivative with respect to the parameters.

    Args:
        parameters (torch.Tensor): the parameters
        xvalues (torch.Tensor): the domain values of x where we want to evaluate the model.

    Returns:
        torch.Tensor: the model at that parameter and the first derivative
    """
    parameters.requires_grad = True
    grad = list()

    for xval in xvalues:
        model = second_model(parameters, xval)
        gradient = torch.autograd.grad(model, parameters)[0]
        grad.append(gradient)
    grad = torch.vstack(grad)
    parameters.requires_grad = False
    return grad
