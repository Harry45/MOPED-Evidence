"""
Author: Dr. Arrykrishna Mootoovaloo
Collaborators: Prof. Alan Heavens, Dr. Elena Sellentin
Date: July 2022
Email: arrykrish@gmail.com
Project: Frequentist properties of the Bayes Factor
Code: Investigating nested models.
"""
import os
from typing import Tuple
import torch
import numpy as np
import matplotlib.pylab as plt

# our script and functions
import config as CONFIG
import utils.helpers as hp
from src.glm import quadratic_two, generate_design, evidence, posterior

# settings for plottings
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'serif': ['Palatino']})
figSize = (12, 8)
fontSize = 20


def sinusoidal(xvalues: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    """Calculates the sin function given the domain and the parameters.

    Args:
        xvalues (torch.Tensor): the domain of the function.
        parameters (torch.Tensor): the parameters of the function.

    Returns:
        torch.Tensor: the output of the function.
    """

    return parameters[0] * torch.sin(parameters[1] * xvalues)


def generate_data_sin(params: torch.Tensor, sigma: float, sin: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate the data using one of the two models.

    Args:
        params (torch.Tensor): the set of parameters to use
        sigma (float): the noise level
        sin (bool): use the sinusiodal function with the two parameters if True

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: the domain and the function
    """
    assert len(params) == 2, "The number of parameters must be two."

    xvalues = torch.linspace(CONFIG.N_XMIN, CONFIG.N_XMAX, CONFIG.N_NDATA)

    if sin:
        output = sinusoidal(xvalues, params)
    else:
        output = quadratic_two(xvalues, params)

    # add the noise
    output += torch.randn(CONFIG.N_NDATA) * sigma
    output = output.type(torch.float64)

    return xvalues, output


def loglike(data: torch.Tensor, xvalues: torch.Tensor,
            parameters: torch.Tensor, sigma: float = 0.1, sin: bool = True) -> float:
    """Calculate the log likelihood of the data given the model and the parameters.

    Args:
        data (torch.Tensor): the data to use.
        xvalues (torch.Tensor): the domain of the function.
        parameters (torch.Tensor): the parameters of the function.
        sigma (float, optional): the noise level. Defaults to 0.1.
        sin (bool, optional): use sinusoidal if True. Defaults to True.

    Returns:
        float: the log likelihood of the data.
    """

    # the function/model/signal
    if sin:
        signal = sinusoidal(xvalues, parameters)

    else:
        signal = quadratic_two(xvalues, parameters)

    # number of data points
    ndata = data.shape[0]

    # difference between the data and the signal
    epsilon = data - signal

    # the chi2 term
    chi2 = torch.sum((epsilon / sigma)**2)

    # the constant term
    constant = torch.tensor([2.0 * torch.pi * sigma**2])
    constant = ndata * torch.log(constant)

    return -0.5 * (chi2.item() + constant.item())


def contour_plot(sin_dict: dict, quad_dict: dict, save: bool):

    # the sinusoidal function
    amp_grid = sin_dict['amp_grid']
    ang_grid = sin_dict['ang_grid']
    logl_sin = sin_dict['logl']

    # the quadratic model
    a_grid = quad_dict['a_grid']
    b_grid = quad_dict['b_grid']
    logl_quad = quad_dict['logl']

    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    contours = plt.contour(amp_grid, ang_grid, torch.exp(logl_sin - torch.amax(logl_sin)), 3, colors='black')
    contours.collections[0].remove()
    plt.scatter([CONFIG.AMP], [CONFIG.ANG], s=40, label='Fiducial Point')
    plt.xlim(CONFIG.AMP_MIN, CONFIG.AMP_MAX)
    plt.ylim(CONFIG.ANG_MIN, CONFIG.ANG_MAX)
    plt.ylabel(r'$\omega$', fontsize=fontSize)
    plt.xlabel(r'$A$', fontsize=fontSize)
    plt.tick_params(axis='x', labelsize=fontSize)
    plt.tick_params(axis='y', labelsize=fontSize)
    plt.legend(loc='best', prop={'family': 'sans-serif', 'size': 15})

    plt.subplot(122)
    contours = plt.contour(a_grid, b_grid, torch.exp(logl_quad - torch.amax(logl_quad)), 3, colors='black')
    contours.collections[0].remove()
    plt.xlim(CONFIG.A_MIN, CONFIG.A_MAX)
    plt.ylim(CONFIG.B_MIN, CONFIG.B_MAX)
    plt.ylabel(r'$b$', fontsize=fontSize)
    plt.xlabel(r'$a$', fontsize=fontSize)
    plt.tick_params(axis='x', labelsize=fontSize)
    plt.tick_params(axis='y', labelsize=fontSize)

    if save:
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/contours.pdf', bbox_inches='tight')
    else:
        plt.show()


def loglike_grid(data: torch.Tensor, xvalues: torch.Tensor, sigma: float, verbose: bool = False, save: bool = False) -> dict:

    # the amplitude and angular frequency on a grid
    amp = torch.linspace(CONFIG.AMP_MIN, CONFIG.AMP_MAX, CONFIG.NPOINTS)
    ang = torch.linspace(CONFIG.ANG_MIN, CONFIG.ANG_MAX, CONFIG.NPOINTS)
    amp_grid, ang_grid = torch.meshgrid(amp, ang, indexing='ij')

    # the parameters a and b on a grid
    a_quad = torch.linspace(CONFIG.A_MIN, CONFIG.A_MAX, CONFIG.NPOINTS)
    b_quad = torch.linspace(CONFIG.B_MIN, CONFIG.B_MAX, CONFIG.NPOINTS)
    a_grid, b_grid = torch.meshgrid(a_quad, b_quad, indexing='ij')

    # the loglikelihood for each point on the grid (sinusoidal model)
    logl_sin = list()
    for amplitude in amp:
        for angular in ang:
            parameters = torch.tensor([amplitude, angular])
            logl_sin.append(loglike(data, xvalues, parameters, sigma, True))

    # the loglikelihood for each point on the grid (quadratic model)
    logl_quad = list()
    for a_val in a_quad:
        for b_val in b_quad:
            parameters = torch.tensor([a_val, b_val])
            logl_quad.append(loglike(data, xvalues, parameters, sigma, False))

    # convert list to tensor
    logl_sin = torch.FloatTensor(logl_sin).view(CONFIG.NPOINTS, CONFIG.NPOINTS)
    logl_quad = torch.FloatTensor(logl_quad).view(CONFIG.NPOINTS, CONFIG.NPOINTS)

    # find the MLE point when using the sinusoidal model
    id_max_sin = np.unravel_index(torch.argmax(logl_sin), logl_sin.shape)
    sin_mle = [amp[id_max_sin[0]].item(), ang[id_max_sin[1]].item()]

    # find the MLE point when using the quadratic model
    id_max_quad = np.unravel_index(torch.argmax(logl_quad), logl_quad.shape)

    # record the important quantities in a dictionary
    sin_dict = {'amp_grid': amp_grid, 'ang_grid': ang_grid, 'logl': logl_sin, 'mle': sin_mle}

    return sin_dict
