"""
Author: Dr. Arrykrishna Mootoovaloo
Collaborators: Prof. Alan Heavens, Dr. Elena Sellentin
Date: July 2022
Email: arrykrish@gmail.com
Project: Frequentist properties of the Bayes Factor
Code: Basic configurations for the project
"""

import torch

# minimum and maximum of x
XMIN = 0.0
XMAX = torch.pi

# number of data points to generate
NDATA = 20

# the noise level
SIGMA = 0.2

# the set of parameters to use
THETA_0 = -1.0
THETA_1 = 4.0
THETA_2 = 0.05

# the mean of the priors
MU_PR_1 = torch.tensor([THETA_0, THETA_1])
MU_PR_2 = torch.tensor([THETA_0, THETA_1, THETA_2])

# the covariance of the priors
COV_PR_1 = 100 * torch.eye(2)
COV_PR_2 = torch.eye(3)

# the inverse of the covariance of the priors
INV_COV_PR_1 = torch.linalg.inv(COV_PR_1)
INV_COV_PR_2 = torch.linalg.inv(COV_PR_2)

# ----------------------------------------------------------------------------------------------------------------------
# Configurations for the sinusoidal and quadratic model (non-nested case)

# amplitude of the sinusoidal model
AMP = 4.0

# angular frequency
ANG = 0.75

# the set of parameters to use
SIN_PARAMS = torch.tensor([AMP, ANG])

# this is for calculating the log-likelihood on a grid of points
AMP_MIN = 3.80
AMP_MAX = 4.20
ANG_MIN = 0.65
ANG_MAX = 0.85

A_MIN = -1.20
A_MAX = -0.60
B_MIN = 3.00
B_MAX = 4.50

# number of points on the grid
NPOINTS = 200

# Priors
MU_PR_SIN = SIN_PARAMS
COV_PR_SIN = torch.eye(2)
PRIOR_QUAD = torch.distributions.MultivariateNormal(MU_PR_1, COV_PR_1)
PRIOR_SIN = torch.distributions.MultivariateNormal(MU_PR_SIN, COV_PR_SIN)
