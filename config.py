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
XMAX = 4.0

# number of data points to generate
NDATA = 20

# the set of parameters to use
THETA_0 = -1.0
THETA_1 = 4.0
THETA_2 = 0.05

# the mean of the priors
MU_PR_1 = torch.tensor([THETA_0, THETA_1])
MU_PR_2 = torch.tensor([THETA_0, THETA_1, THETA_2])

# the covariance of the priors
COV_PR_1 = torch.eye(2)
COV_PR_2 = torch.eye(3)

# the inverse of the covariance of the priors
INV_COV_PR_1 = torch.linalg.inv(COV_PR_1)
INV_COV_PR_2 = torch.linalg.inv(COV_PR_2)
