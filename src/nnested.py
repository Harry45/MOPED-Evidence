"""
Author: Dr. Arrykrishna Mootoovaloo
Collaborators: Prof. Alan Heavens, Dr. Elena Sellentin
Date: July 2022
Email: arrykrish@gmail.com
Project: Frequentist properties of the Bayes Factor
Code: Investigating nested models.
"""
import torch
import numpy as np
from scipy.integrate import dblquad
from typing import Tuple

# our scripts and functions
from src.glm import generate_design, posterior, evidence
from src.sin import derivatives, loglike_grid, sinusoidal, quadratic, loglike
import config as CONFIG


class BayesNonNested:

    def __init__(self, data):
        # we assume the data has already been generated
        self.data = data

        # the x values
        self.xvalues = torch.linspace(CONFIG.XMIN, CONFIG.XMAX, CONFIG.NDATA)

        # compute the maximum likelihood estimator
        print('Evaluating the likelihood on a grid for the uncompressed data.')
        self.sin_dict, self.quad_dict = loglike_grid(self.data, self.xvalues, CONFIG.SIGMA)

        # value of the maximum likelihood
        self.max_loglike_sin = torch.amax(self.sin_dict['logl']).item()
        self.max_loglike_quad = torch.amax(self.quad_dict['logl']).item()

        self.theory_fid_sin = sinusoidal(self.xvalues, self.sin_dict['mle'])
        self.theory_fid_quad = quadratic(self.xvalues, self.quad_dict['mle'])
        self.theory_fid_sin = self.theory_fid_sin.type(torch.float64)
        self.theory_fid_quad = self.theory_fid_quad.type(torch.float64)

        self.postinit()

        print('Evaluating the likelihood on a grid for the compressed data.')
        self.sin_moped_dict, self.quad_moped_dict = self.loglike_moped_grid()

        # value of the maximum likelihood with MOPED
        self.max_loglike_moped_sin = torch.amax(self.sin_moped_dict['logl']).item()
        self.max_loglike_moped_quad = torch.amax(self.quad_moped_dict['logl']).item()

    def postinit(self):
        """Calculates all the important matrices, which will be used later.
        """

        # the covariance matrix
        self.cov = torch.from_numpy(np.diag([CONFIG.SIGMA ** 2] * CONFIG.NDATA))
        self.cov_inv = torch.from_numpy(np.diag([1.0 / CONFIG.SIGMA ** 2] * CONFIG.NDATA))

        # the the design matrix for the quadratic function
        self.phi = generate_design(self.xvalues, True)

        # MOPED vectors
        self.moped_vectors_quad = self.cov_inv @ self.phi
        self.moped_vectors_sin = self.cov_inv @ derivatives(self.xvalues, self.sin_dict['mle'], sin=True)

        # product of B and Phi
        self.b_phi_quad = self.moped_vectors_quad.t() @ self.phi

        # new covariance matrix
        self.cov_moped_quad = self.moped_vectors_quad.t() @ self.cov @ self.moped_vectors_quad
        self.cov_moped_sin = self.moped_vectors_sin.t() @ self.cov @ self.moped_vectors_sin

        # new precision matrix
        self.inv_cov_moped_quad = torch.linalg.inv(self.cov_moped_quad)
        self.inv_cov_moped_sin = torch.linalg.inv(self.cov_moped_sin)

        # the compressed data
        self.data_moped_sin = self.moped_vectors_sin.t() @ (self.data - self.theory_fid_sin)
        self.data_moped_quad = self.moped_vectors_quad.t() @ (self.data - self.theory_fid_quad)

    def summaries_quad(self) -> dict:
        """Calculate the posterior and evidence of the two models

        Args:
            nrealisation (int, optional): Number of realisations. Defaults to 5.
            verbose (bool, optional): Option to display the main results on the terminal. Defaults to False.
            save (bool, optional): Option to save the outputs. Defaults to False.

        Returns:
            dict: the summaries of the two models
        """

        # arguments for the posterior
        args_quad = (self.phi, self.cov_inv, CONFIG.INV_COV_PR_1, self.data, CONFIG.MU_PR_1)

        # mean and covariance for the uncompressed data
        mu_quad, cov_quad = posterior(*args_quad)

        # arguments for the evidence calculation
        args_evi_quad = (self.phi, self.cov, CONFIG.COV_PR_1, self.data, CONFIG.MU_PR_1)

        # evidence for the uncompressed data
        evi_quad = evidence(*args_evi_quad)

        # MOPED
        moped_data_quad = self.moped_vectors_quad.t() @ self.data

        args_moped_quad = (self.b_phi_quad, self.inv_cov_moped_quad,
                           CONFIG.INV_COV_PR_1, moped_data_quad, CONFIG.MU_PR_1)

        # mean and covariance for the compressed data
        mu_moped_quad, cov_moped_quad = posterior(*args_moped_quad)

        # arguments for the evidence calculation for the compressed data
        args_evi_moped_quad = (self.b_phi_quad, self.cov_moped_quad,
                               CONFIG.INV_COV_PR_1, moped_data_quad, CONFIG.MU_PR_1)

        # evidence for the compressed data
        evi_moped_quad = evidence(*args_evi_moped_quad)

        dictionary = {}
        dictionary['mu'] = mu_quad
        dictionary['cov'] = cov_quad
        dictionary['evi'] = evi_quad

        dictionary_moped = {}
        dictionary_moped['mu_moped'] = mu_moped_quad
        dictionary_moped['cov_moped'] = cov_moped_quad
        dictionary_moped['evi_moped'] = evi_moped_quad

        return dictionary, dictionary_moped

    def logpost_sin(self, amp: float, ang: float) -> np.ndarray:
        """Calculates the log-posterior when using the sinusoidal model.

        Args:
            amp (float): the amplitude
            ang (float): the angular frequency

        Returns:
            np.ndarray: the value of the log-posterior
        """

        # the parameters in a tensor
        param = torch.tensor([amp, ang])

        # the log-likelihood value
        logl = loglike(self.data, self.xvalues, param, CONFIG.SIGMA, True) - self.max_loglike_sin

        # the log-prior
        logp = CONFIG.PRIOR_SIN.log_prob(param).item()

        return np.exp(logl + logp)

    def logpost_sin_moped(self, amp: float, ang: float) -> np.ndarray:

        # the parameters in a tensor
        param = torch.tensor([amp, ang])

        # the log-likelihood value
        logl = self.loglike_moped(param, True) - self.max_loglike_moped_sin

        # the log-prior
        logp = CONFIG.PRIOR_SIN.log_prob(param).item()

        return np.exp(logl + logp)

    def log_evidence_sin(self) -> np.ndarray:
        """Calculates the value of the log-evidence for the sinusoidal model
        using the uncompressed data.

        Returns:
            np.ndarray: the value of the log-evidence for the uncompressed data.
        """
        int_sin = dblquad(self.logpost_sin, CONFIG.ANG_MIN, CONFIG.ANG_MAX,
                          lambda amp: CONFIG.AMP_MIN, lambda amp: CONFIG.AMP_MAX)

        evi = self.max_loglike_sin + np.log(int_sin[0])

        return evi

    def log_evidence_sin_moped(self) -> np.ndarray:
        """Calculates the value of the log-evidence for the sinusoidal model
        using the uncompressed data.

        Returns:
            np.ndarray: the value of the log-evidence for the uncompressed data.
        """
        int_sin = dblquad(self.logpost_sin_moped, CONFIG.ANG_MIN, CONFIG.ANG_MAX,
                          lambda amp: CONFIG.AMP_MIN, lambda amp: CONFIG.AMP_MAX)

        evi = self.max_loglike_moped_sin + np.log(int_sin[0])

        return evi

    def summaries_sin(self) -> Tuple[dict, dict]:
        """Calculates the evidence for the sinusoidal function for the
        uncompressed and compressed data.

        Returns:
            Tuple[dict, dict]: the evidence value for the uncompressed and
            compressed data.
        """
        dictionary = {}
        dictionary_moped = {}

        dictionary['evi'] = self.log_evidence_sin()
        dictionary_moped['evi_moped'] = self.log_evidence_sin_moped()

        return dictionary, dictionary_moped

    def loglike_moped(self, parameters: torch.Tensor, sin: bool = True) -> float:
        """Calculates the value of the log-likelihood for the compressed version
        of the data/theory.

        Args:
            parameters (torch.Tensor): the input parameters to the model.
            sin (bool, optional): Will use the sinusoidal function if True. Defaults to True.

        Returns:
            float: the value of the log-likelihood.
        """

        if sin:
            model = sinusoidal(self.xvalues, parameters)
            model = model.type(torch.float64)
            model = torch.matmul(self.moped_vectors_sin.t(), model - self.theory_fid_sin)

            epsilon = self.data_moped_sin - model
            chi2 = torch.matmul(epsilon, torch.matmul(self.inv_cov_moped_sin, epsilon))
            constant = torch.logdet(2.0 * torch.pi * self.cov_moped_sin)

        else:
            model = quadratic(self.xvalues, parameters)
            model = model.type(torch.float64)
            model = torch.matmul(self.moped_vectors_quad.t(), model - self.theory_fid_quad)
            epsilon = self.data_moped_quad - model
            chi2 = torch.matmul(epsilon, torch.matmul(self.inv_cov_moped_quad, epsilon))
            constant = torch.logdet(2.0 * torch.pi * self.cov_moped_quad)

        return -0.5 * (chi2 + constant)

    def loglike_moped_grid(self) -> Tuple[dict, dict]:
        """Calculates the log-likelihood values on a grid for both the sinusoidal
        and quadratic functions.

        Args:
            data (torch.Tensor): the data generated
            xvalues (torch.Tensor): the domain (values of x)
            sigma (float): the noise level

        Returns:
            Tuple[dict, dict]: dictionaries for the sinusoidal and quadratic models' likelihood
        """

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
                logl_sin.append(self.loglike_moped(parameters, True))

        # the loglikelihood for each point on the grid (quadratic model)
        logl_quad = list()
        for a_val in a_quad:
            for b_val in b_quad:
                parameters = torch.tensor([a_val, b_val])
                logl_quad.append(self.loglike_moped(parameters, False))

        # convert list to tensor
        logl_sin = torch.FloatTensor(logl_sin).view(CONFIG.NPOINTS, CONFIG.NPOINTS)
        logl_quad = torch.FloatTensor(logl_quad).view(CONFIG.NPOINTS, CONFIG.NPOINTS)

        # find the MLE point when using the sinusoidal model
        id_max_sin = np.unravel_index(torch.argmax(logl_sin), logl_sin.shape)
        sin_mle = torch.Tensor([amp[id_max_sin[0]].item(), ang[id_max_sin[1]].item()])

        # find the MLE point when using the quadratic model
        id_max_quad = np.unravel_index(torch.argmax(logl_quad), logl_quad.shape)
        quad_mle = torch.Tensor([a_quad[id_max_quad[0]].item(), b_quad[id_max_quad[1]].item()])

        # record the important quantities in a dictionary
        with torch.no_grad():
            sin_dict = {'amp_grid': amp_grid, 'ang_grid': ang_grid, 'logl': logl_sin, 'mle': sin_mle}
            quad_dict = {'a_grid': a_grid, 'b_grid': b_grid, 'logl': logl_quad, 'mle': quad_mle}

        return sin_dict, quad_dict
