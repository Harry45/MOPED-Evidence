"""
Author: Dr. Arrykrishna Mootoovaloo
Collaborators: Prof. Alan Heavens, Dr. Elena Sellentin
Date: July 2022
Email: arrykrish@gmail.com
Project: Frequentist properties of the Bayes Factor
Code: Main script for computing the evidence and Bayes Factor.
"""

from src.nested import BayesFactor
import config as CONFIG

NESTED = BayesFactor(0.2, CONFIG.NDATA)
nested_results = NESTED.summaries(5, verbose=True, save=False)
