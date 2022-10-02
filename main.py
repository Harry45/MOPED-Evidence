"""
Author: Dr. Arrykrishna Mootoovaloo
Collaborators: Prof. Alan Heavens, Dr. Elena Sellentin
Date: July 2022
Email: arrykrish@gmail.com
Project: Frequentist properties of the Bayes Factor
Code: Main script for computing the evidence and Bayes Factor.
"""

# our scripts and functions
from src.nnested import BayesNonNested
from src.nested import BayesNested
from src.sin import generate_data_sin_quad
from src.glm import generate_data
import utils.helpers as hp
import config as CONFIG

import warnings
warnings.filterwarnings('ignore')


def main(nrealisations: int, nested: bool):
    """_summary_

    Args:
        nrealisations (int): _description_
        nested (bool): _description_
    """
    record = list()
    record_moped = list()

    if nested:
        for i in range(nrealisations):

            # the data is generated from the quadratic model
            xinputs, data = generate_data([CONFIG.THETA_0, CONFIG.THETA_1], CONFIG.SIGMA, two=True)
            nested_model = BayesNested(data)
            dict_full_2, dict_moped_2 = nested_model.summaries(order=2)
            dict_full_3, dict_moped_3 = nested_model.summaries(order=3)

            log_bf = dict_full_2['evi'] - dict_full_3['evi']
            log_bf_moped = dict_moped_2['evi_moped'] - dict_moped_3['evi_moped']

            record.append(log_bf)
            record_moped.append(log_bf_moped)

            if (i + 1) % (nrealisations / 100) == 0:
                per = (i + 1) / nrealisations * 100
                print(f'Percentage completed: {int(per):3d} %')

        # save the outputs
        hp.store_arrays(record, 'results', 'nested')
        hp.store_arrays(record_moped, 'results', 'nested-moped')

    else:
        for i in range(nrealisations):

            # the data is generated from the sinusoidal function
            xinputs, data = generate_data_sin_quad(CONFIG.SIN_PARAMS, CONFIG.SIGMA, sin=True)
            nonnested = BayesNonNested(data)
            dict_full_quad, dict_moped_quad = nonnested.summaries_quad()
            dict_full_sin, dict_moped_sin = nonnested.summaries_sin()

            log_bf = dict_full_sin['evi'] - dict_full_quad['evi']
            log_bf_moped = dict_moped_sin['evi_moped'] - dict_moped_quad['evi_moped']

            record.append(log_bf)
            record_moped.append(log_bf_moped)

            if (i + 1) % (nrealisations / 100) == 0:
                per = (i + 1) / nrealisations * 100
                print(f'Percentage completed: {int(per):3d} %')

        # save the outputs
        hp.store_arrays(record, 'results', 'non-nested')
        hp.store_arrays(record_moped, 'results', 'non-nested-moped')


if __name__ == '__main__':
    main(nrealisations=1000, nested=False)
