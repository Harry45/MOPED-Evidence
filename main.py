"""
Author: Dr. Arrykrishna Mootoovaloo
Email: arrykrish@gmail.com
Date: November 2022
Project: Frequentist Properties of Bayes Factor
Script: The main configuration file
"""

from absl import flags, app
from ml_collections.config_flags import config_flags

# our scripts
from utils.helpers import makedirs, load_pickle
from utils.logger import get_logger
from src.data import generate_data, compression
from src.data import calculate_difference, calculate_fiducial
from src.bayes import evidence_uncompressed


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Main configuration file.", lock_config=True)


def main(args):
    """
    Run the main script
    """

    makedirs(FLAGS.config)
    logger = get_logger(FLAGS.config, 'main')
    logger.info("Running main script")

    # data = generate_data(FLAGS.config)
    # model_fid = calculate_fiducial(FLAGS.config, save=True)
    # diff = calculate_difference(data, model_fid, save=True)
    # dictionary = compression(FLAGS.config, save=True)

    model_fid = load_pickle('data', 'model_fiducial')
    dictionary = load_pickle('data', 'compression')
    data = generate_data(FLAGS.config)
    diff = calculate_difference(data, model_fid, save=False)
    evidence_uncompressed(FLAGS.config, diff, dictionary)


if __name__ == "__main__":
    app.run(main)
