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
from utils.helpers import makedirs
from utils.logger import get_logger

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Main configuration file.", lock_config=True)


def main(args):
    """
    Run the main script
    """

    makedirs(FLAGS.config)
    logger = get_logger(FLAGS.config, 'main')
    logger.info("Running main script")


if __name__ == "__main__":
    app.run(main)
