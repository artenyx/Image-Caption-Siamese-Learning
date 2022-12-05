import logging

import torch

from src.train import train_imgcap_network, eval_imgcap_network
from src.config import get_exp_config


def main():
    log_fmt = "[%(levelname)s] %(asctime)s - %(message)s"
    logging.basicConfig(format=log_fmt, level=logging.DEBUG)
    logging.info("Starting experiment.")
    eval_imgcap_network()
    logging.info("Experiment complete.")






#train_imgcap_network()


if __name__ == "__main__":
    main()

