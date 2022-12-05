import logging

import torch

from src.experiments import run_img_cap_learning


def main():
    log_fmt = "[%(levelname)s] %(asctime)s - %(message)s"
    logging.basicConfig(format=log_fmt, level=logging.DEBUG)
    logging.info("Starting experiment.")
    run_img_cap_learning()
    logging.info("Experiment complete.")


if __name__ == "__main__":
    main()

