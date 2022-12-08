import logging
import torch
from datetime import datetime

from src.experiments import epoch_exp, alpha_exp
from src.config import get_exp_config


def main():
    alpha_exp()


if __name__ == "__main__":
    main()

