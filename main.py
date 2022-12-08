import logging
import torch
from datetime import datetime

from src.experiments import epoch_exp, alpha_exp
from src.config import get_exp_config


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    epoch_exp()
    alpha_exp()


if __name__ == "__main__":
    main()

