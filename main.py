import logging
import torch
import time

from src.experiments import run_img_cap_learning
from src.config import get_exp_config


def main():
    lr_list = [10**i for i in range(-6, -4)]
    print(f"Testing learning rates: {lr_list}")
    print(f"Starting at {time.time()}")
    for lr in lr_list:
        config = get_exp_config()
        config['lr'] = lr
        run_img_cap_learning(config=config, add_string=str(lr))


if __name__ == "__main__":
    main()

