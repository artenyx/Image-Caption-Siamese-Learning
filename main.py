import logging
import torch
from datetime import datetime

from src.experiments import run_img_cap_learning
from src.config import get_exp_config


def main():
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"Starting at {current_time}")
    run_img_cap_learning()


if __name__ == "__main__":
    main()

