import logging

from src.train import train_imgcap_network
from src.config import get_exp_config
from src.loaders import get_cifar10_loader


def main():
    log_fmt = "[%(levelname)s] %(asctime)s - %(message)s"
    logging.basicConfig(format=log_fmt, level=logging.DEBUG)
    logging.info("Starting experiment.")
    get_cifar10_loader()
    train_imgcap_network()


if __name__ == "__main__":
    main()
