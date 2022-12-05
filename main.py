import logging

from src.train import train_imgcap_network
from src.config import get_exp_config


def main():
    log_fmt = "[%(levelname)s] %(asctime)s - %(message)s"
    logging.basicConfig(format=log_fmt)
    lr_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    for lr in lr_list:
        print("lr:", lr)
        config = get_exp_config()
        config['lr'] = lr
        train_imgcap_network(config=config)


if __name__ == "__main__":
    main()
