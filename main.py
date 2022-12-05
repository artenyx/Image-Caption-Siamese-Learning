import logging

from src.train import train_imgcap_network
from src.config import get_exp_config
from src.loaders import get_cifar10_loader


def main():
    log_fmt = "[%(levelname)s] %(asctime)s - %(message)s"
    logging.basicConfig(format=log_fmt, level=logging.DEBUG)
    logging.info("Starting experiment.")

    config = get_exp_config()


    cifar_labels = {0: "airplane",
                    1: "automobile",
                    2: "bird",
                    3: "cat",
                    4: "deer",
                    5: "dog",
                    6: "frog",
                    7: "horse",
                    8: "ship",
                    9: "truck"}

    prompts = {k: f"This image is a {v}" for k, v in cifar_labels.items()}
    print(prompts)
    get_cifar10_loader(config=config)


#train_imgcap_network()


if __name__ == "__main__":
    main()
