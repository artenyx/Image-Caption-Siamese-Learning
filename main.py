import logging

import torch

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
    cifar10_loader = get_cifar10_loader(config=config)
    for i, (img, label) in enumerate(cifar10_loader):

        imgs = torch.cat([img]*10, dim=0)
        print(imgs.shape)
        print(imgs[0,0,0,:10])
        print(imgs[9,0,0,:10])
        break


#train_imgcap_network()


if __name__ == "__main__":
    main()

