import logging
import torch

from src.experiments import run_img_cap_learning


def main():
    torch.cuda.memory_summary(device=None, abbreviated=False)
    lr_list = [10**i for i in range(-5, 0)]
    print(f"Testing learning rates: {lr_list}")
    for lr in lr_list:
        run_img_cap_learning(add_string=str(lr))


if __name__ == "__main__":
    main()

