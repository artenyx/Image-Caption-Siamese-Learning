import os
import torch
from datetime import datetime
import pandas as pd

from src.config import get_exp_config
from src.models import ImgCapModel
from src.train import train_imgcap_network, eval_imgcap_network


def run_img_cap_learning(config=None, model=None, add_string=''):
    if config is None:
        config = get_exp_config()
    if model is None:
        model = ImgCapModel(config=config).to(config['device'])

    train_data = train_imgcap_network(config=config, model=model)
    with torch.no_grad():
        eval_data = eval_imgcap_network(config=config, model=model)
    print(f"Model trained to {config['epochs']} epochs and achieved evaluation rate of {eval_data} on CIFAR10.")

    if config['save_data']:
        date_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + add_string
        exp_path = f"Experiment_Files/{date_time}"
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        data = train_data + [(eval_data, "NA")]
        data = pd.DataFrame(data)
        data.to_csv(f"{exp_path}/data")
        torch.save(model.state_dict(), f"{exp_path}/model.pt")









