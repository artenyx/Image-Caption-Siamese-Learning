import torch.nn as nn

from src import loaders as data_loaders, config as get_config, models as models
from src.simclr import simclr_loss_func

config = get_config.get_exp_config()
model = models.ImgCapModel(config=config).to(config['device'])
tokenizer = config['tokenizer']
optimizer = config['optimizer_type'](model.parameters(), lr=config['lr'])

mscoco_loader = data_loaders.get_mscoco_loader(config)
for i, (img, cap) in enumerate(mscoco_loader):
    cap = tokenizer(cap, return_tensors="pt", max_length=50, padding="max_length").to(config['device'])
    img = img.to(config['device'])
    img_emb, cap_emb = model(img.to(config['device']), cap)

    loss = simclr_loss_func(img_emb, cap_emb, lam=config['simclr_lam'])

    loss.backward()
    optimizer.step()

    print(loss.item())

    if i == 0:
        break

