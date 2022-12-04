import torch.nn as nn

from src import data_loaders as data_loaders, get_config as get_config, models as models

config = get_config.get_exp_config()
model = models.ImgCapModel(config=config).to(config['device'])
tokenizer = config['tokenizer']
optimizer = config['optimizer_type'](model.parameters(), lr=config['lr_usl'])

mscoco_loader = data_loaders.get_mscoco_loader(config)
for i, (img, cap) in enumerate(mscoco_loader):
    cap = tokenizer(cap, return_tensors="pt", max_length=50, padding="max_length").to(config['device'])
    img = img.to(config['device'])
    output = model(img.to(config['device']), cap)

    loss = nn.CosineSimilarity(img, cap)
    loss.backwards()
    optimizer.step()

    print(loss.item())

    if i == 0:
        break

