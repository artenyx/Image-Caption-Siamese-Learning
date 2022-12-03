from src import data_loaders as data_loaders, get_config as get_config, models as models

config = get_config.get_exp_config()
model = models.ImgCapModel(config=config).to(config['device'])

mscoco_loader = data_loaders.get_mscoco_loader()
for i, (img, caption) in enumerate(mscoco_loader):
    print(img.shape)
    print(caption)
    img1 = img
    caption1 = caption
    if i == 0:
        break

output = model(img1, caption1)
