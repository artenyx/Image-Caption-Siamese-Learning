from src import data_loaders as data_loaders, get_config as get_config, models as models

config = get_config.get_exp_config()
model = models.ImgCapModel(config=config).to(config['device'])
tokenizer = config['tokenizer']

mscoco_loader = data_loaders.get_mscoco_loader(config)
for i, (img, caption) in enumerate(mscoco_loader):
    print(type(caption))
    caption = [cap for cap in caption]
    caption = tokenizer(caption, return_tensors="pt", max_length=50, padding="max_length").to(config['device'])
    img1 = img
    caption1 = caption
    if i == 0:
        break

output = model(img1.to(config['device']), caption1)
