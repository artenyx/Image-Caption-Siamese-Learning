from src import data_loaders as data_loaders, get_config as get_config, ImgCapModel as ImgCapModel

config = get_config.get_config()
model = ImgCapModel(config).to(config['device'])
