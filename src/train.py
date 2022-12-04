from src.simclr import simclr_loss_func
from src.config import get_exp_config
from src.models import ImgCapModel
from src.loaders import get_mscoco_loader


def run_epoch(model, config, batches_to_run=10000):
    tokenizer = config['tokenizer']
    loader = config['loaders'][0]
    optimizer = config['optimizer']

    running_loss = 0
    for i, (img, cap) in enumerate(loader):
        cap = tokenizer(cap, return_tensors="pt", max_length=50, padding="max_length").to(config['device'])
        img = img.to(config['device'])
        img_emb, cap_emb = model(img.to(config['device']), cap)

        loss = simclr_loss_func(img_emb, cap_emb, lam=config['simclr_lam'])

        loss.backward()
        optimizer.step()

        print(loss.item())
        running_loss += loss.item()

        if i == batches_to_run - 1:
            break
    running_loss /= len(loader)
    return running_loss


def train_imgcap_network(model=None, config=None):
    if config is None:
        config = get_exp_config()
    if model is None:
        model = ImgCapModel(config=config).to(config['device'])

    config['loaders'] = get_mscoco_loader(config)
    config['optimizer'] = config['optimizer_type'](model.parameters(), lr=config['lr'])
    loss_list = []
    for i in range(config['epochs']):
        loss_list.append(run_epoch(model, config, 1))

    print(loss_list)
    return