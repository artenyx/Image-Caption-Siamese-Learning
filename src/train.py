import time

import torch
import torch.nn as nn

from src.config import get_exp_config
from src.models import ImgCapModel
from src.loaders import get_mscoco_loader, get_cifar10_loader


def standardize(tensor):
    # Vectorize each example
    tensor = tensor.reshape(tensor.shape[0], -1)
    sd = torch.sqrt(torch.sum(tensor * tensor, dim=1)).reshape(-1, 1)
    tensor = tensor / (sd + 0.001)
    return tensor


def simclr_loss_func(embedding1, embedding2, lam=0.5):
    assert embedding1.shape == embedding2.shape
    batch_size = embedding1.shape[0]

    # Standardize 64 dim outputs of original and deformed images
    embedding1_stand = standardize(embedding1)
    embedding2_stand = standardize(embedding2)
    # Compute 3 covariance matrices - 0-1, 0-0, 1-1.
    cov12 = torch.mm(embedding1_stand, embedding2_stand.transpose(0, 1))  # COV
    cov11 = torch.mm(embedding1_stand, embedding1_stand.transpose(0, 1))  # COV0
    cov22 = torch.mm(embedding2_stand, embedding2_stand.transpose(0, 1))  # COV1
    # Diagonals of covariances.
    d12 = torch.diag(cov12)  # v
    d11 = torch.diag(cov11)  # v0
    d22 = torch.diag(cov22)  # v1
    # Mulitnomial logistic loss just computed on positive match examples, with all other examples as a separate class.
    lecov = torch.log(
        torch.exp(torch.logsumexp(cov12, dim=1)) + torch.exp(torch.logsumexp(cov11 - torch.diag(d11), dim=1)))
    lecov += torch.log(
        torch.exp(torch.logsumexp(cov12, dim=1)) + torch.exp(torch.logsumexp(cov22 - torch.diag(d22), dim=1)))
    lecov = lam * lecov - d12

    loss = torch.mean(lecov)

    '''
    # Accuracy
      if torch.cuda.is_available():
        ID = 2. * torch.eye(batch_size).to('cuda') - 1.
      else:
        ID = 2. *torch.eye(batch_size) - 1
      icov=ID*cov12
      acc=torch.sum((icov>0).type(torch.float))/ batch_size
      '''

    return loss


def tokenize_text(cap, tokenizer):
    cap = tokenizer(cap, return_tensors="pt", max_length=50, padding="max_length")
    return cap


def run_epoch_image_caption(model, config, batches_to_run=10000):
    t0 = time.perf_counter()
    tokenizer = config['tokenizer']
    loader = config['loaders_train'][0]
    optimizer = config['optimizer']

    running_loss = 0
    for i, (img, cap) in enumerate(loader):
        cap = tokenize_text(cap, tokenizer).to(config['device'])
        img = img.to(config['device'])
        img_emb, cap_emb = model(img.to(config['device']), cap)

        loss = simclr_loss_func(img_emb, cap_emb, lam=config['simclr_lam'])

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i == batches_to_run - 1:
            break
    running_loss /= len(loader)
    t1 = time.perf_counter()
    return t1 - t0, running_loss


def train_imgcap_network(model=None, config=None):
    if config is None:
        config = get_exp_config()
    if model is None:
        model = ImgCapModel(config=config).to(config['device'])

    config['loaders_train'] = [get_mscoco_loader(config)]
    config['optimizer'] = config['optimizer_type'](model.parameters(), lr=config['lr'])
    data_list = []
    for i in range(config['epochs']):
        data_list.append(run_epoch_image_caption(model, config))

    return data_list


def eval_imgcap_network(model=None, config=None):
    if config is None:
        config = get_exp_config()
    if model is None:
        model = ImgCapModel(config=config).to(config['device'])

    model.eval()
    tokenizer = config['tokenizer']
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
    cifar10_loader = get_cifar10_loader(config=config)
    correct = 0
    for i, (img, label) in enumerate(cifar10_loader):
        label = label.to(config['device'])
        img = torch.cat([img] * 10, dim=0).to(config['device'])  # creating batch tensor of input image
        cap = tokenize_text([v for k, v in prompts.items()], tokenizer).to(config['device'])
        img_emb, cap_emb = model(img, cap)

        cos = nn.CosineSimilarity()
        sim = cos(img_emb, cap_emb)

        correct += 1 if torch.argmax(sim, dim=0) == label else 0

    correct /= len(cifar10_loader)
    return correct

