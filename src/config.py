import torch.cuda
import torchvision.transforms as T
import os

from transformers import GPT2TokenizerFast, GPT2Model

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_exp_config():
    config = {'transforms_mscoco': T.Compose([T.ToTensor(), T.Resize((32, 32))]),
              'transforms_cifar': T.ToTensor(),
              'tokenizer': GPT2TokenizerFast.from_pretrained("gpt2", pad_token='[PAD]'),
              'imgPath': "/home/geraldkwhite/Image-Caption-Siamese-Learning/mscoco_data/val2017",
              'annPath': "/home/geraldkwhite/Image-Caption-Siamese-Learning/mscoco_data/annotations/captions_val2017.json",
              'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              'num_workers': 12,
              'epochs': 10,
              'optimizer_type': torch.optim.Adam,
              'lr': 0.001,
              'batch_size': 256,
              'simclr_lam': 0.5,

              'optimizer': None,
              'loaders': None}
    return config
