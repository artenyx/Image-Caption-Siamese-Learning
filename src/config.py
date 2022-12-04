import torch.cuda
import torchvision.transforms as T

from transformers import GPT2TokenizerFast, GPT2Model


def get_exp_config():
    config = {'transforms': T.Compose([T.ToTensor(), T.Resize((32, 32))]),
              'tokenizer': GPT2TokenizerFast.from_pretrained("gpt2", pad_token='[PAD]'),
              'imgPath': "/home/geraldkwhite/Image-Caption-Siamese-Learning/mscoco_data/val2017",
              'annPath': "/home/geraldkwhite/Image-Caption-Siamese-Learning/mscoco_data/annotations/captions_val2017.json",
              'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              'num_workers': 6,
              'optimizer_type': torch.optim.Adam,
              'optimizer': None,
              'lr': 0.001,
              'simclr_lam': 0.5}
    return config
