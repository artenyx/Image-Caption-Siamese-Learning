import torch.cuda
import torchvision.transforms as T

from transformers import GPT2TokenizerFast, GPT2Model


def get_exp_config():
    config = {'transforms': T.Compose([T.ToTensor(), T.Resize((32, 32))]),
              'tokenizer': GPT2TokenizerFast.from_pretrained("gpt2", pad_token='[PAD]'),
              'imgPath': "IMGPATH",
              'annPath': "ANNPATH",
              'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}
    return config
