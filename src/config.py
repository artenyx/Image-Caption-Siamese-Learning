import torch.cuda
import torchvision.transforms as T
import os

from transformers import GPT2TokenizerFast, GPT2Model

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_exp_config():
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

    config = {'transforms_mscoco': T.Compose([T.ToTensor(), T.Resize((32, 32))]),
              'transforms_cifar': T.ToTensor(),
              'tokenizer': GPT2TokenizerFast.from_pretrained("gpt2", pad_token='[PAD]'),
              'imgPath_val': "/home/geraldkwhite/Image-Caption-Siamese-Learning/mscoco_data/val2017",
              'annPath_val': "/home/geraldkwhite/Image-Caption-Siamese-Learning/mscoco_data/annotations/captions_val2017.json",
              'imgPath_train': "/home/geraldkwhite/Image-Caption-Siamese-Learning/mscoco_data/train2017",
              'annPath_train': "/home/geraldkwhite/Image-Caption-Siamese-Learning/mscoco_data/annotations/captions_train2017.json",
              'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              'num_workers': 12,
              'epochs': 1,
              'optimizer_type': torch.optim.Adam,
              'lr': 0.0001,
              'batch_size': 256,
              'simclr_lam': 0.5,
              'train_dset_size': 15000,
              'eval_dset_size': 1000,
              'save_data': True,
              'latent_dim': 512,
              'cifar_labels': cifar_labels,

              'optimizer': None,
              'loaders_train': None,
              'loaders_eval': None}
    return config
