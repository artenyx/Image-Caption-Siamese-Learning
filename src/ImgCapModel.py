import torch.cuda
import torch.nn as nn

from transformers import GPT2Model

from src.SimpleViT import SimpleViT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ImgCapModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vis_model = SimpleViT(
            image_size=32,
            patch_size=4,
            num_classes=1000,
            dim=512,
            depth=6,
            heads=16,
            mlp_dim=2048)
        tokenizer = config['tokenizer']

        self.gpt2_mod = GPT2Model.from_pretrained("gpt2")
        self.gpt2_mod.resize_token_embeddings(len(tokenizer))
        self.flatten = nn.Flatten()
        self.lm_linear = nn.LazyLinear(512)

    def forward(self, img, cap):
        img = self.vis_model(img)

        cap = self.gpt2_mod(cap)
        cap = cap.last_hidden_state
        cap = self.flatten(cap)
        cap = self.lm_linear(cap)

        print("Img Out Shape:", img.shape, "Cap Out Shape:", cap.shape)
        return img, cap
