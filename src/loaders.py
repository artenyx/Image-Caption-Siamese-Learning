import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms as T

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from PIL import Image
import os
import os.path

from src.config import get_exp_config


class CocoCaptions(torch.utils.data.Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    Example:

        .. code:: python

            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root = 'dir where images are',
                                    annFile = 'json annotation file',
                                    transform=transforms.ToTensor())

            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample

            print("Image Size: ", img.size())
            print(target)

        Output: ::

            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']

    """
    def __init__(self, root, annFile, dset_size=None, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.dset_size = dset_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        target = [ann['caption'] for ann in anns]
        target = target[0]

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = T.Compose([T.ToTensor(), T.Resize((32, 32))])(img)
        if self.transform is not None:
            img_aug = self.transform(img)
        else:
            img_aug = None
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, img_aug, target

    def __len__(self):
        if self.dset_size is not None:
            return self.dset_size
        else:
            return len(self.ids)


def get_mscoco_loaders(config=None):
    if config is None:
        config = get_exp_config()
    if config['new_method']:
        s = 0.25
        config['transforms_mscoco'] = T.Compose([T.RandomCrop(24), T.Resize((32, 32)), T.RandomHorizontalFlip(p=0.8), T.ColorJitter(brightness=0.8 * s, contrast=0.8 * s, saturation=0.8 * s, hue=0.2 * s)])
    mscoco_train = CocoCaptions(root=config['imgPath_train'],
                                annFile=config['annPath_train'],
                                transform=config['transforms_mscoco'],
                                dset_size=config['train_dset_size'])
    mscoco_val = CocoCaptions(root=config['imgPath_val'],
                              annFile=config['annPath_val'],
                              transform=config['transforms_mscoco'])

    mscoco_train = torch.utils.data.Subset(mscoco_train, range(config['train_dset_size']))
    mscoco_loader_tr = torch.utils.data.DataLoader(mscoco_train, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    mscoco_loader_val = torch.utils.data.DataLoader(mscoco_val, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    print(f"Number of samples in train/val: {len(mscoco_train)}/{len(mscoco_val)}")
    return mscoco_loader_tr, mscoco_loader_val


def get_cifar10_loader(config=None):
    if config is None:
        config = get_exp_config()
    cifar10_test = datasets.CIFAR10(root="data", train=False, download=True, transform=config['transforms_cifar'])
    cifar10_test = torch.utils.data.Subset(cifar10_test, range(config['eval_dset_size']))
    cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=1, shuffle=True, num_workers=config['num_workers'])
    return cifar10_test_loader


def get_kmeans_from_embedding(config, model, loader, img_or_cap="img"):
    model.eval()
    embeddings = []
    captions = []
    with torch.no_grad():
        for img, img_aug, cap in loader:
            img = img.cuda()
            if img_or_cap == "img":
                embed = model.encode_img(img)
            else:
                embed = model.encode_text(cap)
            if len(embed.shape) != 1:
                embed = nn.Flatten()(embed)
            embeddings.append(embed)
            captions.append(cap)
    embeddings = torch.cat(embeddings).detach().cpu()
    kmeans = kmeans_embeddings(embeddings.numpy())
    kmeans_labels = pd.DataFrame(kmeans.labels_)
    kmeans_labels.to_csv(f"{config['exp_path']}/kmeans_labels.csv")
    kmeans_inertia = pd.DataFrame(kmeans.inertia_)
    kmeans_inertia.to_csv(f"{config['exp_path']}/kmeans_inertia.csv")
    return kmeans


def kmeans_embeddings(embeddings: np.array):
    embeddings = pd.DataFrame(embeddings)
    kmeans_model = KMeans(n_clusters=10, random_state=32932).fit(embeddings)
    return kmeans_model