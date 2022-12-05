import torch
import torch.utils.data
from torchvision import datasets

from PIL import Image
import os
import os.path

from src.config import get_exp_config


class CocoCaptions1(torch.utils.data.Dataset):
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
    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

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
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.ids)


def get_mscoco_loader(config=None):
    if config is None:
        config = get_exp_config()
    mscoco = CocoCaptions1(root=config['imgPath'],
                           annFile=config['annPath'],
                           transform=config['transforms_mscoco'])
    mscoco_loader = torch.utils.data.DataLoader(mscoco, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    print(f"Number of samples: {len(mscoco)}")
    img, captions = mscoco[0]
    print(f"Image Size: {img.size()}")

    return mscoco_loader


def get_cifar10_loader(config=None, dset_size=100):
    if config is None:
        config = get_exp_config()
    cifar10_test = datasets.CIFAR10(root="data", train=False, download=True, transform=config['transforms_cifar'])
    cifar10_test = torch.utils.data.Subset(cifar10_test, range(dset_size))
    cifar10_test = [cifar10_test] * 10
    cifar10_test_loader = torch.utils.data.DataLoader(list(zip(*cifar10_test)), batch_size=10, shuffle=True, num_workers=config['num_workers'])
    return cifar10_test_loader

def get_cifar100_usl(config):
    transforms = T.Compose([config['transforms_dict'][key] for key in config['transforms_active']])
    print(transforms)
    batch_size = config['batch_size']

    if config['usl_type'] == 'ae_single' and not config['denoising']:
        dataset_list_train = [datasets.CIFAR100(root="data", train=True, download=True, transform=T.ToTensor())]
        dataset_list_test = [datasets.CIFAR100(root="data", train=False, download=True, transform=T.ToTensor())]
    elif config['usl_type'] == 'ae_single' and config['denoising']:
        dataset_list_train = [datasets.CIFAR100(root="data", train=True, download=True, transform=T.ToTensor()),
                              datasets.CIFAR100(root="data", train=True, download=True, transform=transforms)]
        dataset_list_test = [datasets.CIFAR100(root="data", train=False, download=True, transform=T.ToTensor()),
                             datasets.CIFAR100(root="data", train=False, download=True, transform=transforms)]
    else:
        dataset_list_train = [datasets.CIFAR100(root="data", train=True, download=True, transform=T.ToTensor()),
                              datasets.CIFAR100(root="data", train=True, download=True, transform=transforms),
                              datasets.CIFAR100(root="data", train=True, download=True, transform=transforms)]
        dataset_list_test = [datasets.CIFAR100(root="data", train=False, download=True, transform=T.ToTensor()),
                             datasets.CIFAR100(root="data", train=False, download=True, transform=transforms),
                             datasets.CIFAR100(root="data", train=False, download=True, transform=transforms)]
    if batch_size < 64:
        nworkers = 6
    else:
        nworkers = 12
    train_loader = torch.utils.data.DataLoader(list(zip(*dataset_list_train)), batch_size=batch_size, shuffle=True, num_workers=nworkers)
    test_loader = torch.utils.data.DataLoader(list(zip(*dataset_list_test)), batch_size=batch_size, shuffle=True, num_workers=nworkers)
    return train_loader, test_loader
