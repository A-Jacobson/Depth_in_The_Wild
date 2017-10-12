import os
import pickle

from PIL import Image
from torch.utils.data import Dataset


class NYUDepth(Dataset):
    def __init__(self, path_img, path_target, transforms=None):
        self.path_img = path_img
        with open(path_target, 'rb') as f:
            self.targets = pickle.load(f)
        self.imgs = [target['name'] for target in self.targets]
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        target = self.targets[index]
        img = Image.open(os.path.join(self.path_img, target['name']))
        if self.transforms:
            img = self.transforms(img)
        return img, target
