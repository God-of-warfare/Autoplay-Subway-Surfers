import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image


class SubwaySurfers(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.LEFT = os.path.join(root, 'LEFT')
        self.left_length = len(os.listdir(self.LEFT))
        self.RIGHT = os.path.join(root, 'RIGHT')
        self.right_length = len(os.listdir(self.RIGHT))
        self.UP = os.path.join(root, 'UP')
        self.up_length = len(os.listdir(self.UP))
        self.DOWN = os.path.join(root, 'DOWN')
        self.down_length = len(os.listdir(self.DOWN))
        self.NOTHING = os.path.join(root, 'NOTHING')
        self.nothing_length = len(os.listdir(self.NOTHING))
        self.labels = np.load(root + 'subway_surfers_labels.npy')

    def __getitem__(self, index):

        if 0 <= index < self.left_length:
            img_path = os.listdir(self.LEFT)[index]
            label = 0  # left
            dir = self.LEFT

        elif self.left_length <= index < self.left_length + self.right_length:
            img_path = os.listdir(self.RIGHT)[index - self.left_length]
            label = 1   # right
            dir = self.RIGHT

        elif self.left_length + self.right_length <= index < self.left_length + self.right_length + self.up_length:
            img_path = os.listdir(self.UP)[index - self.left_length - self.right_length]
            label = 2   # up
            dir = self.UP

        elif self.left_length + self.right_length + self.up_length <= index < self.left_length + self.right_length + self.up_length + self.down_length:
            img_path = os.listdir(self.DOWN)[index - self.left_length - self.right_length - self.up_length]
            label = 3   # down
            dir = self.DOWN

        else:
            img_path = os.listdir(self.NOTHING)[index - self.left_length - self.right_length - self.up_length - self.down_length]
            label = 4   # nothing
            dir = self.NOTHING

        images = os.listdir(os.path.join(dir, img_path))
        img1 = Image.open(os.path.join(img_path, images[0]))
        img2 = Image.open(os.path.join(img_path, images[1]))
        img3 = Image.open(os.path.join(img_path, images[2]))

        img1 = np.array(img1) # [_,_,3]
        img2 = np.array(img2) # [_,_,3]
        img3 = np.array(img3) # [_,_,3]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        stacked = np.stack([img1, img2, img3], axis=0)
        images = np.transpose(stacked, (3, 0, 1, 2))
        images = torch.from_numpy(images)

        # Transpose to get [C,3,H,W]
        return images, label

    def __len__(self):
        length = (self.left_length + self.right_length + self.up_length + self.down_length + self.nothing_length)
        return length


