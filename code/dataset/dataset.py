import os
import json

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms

class IAADataset(data.Dataset):
    """IAA dataset

    Args:
        annos_file: json file of annotations
        data_dir: directory to the data, os.path.join(data_dir, annos_file['files'][index]['image']) should be the absolute path of the image.
        transform: preprocessing and augmentation of the training images
    """
    def __init__(self, annos_file, data_dir, ratings=False, transform=None):
        with open(annos_file, 'r') as f:
            self.annotations = json.load(f)['files']
        self.data_dir = data_dir
        self.ratings = ratings
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        idx_anno = self.annotations[idx]
        img_name = os.path.join(self.data_dir, idx_anno['image'])
        image = default_loader(img_name)

        sample = {
            'image': image,
            'score': idx_anno['score'],
        }
        if self.transform:
            sample['image'] = self.transform(sample['image'])

        if self.ratings:
            ratings = np.array(idx_anno['ratings']).astype(np.float32).reshape(-1,1)
            sample.update({
                'ratings': ratings / np.sum(ratings),
            })

        return sample


if __name__ == '__main__':

    # sanity check
    root = './data/images'
    annos_file = './data/train_labels.csv'
    train_transform = transforms.Compose([
        transforms.Scale(256), 
        transforms.RandomCrop(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dset = IAADataset(annos_file=annos_file, data_dir=root, transform=train_transform)
    train_loader = data.DataLoader(dset, batch_size=4, shuffle=True, num_workers=4)
    for i, data in enumerate(train_loader):
        images = data['image']
        print(images.size())
        labels = data['annotations']
        print(labels.size())
