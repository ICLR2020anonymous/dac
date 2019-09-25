import torch
from torchvision.transforms import Resize
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
import pickle
import tarfile
import os

class MiniImagenet(Dataset):

    url = 'https://drive.google.com/file/d/16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY/view?usp=sharing'

    def check_raw_exists(self):
        if not os.path.isfile(os.path.join(self.root, 'mini-imagenet-cache-train.pkl')) or\
                not os.path.isfile(os.path.join(self.root, 'mini-imagenet-cache-val.pkl')) or\
                not os.path.isfile(os.path.join(self.root, 'mini-imagenet-cache-test.pkl')):
            if not os.path.isfile(os.path.join(self.root, 'mini-imagenet.tar.gz')):
                raise ValueError(
                        "Raw file not downloaded. Please download from {} and place in the root"\
                                .format(self.url)
                                )
            else:
                self.unzip()

    def check_if_preprocessed(self, size):
        return os.path.isfile(os.path.join(self.root, '{}_train.pt'.format(size))) and \
                os.path.isfile(os.path.join(self.root, '{}_test_overlap.pt'.format(size))) and \
                os.path.isfile(os.path.join(self.root, '{}_val.pt'.format(size))) and \
                os.path.isfile(os.path.join(self.root, '{}_test.pt'.format(size)))

    def unzip(self):
        print('Extracting mini-imagenet.tar.gz')
        tar = tarfile.open(os.path.join(self.root, 'mini-imagenet.tar.gz'), "r:gz")
        tar.extractall(path=self.root)
        tar.close()
        os.remove(os.path.join(self.root, 'mini-imagenet.tar.gz'))

    def preprocess(self, size):
        resize_fn = Resize(size)
        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, 'mini-imagenet-cache-{}.pkl'.format(split))
            print('Processing ' + filename)
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            processed = []
            for img in tqdm(data['image_data']):
                img = resize_fn(Image.fromarray(img))
                processed.append(torch.ByteTensor(np.asarray(img)))
            processed = torch.stack(processed, 0)

            if split == 'train':
                imgs = []
                imgs_test = []
                labels = []
                labels_test = []

                for i, key in enumerate(data['class_dict'].keys()):
                    idx = data['class_dict'][key]
                    idx, idx_test = idx[:400], idx[400:]

                    imgs.append(processed[idx])
                    labels.append(i*torch.ones(400, dtype=torch.int))

                    imgs_test.append(processed[idx_test])
                    labels_test.append(i*torch.ones(200, dtype=torch.int))

                imgs = torch.cat(imgs, 0)
                labels = torch.cat(labels, 0)
                torch.save((imgs, labels),
                        os.path.join(self.root, '{}_train.pt'.format(size)))

                imgs_test = torch.cat(imgs_test, 0)
                labels_test = torch.cat(labels_test, 0)
                torch.save((imgs_test, labels_test),
                        os.path.join(self.root, '{}_test_overlap.pt'.format(size)))

            else:
                labels = torch.zeros(processed.shape[0], dtype=torch.int)
                for i, key in enumerate(data['class_dict'].keys()):
                    labels[data['class_dict'][key]] = i

                torch.save((processed, labels),
                        os.path.join(self.root, '{}_{}.pt'.format(size, split)))

            print('Done')

    def __init__(self, root, split='train', size=32, transform=None):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split

        self.check_raw_exists()
        if not self.check_if_preprocessed(size):
            self.preprocess(size)

        self.data, self.targets = torch.load(os.path.join(self.root,
            '{}_{}.pt'.format(size, split)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy())
        if self.transform is not None:
            img = self.transform(img)

        return img, target
