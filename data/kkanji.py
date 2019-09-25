import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import wget
import numpy as np
import os
import pickle
import tarfile
import os

# for clustered datasets
#TRAIN_NUM_PER_CLASS = 30
#TEST_NUM_PER_CLASS = 30
#TRAIN_NUM_CLASSES = 542
#TEST_NUM_CLASSES = 271

class KKanji(Dataset):

    url = 'http://codh.rois.ac.jp/kmnist/dataset/kkanji/kkanji.tar'

    def check_raw_exists(self):
        if self.check_if_preprocessed():
            return

        if not os.path.isdir(os.path.join(self.root, 'kkanji2')):
            if not os.path.isfile(os.path.join(self.root, 'kkanji.tar')):
                print('Downloading data...')
                wget.download(self.url, out=self.root)
                self.unzip()
            else:
                self.unzip()

    def check_if_preprocessed(self):
        return os.path.isfile(os.path.join(self.root, 'train.pt')) and \
                os.path.isfile(os.path.join(self.root, 'test.pt'))

    def preprocess(self):
        rootfolder = os.path.join(self.root, 'kkanji2')

        images = []
        labels = []
        label = 0
        for dirname in tqdm(os.listdir(rootfolder)[1:]):
            classfolder = os.path.join(rootfolder, dirname)
            imgfiles = []
            for filename in os.listdir(classfolder):
                if filename.endswith(".png"):
                    imgfiles.append(os.path.join(classfolder, filename))

            if len(imgfiles) > 30:
                labels.append(label*torch.ones(len(imgfiles), dtype=torch.int))
                for imgfile in imgfiles:
                    img = Image.open(imgfile)
                    images.append(torch.ByteTensor(np.asarray(img.resize((28, 28)))))
                label += 1

        images = torch.stack(images, 0)
        labels = torch.cat(labels, 0)
        ulabels = torch.unique(labels)
        ulabels = ulabels[torch.randperm(len(ulabels))]

        print('total {} classes, {} images'.format(len(ulabels), len(images)))

        idxs = torch.randperm(len(images))
        images = images[idxs]
        labels = labels[idxs]

        num_train_imgs = 2*len(images) // 3
        print('{} train images'.format(num_train_imgs))
        torch.save((images[:num_train_imgs], labels[:num_train_imgs]),
                os.path.join(self.root, 'train.pt'))

        print('{} test images'.format(len(images) - num_train_imgs))
        torch.save((images[num_train_imgs:], labels[num_train_imgs:]),
                os.path.join(self.root, 'test.pt'))

        #train_images = []
        #train_targets = []
        #num_train_classes = (2*len(ulabels))//3
        #for i, l in tqdm(enumerate(ulabels[:num_train_classes])):
        #    idxs = labels == l
        #    train_images.append(images[idxs])
        #    train_targets.append(i*torch.ones(sum(idxs.int()), dtype=torch.int))
        #train_images = torch.cat(train_images, 0)
        #train_targets = torch.cat(train_targets, 0)
        #print('train total {} classes, {} images'.format(
        #    num_train_classes, len(train_images)))
        #torch.save((train_images, train_targets),
        #        os.path.join(self.root, 'train.pt'))

        #test_images = []
        #test_targets = []
        #for i, l in tqdm(enumerate(ulabels[num_train_classes:])):
        #    idxs = labels == l
        #    test_images.append(images[idxs])
        #    test_targets.append(i*torch.ones(sum(idxs.int()), dtype=torch.int))
        #test_images = torch.cat(test_images, 0)
        #test_targets = torch.cat(test_targets, 0)
        #print('test total {} classes, {} images'.format(
        #    len(ulabels) - num_train_classes, len(test_images)))
        #torch.save((test_images, test_targets),
        #        os.path.join(self.root, 'test.pt'))

    def unzip(self):
        print('Extracting kkangji.tar...')
        tar = tarfile.open(os.path.join(self.root, 'kkanji.tar'), "r")
        tar.extractall(path=self.root)
        tar.close()
        os.remove(os.path.join(self.root, 'kkanji.tar'))

    def __init__(self, root, train=True, transform=None):
        super().__init__()
        self.root = os.path.expanduser(root)
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        self.transform = transform

        self.check_raw_exists()
        if not self.check_if_preprocessed():
            self.preprocess()

        self.data, self.targets = torch.load(
                os.path.join(self.root,
                    'train.pt' if train else 'test.pt'))

        self.data = self.data.unsqueeze(-3).float().div(255)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

if __name__ == '__main__':

    from utils.paths import datasets_path
    kkanji = KKanji(os.path.join(datasets_path, 'kkanji'))
    from torchvision.utils import make_grid

    import matplotlib.pyplot as plt

    x, y = kkanji.data, kkanji.targets
    plt.imshow(make_grid(x[y==10][:30]).numpy().transpose(1,2,0))
    plt.show()
