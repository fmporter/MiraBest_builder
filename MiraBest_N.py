from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity


class MiraBest_full(data.Dataset):
    """
    Inspired by `HTRU1 <https://as595.github.io/HTRU1/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``MiraBest-N.py` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = 'batches'
    url = "http://www.jb.man.ac.uk/research/MiraBest/MiraBest_N/MiraBest_N_batches.tar.gz" 
    filename = "MiraBest_N_batches.tar.gz"
    tgz_md5 = '3ea385adfdc879eaeec0f77a766b5315'
    train_list = [
                  ['data_batch_1', '07bb10e2b6d94803032018855de1141d'],
                  ['data_batch_2', '1fd7393ad520ec302d54c187293ecb2b'],
                  ['data_batch_3', '99e2919b2a19443d69455cb024571732'],
                  ['data_batch_4', 'f9f4061353302ce097f522b12c9c9ee6'],
                  ['data_batch_5', 'a87ec87f19ed0bbf0ec4ec8e924437a1'],
                  ['data_batch_6', '45a93fa81381515c409e0011f95c7206'],
                  ['data_batch_7', '573a46dc90e746e8e7cb2b219f091cf1'],
                  ]

    test_list = [
                 ['test_batch', '7ec16472f4c9479afb54f6b3dc05c5bf'],
                 ]
    meta = {
                'filename': 'batches.meta',
                'key': 'label_names',
                'md5': 'e1b5450577209e583bc43fbf8e851965',
                }


    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.filenames = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                    self.filenames.extend(entry['filenames'])
                else:
                    self.targets.extend(entry['fine_labels'])
                    self.filenames.extend(entry['filenames'])


        self.data = np.vstack(self.data).reshape(-1, 1, 150, 150)
        self.data = self.data.transpose((0, 2, 3, 1))

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.reshape(img,(150,150))
        img = Image.fromarray(img,mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            #print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str