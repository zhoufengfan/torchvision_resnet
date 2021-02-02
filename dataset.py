from torch.utils.data import Dataset
import torch
import pickle
import os
import numpy as np
import torchvision.transforms as transforms


class Cifar10(Dataset):
    def __init__(self, is_train, transform):
        self.cifar10_root_dir = r"/root/datasets/cifar-10-batches-py"
        self.is_train = is_train
        self.transform = transform
        self.imgs, self.labels = self.get_data_from_pickled_files(self.is_train)

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def get_data_from_pickled_files(self, is_train):
        imgs = []
        labels = []
        if is_train:
            for i in range(1, 6):
                dict_of_single_pickled_file = self.unpickle(
                    os.path.join(self.cifar10_root_dir, "data_batch_{}".format(str(i))))
                imgs.append(dict_of_single_pickled_file[b'data'])
                labels.extend(dict_of_single_pickled_file[b'labels'])
        else:
            dict_of_single_pickled_file = self.unpickle(
                os.path.join(self.cifar10_root_dir, "test_batch"))
            imgs.append(dict_of_single_pickled_file[b'data'])
            labels.extend(dict_of_single_pickled_file[b'labels'])

        imgs = np.vstack(imgs).reshape((-1, 3, 32, 32))
        imgs = imgs.transpose((0, 2, 3, 1))
        labels = np.array(labels)
        return imgs, labels

    def __getitem__(self, item):
        img, label = self.imgs[item], self.labels[item]
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    cifar10 = Cifar10(is_train=False, transform=transform2)
    d1 = cifar10.unpickle(os.path.join(cifar10.cifar10_root_dir, "test_batch"))
    print("d1.keys() is", d1.keys())
    print("type(d1[b'batch_label']) is", type(d1[b'batch_label']))
    print("type(d1[b'data']) is", type(d1[b'data']))
    print("d1[b'data'].shape is", d1[b'data'].shape)
    print("type(d1[b'labels']) is", type(d1[b'labels']))
    print("len(d1[b'labels']) is", len(d1[b'labels']))
    print("len(cifar10.labels) is", len(cifar10.labels))
