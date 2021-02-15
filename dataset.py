from torch.utils.data import Dataset
import torch
import pickle
import os
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torchvision


class Cifar10(Dataset):
    def __init__(self, is_train, transform):
        self.cifar10_root_dir = r"/root/datasets/cifar-10-batches-py"
        self.is_train = is_train
        self.transform = transform
        self.imgs, self.labels = self.get_data_from_pickled_files(self.is_train)

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict

    def get_data_from_pickled_files(self, is_train):
        imgs = []
        labels = []
        if is_train:
            for i in range(1, 6):
                dict_of_single_pickled_file = self.unpickle(
                    os.path.join(self.cifar10_root_dir, "data_batch_{}".format(str(i))))
                imgs.append(dict_of_single_pickled_file['data'])
                labels.extend(dict_of_single_pickled_file['labels'])
        else:
            dict_of_single_pickled_file = self.unpickle(
                os.path.join(self.cifar10_root_dir, "test_batch"))
            imgs.append(dict_of_single_pickled_file['data'])
            labels.extend(dict_of_single_pickled_file['labels'])

        imgs = np.vstack(imgs).reshape((-1, 3, 32, 32))
        imgs = imgs.transpose((0, 2, 3, 1))
        labels = np.array(labels)
        return imgs, labels

    def __getitem__(self, item):
        img, label = self.imgs[item], self.labels[item]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.labels)


def cal_the_difference_of_mine_and_torchvisions_cifar10():
    batch_size = 128
    # torch.backends.cudnn.deterministic = True
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    my_transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    torchvisions_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    my_train_dataset = Cifar10(is_train=False, transform=my_transform)
    torchvisions_train_dataset = torchvision.datasets.CIFAR10(
        root='../dataset', train=False, download=True, transform=torchvisions_transform)
    my_train_dataloader = torch.utils.data.DataLoader(
        dataset=my_train_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    torchvisions_train_dataloader = torch.utils.data.DataLoader(
        dataset=torchvisions_train_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    # print("my_train_dataloader[0] is", my_train_dataloader[0])
    my_train_dataloader_list = [[imgs, labels] for imgs, labels in my_train_dataloader]
    torchvisions_train_dataloader_list = [[imgs, labels] for imgs, labels in torchvisions_train_dataloader]
    for i in range(len(my_train_dataloader_list)):
        # print(torch.all(my_train_dataloader_list[i][0] == torchvisions_train_dataloader_list[i][0]))
        assert torch.all(
            my_train_dataloader_list[i][0] == torchvisions_train_dataloader_list[i][0]), "{}th img is different".format(
            i)
        assert torch.all(
            my_train_dataloader_list[i][1] == torchvisions_train_dataloader_list[i][
                1]), "{}th label is different".format(
            i)
    # The line below will be executed.
    print("The data in my_train_dataloader_list and torchvisions_train_dataloader_list are the same.")


def test_of_official_cifar():
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


if __name__ == '__main__':
    cal_the_difference_of_mine_and_torchvisions_cifar10()
