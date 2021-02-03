import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from backbone import Network2
from dataset import Cifar10

if __name__ == '__main__':
    batch_size = 128
    my_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    torchvisions_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    my_train_dataset = Cifar10(is_train=True, transform=my_transform)

    torchvisions_train_dataset = torchvision.datasets.CIFAR10(
        root='../dataset', train=True, download=True, transform=torchvisions_transform)
    my_train_dataloader = torch.utils.data.DataLoader(
        dataset=my_train_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    torchvisions_train_dataloader = torch.utils.data.DataLoader(
        dataset=torchvisions_train_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    # print("my_train_dataloader[0] is", my_train_dataloader[0])
    my_train_dataloader_list = [[imgs, labels] for imgs, labels in my_train_dataloader]
    torchvisions_train_dataloader_list = [[imgs, labels] for imgs, labels in torchvisions_train_dataloader]
    for imgs, labels in my_train_dataloader:
        print("imgs.shape is", imgs.shape)
        print("labels.shape is", labels.shape)
        exit()
