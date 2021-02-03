import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from backbone import Network2
from dataset import Cifar10

# def is_different(my_)

if __name__ == '__main__':
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
