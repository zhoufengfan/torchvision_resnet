import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from backbone import Network2
from dataset import Cifar10
import os


def evaluate(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)

    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, y).sum().float().item()
    return correct / total


if __name__ == '__main__':
    num_epoch = 500
    data_vector_dim = 20
    item_of_single_class = 10
    batch_size = 128
    cycle_epoches_for_test = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # transform2 = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])
    transform2 = transforms.Compose([

        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform3 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # train_dataset = Cifar10(is_train=True, transform=transform2)
    train_dataset = torchvision.datasets.CIFAR10(
        root='../dataset', train=True, download=True, transform=transform2)
    # test_dataset = Cifar10(is_train=False, transform=transform3)
    test_dataset = torchvision.datasets.CIFAR10(
        root='../dataset', train=True, download=True, transform=transform3)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    net = Network2()

    # if torch.cuda.device_count() > 1:
    #     net = torch.nn.DataParallel(net, device_ids=os.environ["CUDA_VISIBLE_DEVICES"])
    if torch.cuda.is_available():
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1,
    #                             momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
    n_item = 0
    for epoch in range(num_epoch):
        for i, (data_batch, label_batch) in enumerate(train_dataloader):
            data_batch = data_batch.cuda()
            feature_vector = net(data_batch)
            label_batch = label_batch.cuda()
            optimizer.zero_grad()
            loss = criterion(feature_vector, label_batch)
            loss.backward()
            optimizer.step()
            n_item = n_item + 1
        if epoch % cycle_epoches_for_test == 1:
            print("epoch:{}\tn_item:{}\tacc:{:.6f}".format(epoch, n_item, evaluate(net, test_dataloader)))
        # scheduler.step()
