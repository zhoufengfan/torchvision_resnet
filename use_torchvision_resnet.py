import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from backbone import Network2


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
    model = torchvision.models.resnet50()
    # dataset = torchvision.datasets.cifar.CIFAR10(root=r"C:\Users\zff\dataset\cifar-10-batches-py", train=True)
    num_epoch = 500
    data_vector_dim = 20
    item_of_single_class = 10
    batch_size = 32
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.cifar.CIFAR10(root=r"C:\Users\zff\dataset\cifar-10-batches-py", train=True,
                                                       download=True, transform=transform2)
    test_dataset = torchvision.datasets.cifar.CIFAR10(root=r"C:\Users\zff\dataset\cifar-10-batches-py", train=False,
                                                      download=True, transform=transform2)
    # class_num = len(train_dataset.noise_scope_list)
    # dataset_length = item_of_single_class * class_num
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True
    )

    net = Network2()
    if torch.cuda.is_available():
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    for epoch in range(num_epoch):
        for i, (data_batch, label_batch) in enumerate(train_dataloader):
            data_batch = data_batch.cuda()
            real_out = net(data_batch)
            label_batch = label_batch.cuda()
            loss = criterion(real_out, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("acc:{:.6f}".format(evaluate(net, test_dataloader)))
