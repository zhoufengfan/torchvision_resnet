import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from backbone import Network2
from dataset import Cifar10


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
    batch_size = 32
    cycle_items_for_test = 50
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_dataset = Cifar10(is_train=True, transform=transform2)
    test_dataset = Cifar10(is_train=False, transform=transform2)

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
            if i % cycle_items_for_test == 1:
                print("acc:{:.6f}".format(evaluate(net, test_dataloader)))
