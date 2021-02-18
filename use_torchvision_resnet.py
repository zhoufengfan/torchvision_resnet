import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from backbone import Network2
from dataset import Cifar10
import os

from util import init_log
import logging
from rich.progress import track
from tensorboardX import SummaryWriter


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


def run(output_files_dir2):
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
        root='/root/datasets/', train=True, download=True, transform=transform2)
    # test_dataset = Cifar10(is_train=False, transform=transform3)
    test_dataset = torchvision.datasets.CIFAR10(
        root='/root/datasets/', train=True, download=True, transform=transform3)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    net = Network2()
    sw = SummaryWriter(comment='net2', logdir=os.path.join(output_files_dir2, "tfbd_log"))
    ramdom_input = torch.rand(12, 3, 32, 32)
    sw.add_graph(net, (ramdom_input,))
    if torch.cuda.is_available():
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1,
    #                             momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
    best_acc = 0
    begin_epoch = 0
    n_item = 0
    pth_dir_path = output_files_dir2
    pth_name = "best.pth"
    pth_path = os.path.join(pth_dir_path, pth_name)
    if os.path.exists(pth_path):
        checkpoint = torch.load(pth_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        begin_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        n_item = checkpoint['n_item']

    for epoch in range(begin_epoch, begin_epoch + 512):
        for i, (data_batch, label_batch) in track(total=len(train_dataset) // batch_size + 1,
                                                  sequence=enumerate(train_dataloader),
                                                  description="epoch:{}".format(epoch)):
            data_batch = data_batch.cuda()
            feature_vector = net(data_batch)
            label_batch = label_batch.cuda()
            optimizer.zero_grad()
            loss = criterion(feature_vector, label_batch)
            if n_item % 50 == 1:
                sw.add_scalar('loss_curve', loss, n_item)
            loss.backward()
            optimizer.step()
            n_item = n_item + 1
        if epoch % cycle_epoches_for_test == 1:
            acc_now = evaluate(net, test_dataloader)
            sw.add_scalar('acc_curve', acc_now, epoch)
            logging.info("epoch:{}\tn_item:{}\tacc:{:.6f}".format(epoch, n_item, acc_now))

            if acc_now > best_acc:
                state = {
                    'net': net.state_dict(),
                    'acc': acc_now,
                    'epoch': epoch,
                    "optimizer": optimizer.state_dict(),
                    'n_item': n_item,
                }
                if not os.path.exists(pth_dir_path):
                    os.mkdir(pth_dir_path)
                torch.save(state, pth_path)
    sw.close()


if __name__ == '__main__':
    output_files_dir = r"/data/cifar/10"
    init_log(os.path.join(output_files_dir,'experiment_result.log'))
    run(output_files_dir)
