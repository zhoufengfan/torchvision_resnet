# torchvision_resnet
Some test on the torchvision.resnet

Before using the code, please download the CIFAR10 dataset. Then change the path in dataset to the real dataset path in your disk.

All results are without data augment.

id|datset | dataset downloaded manually | shuffled| data aug | acc|fluctuation of curve
:--------------:|:--------------:|:---------:|:-------:|:-------:|:-------:|:-------:
0|torchvision(test in trainset)   | no | yes | none |99%|small
1|mine  | yes  | yes | none| 60%|small
2|torchvision(test in trainset)   | no | no | none|99%|small
3|mine  | yes  | no | none| 60%|small
4|torchvision(test in trainset)   | yes | no | none|99%|small
5|mine inherit VisionDataset   | yes | no | none|68%|large
6|torchvision(test in testset)   | yes | no | none|68%|large
7|torchvision(test in testset)   | yes | yes | none|%|
8|torchvision(test in testset)   | yes | yes | RandomCrop(32, padding=4) RandomHorizontalFlip()|%|