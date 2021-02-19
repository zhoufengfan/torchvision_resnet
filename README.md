# torchvision_resnet
Some test on the torchvision.resnet

Before using the code, please download the CIFAR10 dataset. Then change the path in dataset to the real dataset path in your disk.

All results are without data augment.

id|datset | dataset downloaded manually | shuffled | acc|fluctuation of curve
:--------------:|:--------------:|:---------:|:-------:|:-------:|:-------:
0|torchvision(test in trainset)   | no | yes |99%|small
1|mine  | yes  | yes | 60%|small
2|torchvision(test in trainset)   | no | no |99%|small
3|mine  | yes  | no | 60%|small
4|torchvision(test in trainset)   | yes | no |99%|small
5|mine inherit VisionDataset   | yes | no |68%|large
6|torchvision(test in testset)   | yes | no |68%|small