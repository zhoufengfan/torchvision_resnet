# torchvision_resnet
Some test on the torchvision.resnet

Before using the code, please download the CIFAR10 dataset. Then change the path in dataset to the real dataset path in your disk.

Acc of `1454e48f82e279707f33b2872a14857d4b00f726` in`linux` branch has reached 99% when epoch is 201.

All results are without data augment.

id|datset | dataset downloaded manually | shuffled | acc
:--------------:|:--------------:|:---------:|:-------:|:-------:
0|torchvision   | no | yes |99%
1|mine  | yes  | yes | 60%
2|torchvision   | no | no |99%
3|mine  | yes  | no | 60%
4|torchvision   | yes | no |99%
5|mine inherit VisionDataset   | yes | no |%