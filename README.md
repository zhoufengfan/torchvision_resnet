# torchvision_resnet
Some test on the torchvision.resnet

Before using the code, please download the CIFAR10 dataset. Then change the path in dataset to the real dataset path in your disk.

Acc of `1454e48f82e279707f33b2872a14857d4b00f726` in`linux` branch has reached 99% when epoch is 201.

All results are without data augment.

datset | dataset downloaded manually | shuffled | acc
:--------------:|:---------:|:-------:|:-------:
torchvision   | no | yes |99%
mine  | yes  | yes | 60%
torchvision   | no | no |99%
mine  | yes  | no | 60%
torchvision   | yes | no |%