from torch.utils.data import Dataset
import torch
import pickle


class Cifar10(Dataset):
    def __init__(self):
        self.cifar10_root_dir = r"C:\Users\zff\dataset\cifar-10-batches-py"
        

    def unpickle(self,file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict


if __name__ == '__main__':
    cifar10=Cifar10()