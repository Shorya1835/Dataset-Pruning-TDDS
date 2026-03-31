import numpy as np
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from multiprocessing import Pool
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class SubsetMNIST(datasets.MNIST):
    def __init__(self, root, indices, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        # Subset the data and targets based on the TDDS mask indices
        self.data = self.data[indices]
        self.targets = self.targets[indices]
        
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

########################################################################################################################
#  Load Data
########################################################################################################################

def load_cifar10_sub(args, data_mask, sorted_score):
    """
    Load CIFAR10 dataset with specified transformations and subset selection.
    """
    print('Loading CIFAR10... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    score = (sorted_score - min(sorted_score)) / (max(sorted_score) - min(sorted_score))
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    z = [[train_data.targets[i], score[np.where(data_mask == i)]] for i in range(len(train_data.targets))]
    train_data.targets = z

    subset_mask = data_mask[int(args.subset_rate * len(data_mask)):]
    data_set = torch.utils.data.Subset(train_data, subset_mask)

    train_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader

def load_cifar100_sub(args, data_mask, sorted_score):
    """
    Load CIFAR100 dataset with specified transformations and subset selection.
    """
    print('Loading CIFAR100... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
    
    score = (sorted_score - min(sorted_score)) / (max(sorted_score) - min(sorted_score))
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
    z = [[train_data.targets[i], score[np.where(data_mask == i)]] for i in range(len(train_data.targets))]
    train_data.targets = z

    subset_mask = data_mask[int(args.subset_rate * len(data_mask)):]
    data_set = torch.utils.data.Subset(train_data, subset_mask)

    train_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)

def load_mnist_sub(args, data_mask, sorted_score):
    # 1. Calculate how many samples to keep based on the subset_rate
    num_samples = int(60000 * args.subset_rate)
    
    # 2. Extract the indices to keep. 
    # (Check how load_cifar10_sub does this in the original code, 
    # but it usually takes the top 'num_samples' from sorted_score)
    keep_indices = sorted_score[:num_samples] 
    
    # 3. Apply the 3-channel, 32x32 transformations
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307, 0.1307, 0.1307), std=(0.3081, 0.3081, 0.3081)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307, 0.1307, 0.1307), std=(0.3081, 0.3081, 0.3081)),
    ])

    # 4. Instantiate the custom Subset dataset
    train_dataset = SubsetMNIST(root=args.data_path, indices=keep_indices, train=True, download=True, transform=transform_train)
    
    # The test set remains the full 10,000 images, so we use the standard MNIST class
    test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform_test)

    # 5. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    return train_loader, test_loader
    ])

    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader
