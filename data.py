import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

class IndexedMNIST(datasets.MNIST):
    def __getitem__(self, index):
        """
        Overrides the default __getitem__ to also return the index.
        This is critical for TDDS to save the scores for the correct sample.
        """
        img, target = self.data[index], int(self.targets[index])

        # Convert to PIL Image to be compatible with standard torchvision transforms
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # RETURN INDEX ALONG WITH IMG AND TARGET
        return img, target, index

########################################################################################################################
# Load Data
########################################################################################################################

def load_data(args):
    """
    Load data for training and testing.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    train_loader, test_loader = load_dataset(args)
    return train_loader, test_loader

def load_dataset(args):
    """
    Load dataset based on the specified dataset in args.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    if args.dataset == 'cifar10':
        train_loader, test_loader = load_cifar10(args)
    elif args.dataset == 'cifar100':
        train_loader, test_loader = load_cifar100(args)
    elif args.dataset == 'mnist':
        num_classes = 10
        
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
    
        train_dataset = IndexedMNIST(root=args.data_path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    else:
        raise NotImplementedError("Dataset not supported: {}".format(args.dataset))
    return train_loader, test_loader

def load_cifar10(args):
    """
    Load CIFAR-10 dataset.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    print('Loading CIFAR-10... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    target_index = [[train_data.targets[i], i] for i in range(len(train_data.targets))]
    train_data.targets = target_index
    
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    
    print(f"done in {time.time() - time_start:.2f} seconds.")
    return train_loader, test_loader

def load_cifar100(args):
    """
    Load CIFAR-100 dataset.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    print('Loading CIFAR-100... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
    target_index = [[train_data.targets[i], i] for i in range(len(train_data.targets))]
    train_data.targets = target_index
    
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    
    print(f"done in {time.time() - time_start:.2f} seconds.")
    return train_loader, test_loader
