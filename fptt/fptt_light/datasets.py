import math
import numpy as np
import torch
from torchvision import datasets, transforms

def data_generator(dataset, batch_size, dataroot, shuffle=True):
    n_classes = 10
    seq_length = -1
    input_channels = -1
    if dataset == 'CIFAR-10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_set = datasets.CIFAR10(root=dataroot, train=True,
                                     download=True, transform=transform)
        test_set = datasets.CIFAR10(root=dataroot, train=False,
                                    download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   shuffle=shuffle, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                  shuffle=False, num_workers=0)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        n_classes = 10
        seq_length = 32*32
        input_channels = 3
    elif dataset == 'MNIST-10':
        # train_set = datasets.MNIST(root=dataroot, train=True, download=True,
        #                            transform=transforms.Compose([
        #                                transforms.ToTensor(),
        #                                transforms.Normalize((0.1307,), (0.3081,))
        #                            ]))
        # test_set = datasets.MNIST(root=dataroot, train=False, download=True,
        #                           transform=transforms.Compose([
        #                               transforms.ToTensor(),
        #                               transforms.Normalize((0.1307,), (0.3081,))
        #                           ]))
        train_set = datasets.MNIST(root=dataroot, train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor()
                                   ]))
        test_set = datasets.MNIST(root=dataroot, train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor()
                                  ]))

        train_loader = torch.utils.data.DataLoader(train_set, shuffle=shuffle, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=batch_size)
        n_classes = 10
        seq_length = 28*28
        input_channels = 1 
    else:
        print('Please provide a valid dataset name.')
        exit(1)
    return train_loader, test_loader, seq_length, input_channels, n_classes


def adding_problem_generator(N, seq_len=6, high=1, number_of_ones=2): 
    X_num = np.random.uniform(low=0, high=high, size=(N, seq_len, 1))
    X_mask = np.zeros((N, seq_len, 1))
    Y = np.ones((N, 1))
    for i in range(N):
        # Default uniform distribution on position sampling
        positions1 = np.random.choice(np.arange(math.floor(seq_len/2)), size=math.floor(number_of_ones/2), replace=False)
        positions2 = np.random.choice(np.arange(math.ceil(seq_len/2), seq_len), size=math.ceil(number_of_ones/2), replace=False)

        positions = []
        positions.extend(list(positions1))
        positions.extend(list(positions2))
        positions = np.array(positions)

        X_mask[i, positions] = 1        
        Y[i, 0] = np.sum(X_num[i, positions])
    X = np.append(X_num, X_mask, axis=2)
    return torch.FloatTensor(X), torch.FloatTensor(Y)

