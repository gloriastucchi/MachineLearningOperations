import torch
from torch.utils.data import DataLoader


def mnist(data_folder='./corruptmnist', batch_size=64, num_workers=0):
    """Return train and test dataloaders for Corrupted MNIST."""
    
    # Load train data
    train_images = torch.cat([torch.load(f'{data_folder}/train_images_{i}.pt') for i in range(6)], dim=0)
    train_targets = torch.cat([torch.load(f'{data_folder}/train_target_{i}.pt') for i in range(6)], dim=0)
    
    # Combine images and targets into a single TensorDataset
    train_dataset = torch.utils.data.TensorDataset(train_images, train_targets)
    
    # Create a DataLoader for training data
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Load test data
    test_images = torch.load(f'{data_folder}/test_images.pt')
    test_targets = torch.load(f'{data_folder}/test_target.pt')
    
    # Combine test images and targets into a single TensorDataset
    test_dataset = torch.utils.data.TensorDataset(test_images, test_targets)
    
    # Create a DataLoader for test data
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader
