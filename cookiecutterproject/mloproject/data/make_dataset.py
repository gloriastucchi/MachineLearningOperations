import torch
from torch.utils.data import TensorDataset  
from torchvision import datasets, transforms

def process_data(data_folder='../MachineLearningOperations/mloproject/data/raw', output_folder='../MachineLearningOperations/mloproject/data//processed'):
    # Load corrupted MNIST data
    train_images = torch.cat([torch.load(f'{data_folder}/train_images_{i}.pt') for i in range(10)], dim=0)
    train_target = torch.cat([torch.load(f'{data_folder}/train_target_{i}.pt') for i in range(10)], dim=0)
    test_images = torch.load(f'{data_folder}/test_images.pt')
    test_target = torch.load(f'{data_folder}/test_target.pt')

    # Concatenate train and test data for normalization
    all_images = torch.cat([train_images, test_images], dim=0)

    # Calculate mean and standard deviation for normalization
    mean = all_images.mean()
    std = all_images.std()

    # Normalize the data
    normalize = transforms.Normalize(mean=[mean], std=[std])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Apply normalization to train and test data
    train_dataset = TensorDataset(train_images, train_target)  # Change this line
    test_dataset = TensorDataset(test_images, test_target)  # Change this line

    # Save processed data
    torch.save(train_dataset, f'{output_folder}/train_dataset.pt')
    torch.save(test_dataset, f'{output_folder}/test_dataset.pt')

if __name__ == '__main__':
    process_data()
