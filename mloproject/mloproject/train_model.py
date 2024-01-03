import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from model import MyAwesomeModel
from data import load_processed_data

def train_model(lr, epochs):
    # Get the processed data
    train_set, _ = load_processed_data()

    # Initialize the model
    model = MyAwesomeModel()

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    train_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_set:
            optimizer.zero_grad()
            # Forward pass
            output = model(images)
            # Calculate the loss
            loss = criterion(output, labels)
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_set))

    # Save the trained model
    model_folder = "models"
    torch.save(model.state_dict(), f"{model_folder}/trained_model.pth")

    # Plot training curve and save as .png
    plt.plot(train_losses)
    plt.xlabel('Training Step')
    plt.ylabel('Training Loss')
    plt.title('Training Curve')
    figure_folder = "reports/figures"
    plt.savefig(f"{figure_folder}/training_curve.png")

if __name__ == '__main__':
    lr = 0.001
    epochs = 10
    train_model(lr, epochs)

