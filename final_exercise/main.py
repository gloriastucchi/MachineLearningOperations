#S2 module I
import click
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from model import MyAwesomeModel
from data import mnist

@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # Get the training set
    train_set, _ = mnist()
    
    # Initialize the model
    model = MyAwesomeModel()

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    train_losses = []
    for epoch in range(30):
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
    torch.save(model, "trained_model.pt")
    
    # Plot training curve
    plt.plot(train_losses)
    plt.xlabel('Training Step')
    plt.ylabel('Training Loss')
    plt.title('Training Curve')
    plt.show()

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # Get the test set
    _, test_set = mnist()

    # Load the trained model
    model = torch.load(model_checkpoint)
    
    # Evaluation loop
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_set:
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = correct / total
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

cli.add_command(train)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
