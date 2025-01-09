from typing import Tuple

import torch
import torch.nn as nn
from flwr.client.mod import LocalDpMod
from torchvision.transforms import Compose, Normalize, ToTensor, Grayscale, Resize
from torch.optim import Optimizer
from torch.utils.data import DataLoader



# Transformation to convert images to tensors and apply normalization
def apply_transforms(batch: dict) -> dict:

    transform = Compose([
        #Resize((224, 224)),
        #Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        ToTensor(),
        #Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean and std normalization
    ])

    # Apply transformations
    batch["image"] = [transform(img) for img in batch["image"]]
    return batch


def get_local_dp(config) -> LocalDpMod:
    """
    Create and return a Local Differential Privacy (DP) module using the given config.

    Args:
        config: Configuration object containing DP parameters like clipping_norm, sensitivity, epsilon, and delta.

    Returns:
        LocalDpMod: A local DP module for applying differential privacy during training.
    """
    return LocalDpMod(
        config.clipping_norm,
        config.sensitivity,
        config.epsilon,
        config.delta
    )


# Borrowed from Pytorch quickstart example
def train(net: nn.Module, trainloader: DataLoader, optim: Optimizer, attack, epochs: int, device: str):
    """
    Train the neural network on the training dataset.

    Args:
        attack:
        net (nn.Module): The neural network model to train.
        trainloader (DataLoader): DataLoader providing batches of training data.
        optim (Optimizer): Optimizer used to update model weights.
        epochs (int): Number of training epochs.
        device (str): Device to perform training on ('cuda' or 'cpu').
    """
    criterion = nn.CrossEntropyLoss()
    net.train()

    for epoch in range(epochs):
        for batch in trainloader:
            # Move data to device
            images, labels = batch["image"].to(device), batch["label"].to(device)
            images, labels = attack.on_batch_selection(images, labels)
            # Zero the parameter gradients
            optim.zero_grad()

            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            net, loss = attack.on_before_backprop(net, loss)
            loss.backward()

            optim.step()

            net, loss = attack.on_after_backprop(net, loss)



# Borrowed from Pytorch quickstart example
def test(net: nn.Module, testloader: DataLoader, device: str) -> Tuple[float, float]:
    """
    Evaluate the neural network on the test dataset.

    Args:
        net (nn.Module): The neural network model to evaluate.
        testloader (DataLoader): DataLoader providing batches of test data.
        device (str): Device to perform testing on ('cuda' or 'cpu').

    Returns:
        Tuple[float, float]: Tuple containing the test loss and accuracy.
    """
    criterion = nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0
    net.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for batch in testloader:
            # Move data to device
            images, labels = batch["image"].to(device), batch["label"].to(device)

            # Forward pass
            outputs = net(images)
            total_loss += criterion(outputs, labels).item()

            # Predictions
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = correct / len(testloader.dataset)
    return total_loss, accuracy
