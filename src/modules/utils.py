from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.client.mod import LocalDpMod
from torchvision.transforms import Compose, Normalize, ToTensor, Grayscale, Resize
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torchvision.transforms.functional import to_pil_image


def model2vector(model):
    nparr = np.array([])
    for key, var in model.items():
        nplist = var.cpu().numpy()
        nplist = nplist.ravel()
        nparr = np.append(nparr, nplist)
    return nparr

def cosine_similarity(server_params, client_params):
    s = model2vector(server_params)[0].flatten()
    c = model2vector(client_params)[0].flatten()
    return np.dot(s, c) / (np.linalg.norm(s) * np.linalg.norm(c) + 1e-8)

def revert_transforms(batch: dict) -> dict:
    # Convert tensors back to PIL images
    batch["image"] = [to_pil_image(tensor) for tensor in batch["image"]]
    return batch

def apply_transforms_0(batch: dict) -> dict:
    transform = Compose([
        ToTensor(),
    ])
    batch["image"] = [transform(img) for img in batch["image"]]
    return batch



# Transformation to convert images to tensors and apply normalization
def apply_transforms(batch: dict) -> dict:
    """
    Apply transformations to the batch of images, including resizing, grayscale conversion,
    tensor conversion, and normalization.

    Args:
        batch (dict): Batch of images and labels where 'image' is a list of images.

    Returns:
        dict: Batch with transformed images.
    """
    transform = Compose([
        Resize((224, 224)),
        Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean and std normalization
    ])

    # Apply transformations
    #print("image type=================", type(batch["image"]))
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
    print('num epochs', epochs)
    for epoch in range(epochs):
        for batch in trainloader:
            # Move data to device
            images, labels = batch["image"].to(device), batch["label"].to(device)
            images, labels = attack.on_batch_selection(net, device, images, labels)
            # Zero the parameter gradients
            optim.zero_grad()

            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            net, loss = attack.on_before_backprop(net, loss)
            loss.backward()

            optim.step()

            #net, loss = attack.on_after_backprop(net, loss)


def test(net: nn.Module, testloader: DataLoader, device: str) -> Tuple[float, float, float, float, float]:
    """
    Evaluate the neural network on the test dataset and calculate metrics.

    Args:
        net (nn.Module): The neural network model to evaluate.
        testloader (DataLoader): DataLoader providing batches of test data.
        device (str): Device to perform testing on ('cuda' or 'cpu').

    Returns:
        Tuple[float, float, float, float, float]: Test loss, accuracy, precision, recall, and F1 score.
    """
    criterion = nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0
    y_true = []
    y_pred = []

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

            # Append predictions and true labels for metric calculations
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate metrics
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")

    # Calculate accuracy
    accuracy = correct / len(testloader.dataset)
    return total_loss, accuracy, precision, recall, f1

from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader



def test_specific_class_with_exclusion(
        net: nn.Module, testloader: DataLoader, device: str, specific_class: int
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Evaluate the neural network on the test dataset and calculate metrics,
    including normal accuracy and accuracy excluding the poisoned class.

    Args:
        net (nn.Module): The neural network model to evaluate.
        testloader (DataLoader): DataLoader providing batches of test data.
        device (str): Device to perform testing on ('cuda' or 'cpu').
        specific_class (int): The poisoned class to exclude from accuracy calculations.

    Returns:
        Tuple[float, float, float, float, float, float, float, float]:
            Test loss, normal accuracy, accuracy excluding poisoned class, precision, recall, F1 score, ASR.
    """
    criterion = nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0
    y_true = []
    y_pred = []

    correct_class = 0  # Correct predictions for specific class
    total_class = 0  # Total samples for specific class
    excluded_correct = 0  # Correct predictions excluding poisoned class
    excluded_total = 0  # Total samples excluding poisoned class

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

            # Append predictions and true labels for metric calculations
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # Exclude the poisoned class from general accuracy
            for true, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                if true != specific_class:  # Exclude poisoned class
                    excluded_total += 1
                    if true == pred:
                        excluded_correct += 1

                # Specific class metrics
                if true == specific_class:
                    total_class += 1
                    if pred == specific_class:
                        correct_class += 1

    # Calculate metrics
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # General accuracy (normal)
    normal_accuracy = correct / len(testloader.dataset)

    # Accuracy excluding poisoned class
    accuracy_excluding_poisoned = excluded_correct / excluded_total if excluded_total > 0 else 0.0

    # Specific class ASR (attack success rate)
    specific_class_acc = (correct_class / total_class) if total_class > 0 else 0.0
    asr = accuracy_excluding_poisoned - specific_class_acc
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
    print(f"Normal Accuracy: {normal_accuracy * 100:.2f}%")
    print(f"Accuracy Excluding Poisoned Class: {accuracy_excluding_poisoned * 100:.2f}%")
    print(f"Specific Class {specific_class} Acc: {specific_class_acc * 100:.2f}%")
    print(f"ASR: {asr * 100:.2f}%")

    return total_loss, normal_accuracy, accuracy_excluding_poisoned, precision, recall, f1, asr, specific_class_acc


def test_with_attack_success_rate(
    net: nn.Module,
    testloader: DataLoader,
    device: str,
    poison_label: int,
    target_label: int
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Evaluate the neural network on the test dataset and calculate metrics, including
    false positive rate, clean accuracy, and attack success rate.

    Args:
        net (nn.Module): The neural network model to evaluate.
        testloader (DataLoader): DataLoader providing batches of test data.
        device (str): Device to perform testing on ('cuda' or 'cpu').
        poison_label (int): Label used to identify poisoned samples.
        target_label (int): Target label attackers aim to induce via poisoning.

    Returns:
        Tuple[float, float, float, float, float, float, float, float]:
        Test loss, accuracy, precision, recall, F1 score, false positive rate,
        clean accuracy, and attack success rate.
    """
    criterion = nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0
    y_true = []
    y_pred = []
    poison_samples, poisoned_success = 0, 0
    clean_correct, clean_samples = 0, 0

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

            # Append predictions and true labels for metric calculations
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # Count poisoned sample metrics
            poison_mask = labels == poison_label
            poison_samples += poison_mask.sum().item()
            poisoned_success += (predicted[poison_mask] == target_label).sum().item()

            # Count clean sample metrics
            clean_mask = labels != poison_label
            clean_samples += clean_mask.sum().item()
            clean_correct += (predicted[clean_mask] == labels[clean_mask]).sum().item()

    # Calculate metrics
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    # Calculate confusion matrix for FPR
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)  # For binary classification
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Calculate accuracy
    accuracy = correct / len(testloader.dataset)

    # Calculate clean accuracy
    clean_accuracy = clean_correct / clean_samples if clean_samples > 0 else 0.0

    # Calculate attack success rate
    asr = poisoned_success / poison_samples if poison_samples > 0 else 0.0

    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}, FPR: {fpr}, Clean Accuracy: {clean_accuracy}, ASR: {asr}")

    return total_loss, accuracy, precision, recall, f1, fpr, clean_accuracy, asr
