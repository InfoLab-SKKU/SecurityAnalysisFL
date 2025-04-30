from collections import OrderedDict
from typing import List, Tuple, Dict

import flwr as fl
import torch
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from flwr.common import Metrics
from flwr.common.typing import Scalar
from pyarrow import Scalar
from torch import optim
from torch.utils.data import DataLoader

from src.modules.attacks.attack import Benin
from src.modules.utils import test, apply_transforms, test_specific_class_with_exclusion, train
from src.modules.model import ModelFactory


def fit_config(epochs: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 2,  # Number of local epochs done by clients
        "batch_size": 64,  # Batch size to use by clients during fit()
    }
    return config


# Define type aliases for clarity
Metrics = Dict[str, float]

# Dictionary to track client accuracies across rounds
client_accuracies: Dict[int, List[float]] = {}


def update_client_metrics(metrics: List[Tuple[int, Metrics]]) -> None:
    """Update accuracy tracking for each client."""
    for client_id, m in metrics:
        if client_id not in client_accuracies:
            client_accuracies[client_id] = []
        client_accuracies[client_id].append(m["accuracy"])


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregation function for federated evaluation metrics.

    Args:
        metrics (List[Tuple[int, Metrics]]): List of tuples containing the number of examples
                                             and the metrics dictionary for each client.

    Returns:
        Metrics: Dictionary containing weighted averages of metrics.
    """
    # Update individual client metrics (optional, for logging or debugging)
    update_client_metrics(metrics)

    # Initialize accumulators
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    asrs = []
    examples = []
    accuracy_excluding_poisoned = []
    accuracy_targeted = []

    # Iterate over metrics to compute weighted contributions
    for num_examples, m in metrics:
        examples.append(num_examples)
        accuracies.append(num_examples * m.get("accuracy", 0.0))
        accuracy_excluding_poisoned.append(num_examples * m.get("accuracy_excluding_poisoned", 0.0))
        accuracy_targeted.append(num_examples * m.get("accuracy_targeted", 0.0))
        precisions.append(num_examples * m.get("precision", 0.0))
        recalls.append(num_examples * m.get("recall", 0.0))
        f1_scores.append(num_examples * m.get("f1_score", 0.0))
        asrs.append(num_examples * m.get("asr", 0.0))

    # Compute weighted averages
    total_examples = sum(examples)
    weighted_accuracy = sum(accuracies) / total_examples
    weighted_accuracy_excluding_poisoned = sum(accuracy_excluding_poisoned) / total_examples
    weighted_accuracy_targeted = sum(accuracy_targeted) / total_examples
    weighted_precision = sum(precisions) / total_examples
    weighted_recall = sum(recalls) / total_examples
    weighted_f1_score = sum(f1_scores) / total_examples
    weighted_asr = sum(asrs) / total_examples

    # Return aggregated metrics
    return {
        "accuracy": weighted_accuracy,
        "accuracy_excluding_poisoned": weighted_accuracy_excluding_poisoned,
        "accuracy_targeted": weighted_accuracy_targeted,
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1_score": weighted_f1_score,
        "asr": weighted_asr,
    }


def report_client_behavior():
    """Report the behavior of each client after training is complete."""
    print("Client Behavior Report:")
    for client_id, accuracies in client_accuracies.items():
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        max_accuracy = max(accuracies) if accuracies else 0
        min_accuracy = min(accuracies) if accuracies else 0
        print(f"Client {client_id}:")
        print(f"  Average Accuracy: {avg_accuracy:.4f}")
        print(f"  Max Accuracy: {max_accuracy:.4f}")
        print(f"  Min Accuracy: {min_accuracy:.4f}")






def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class ServerFactory:
    @staticmethod
    def get_evaluate_fn(centralized_testset: Dataset, conf):
        """Return an evaluation function for centralized evaluation."""

        def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]):
            """Use the test set for evaluation."""

            # Determine device
            model = ModelFactory.create_model(conf)

            set_params(model, parameters)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # Apply transform to dataset
            testset = centralized_testset.with_transform(apply_transforms)

            # Disable tqdm for dataset preprocessing
            disable_progress_bar()

            testloader = DataLoader(testset, batch_size=32)

            optimizer = optim.Adam(model.parameters(), lr=0.001)
            train(model, testloader, optimizer, Benin(),epochs=3, device=device)

            loss, accuracy, accuracy_excluding_poisoned, precision, recall, f1, asr, accuracy_target = test_specific_class_with_exclusion(model,
                                                                        testloader,
                                                                        device=device,
                                                                        specific_class=conf.poisoning.poison_label)

            return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1,
                          "accuracy_excluding_poisoned": accuracy_excluding_poisoned, "accuracy_target": accuracy_target, "asr": asr}

        return evaluate