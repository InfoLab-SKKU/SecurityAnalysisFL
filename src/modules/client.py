from collections import OrderedDict
from typing import List
import flwr as fl
import numpy as np
import torch

from flwr_datasets import FederatedDataset
from torch import optim
from torch.utils.data import DataLoader

from src.modules.attack import Benin, AttackFactory
from src.modules.utils import train, test, apply_transforms
from src.modules.model import ModelFactory


def add_laplace_noise(tensor: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Add Laplace noise to the tensor for differential privacy."""
    if epsilon == 0.0:
        return tensor
    sensitivity = 0.01  # Sensitivity is usually 1 for gradients
    scale = sensitivity / epsilon
    noise = torch.from_numpy(np.random.laplace(0, scale, tensor.shape)).to(tensor.device)
    return tensor + noise


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, config, client_id=None):
        self.client_id = client_id
        self.trainset = trainset
        self.valset = valset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the model
        self.model = ModelFactory.create_model(config).to(self.device)
        self.epsilon = config.ldp.epsilon

        fraction = int(config.poisoning.fraction)
        r = np.random.random(1)[0]
        if r < fraction:
            print(f"client {client_id} is poisoned------------------------------------")
            self.attack = AttackFactory.create_attack(config)
        else:
            print(f"client {client_id} is Benin----------------------------------------")
            self.attack = Benin()

    def get_parameters(self, config=None):
        """Retrieve model parameters as NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def evaluate(self, parameters, config):
        """Evaluate the model using the validation dataset."""
        set_params(self.model, parameters)
        valloader = DataLoader(self.valset, batch_size=64)

        loss, accuracy = test(self.model, valloader, device=self.device)
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}

    def fit(self, parameters, config):
        raise NotImplementedError("fit method must be implemented in subclasses.")


class FedAvgClient(FlowerClient):
    def fit(self, parameters, config):
        set_params(self.model, parameters)

        batch, epochs = config["batch_size"], config["epochs"]
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Train the model
        train(self.model, trainloader, optimizer, self.attack,epochs=epochs, device=self.device)

        # Add Laplace noise to model parameters for privacy
        for name, param in self.model.named_parameters():
            param.data = add_laplace_noise(param.data, self.epsilon)

        return self.get_parameters({}), len(trainloader.dataset), {}


class FedNovaClient(FlowerClient):
    def fit(self, parameters, config):
        set_params(self.model, parameters)

        batch, epochs = config["batch_size"], config["epochs"]
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Train the model
        train(self.model, trainloader, optimizer, self.attack,epochs=epochs, device=self.device)

        # Add Laplace noise to model parameters for privacy
        for name, param in self.model.named_parameters():
            param.data = add_laplace_noise(param.data, self.epsilon)

        return self.get_parameters({}), len(trainloader.dataset), {}


def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_client_fn(dataset: FederatedDataset, num_classes):
    """Return a function to construct a client."""
    def client_fn(context) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""
        client_dataset = dataset.load_partition(
            int(context.node_config["partition-id"]), "train"
        )
        client_dataset_splits = client_dataset.train_test_split(test_size=0.1, seed=42)
        trainset = client_dataset_splits["train"]
        valset = client_dataset_splits["test"]

        trainset = trainset.with_transform(apply_transforms)
        valset = valset.with_transform(apply_transforms)

        return FlowerClient(trainset, valset, num_classes).to_client()

    return client_fn


class ClientFactory:
    @staticmethod
    def get_client_fn(dataset: FederatedDataset, conf):
        """Return a function to construct a client."""
        def client_fn(context) -> fl.client.Client:
            client_id = int(context.node_config["partition-id"])
            client_dataset = dataset.load_partition(
                client_id, "train"
            )
            client_dataset_splits = client_dataset.train_test_split(test_size=0.1, seed=42)
            trainset = client_dataset_splits["train"]
            valset = client_dataset_splits["test"]

            trainset = trainset.with_transform(apply_transforms)
            valset = valset.with_transform(apply_transforms)

            if conf.strategy.name in ["FedAvg", "FedAvgM", "FedProx"]:
                return FedAvgClient(trainset, valset, conf, client_id).to_client()
            elif conf.strategy.name == "FedNova":
                return FedNovaClient(trainset, valset, conf, client_id).to_client()
            else:
                raise ValueError(f"Unsupported Algorithm name: {conf.model.name}")

        return client_fn