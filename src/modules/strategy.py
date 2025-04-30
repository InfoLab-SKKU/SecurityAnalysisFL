from typing import Tuple, List

import flwr as fl
import numpy as np
from flwr.common import FitRes, Parameters
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, aggregate_median

from src.modules.server import fit_config, weighted_average, ServerFactory
from src.modules.utils import train, cosine_similarity


class ServerValidatesClientStrategy(FedAvg):
    def __init__(self, server_train_fn, compare_fn, *args, **kwargs):
        """
        Args:
            server_train_fn: function (client_id: int, round: int) -> Parameters
            compare_fn: function (server_params, client_params) -> float
        """
        super().__init__(*args, **kwargs)
        self.server_train_fn = server_train_fn
        self.compare_fn = compare_fn

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[int, FitRes]],
        failures,
    ) -> Tuple[Parameters, dict]:

        aggregated_params, metrics = super().aggregate_fit(rnd, results, failures)

        print(f"\n📊 Round {rnd} — Validating client updates with server-trained models")
        for client_id, fit_res in results:
            client_params = fit_res.parameters
            # Server-side training for the same client
            server_params = self.server_train_fn(client_id, rnd)

            # Comparison (loss, cosine, etc.)
            score = self.compare_fn(server_params, client_params)
            print(f"Client {client_id}: Server validation score = {score:.4f}")

        return aggregated_params, metrics


class StrategyFactory:
    @staticmethod
    def get_strategy(centralized_testset, config):

        if config.strategy.name == "FedAvg":
            return fl.server.strategy.FedAvg(
                fraction_fit=config.strategy.fraction_train_clients, # Sample % of available clients for training
                fraction_evaluate=config.strategy.fraction_eval_clients,  # Sample % of available clients for evaluation
                min_available_clients=config.strategy.min_available_clients, # Minimum number of clients to sample
                on_fit_config_fn=fit_config, # Configuration for client training
                evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate federated metrics
                evaluate_fn=ServerFactory.get_evaluate_fn(
                    centralized_testset, config),  # Global evaluation function

            )
        elif config.strategy.name == "ServerValidate":
            return ServerValidatesClientStrategy(
                server_train_fn=train,
                compare_fn=cosine_similarity,
                fraction_fit=config.strategy.fraction_train_clients,
                fraction_evaluate=config.strategy.fraction_eval_clients,
                min_available_clients=config.strategy.min_available_clients,
                on_fit_config_fn=fit_config,
                evaluate_metrics_aggregation_fn=weighted_average,
                evaluate_fn=ServerFactory.get_evaluate_fn(centralized_testset, config),
            )

        elif config.strategy.name == "FedAvgM":
            return fl.server.strategy.FedAvgM(
                fraction_fit=config.strategy.fraction_train_clients, # Sample % of available clients for training
                fraction_evaluate=config.strategy.fraction_eval_clients,  # Sample % of available clients for evaluation
                min_available_clients=config.strategy.min_available_clients, # Minimum number of clients to sample
                on_fit_config_fn=fit_config, # Configuration for client training
                evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate federated metrics
                evaluate_fn=ServerFactory.get_evaluate_fn(
                    centralized_testset, config),  # Global evaluation function

            )
        elif config.strategy.name == "FedProx":
            return fl.server.strategy.FedProx(
                proximal_mu=1,
                fraction_fit=config.strategy.fraction_train_clients, # Sample % of available clients for training
                fraction_evaluate=config.strategy.fraction_eval_clients,  # Sample % of available clients for evaluation
                min_available_clients=config.strategy.min_available_clients, # Minimum number of clients to sample
                on_fit_config_fn=fit_config, # Configuration for client training
                evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate federated metrics
                evaluate_fn=ServerFactory.get_evaluate_fn(
                    centralized_testset, config),  # Global evaluation function
            )
        elif config.strategy.name == "FedNova":
            return fl.server.strategy.FedAvg(
                fraction_fit=config.strategy.fraction_train_clients, # Sample % of available clients for training
                fraction_evaluate=config.strategy.fraction_eval_clients,  # Sample % of available clients for evaluation
                min_available_clients=config.strategy.min_available_clients, # Minimum number of clients to sample
                on_fit_config_fn=fit_config, # Configuration for client training
                evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate federated metrics
                evaluate_fn=ServerFactory.get_evaluate_fn(
                    centralized_testset, config),  # Global evaluation function
            )
        elif config.strategy.name == "FedMedian":
            return fl.server.strategy.FedMedian(
                fraction_fit=config.strategy.fraction_train_clients, # Sample % of available clients for training
                fraction_evaluate=config.strategy.fraction_eval_clients,  # Sample % of available clients for evaluation
                min_available_clients=config.strategy.min_available_clients, # Minimum number of clients to sample
                on_fit_config_fn=fit_config, # Configuration for client training
                evaluate_metrics_aggregation_fn=aggregate_median,  # Aggregate federated metrics
                evaluate_fn=ServerFactory.get_evaluate_fn(
                    centralized_testset, config),  # Global evaluation function
            )
        elif config.strategy.name == "Multi-Krum":
            return fl.server.strategy.Krum(
                fraction_fit=config.strategy.fraction_train_clients, # Sample % of available clients for training
                fraction_evaluate=config.strategy.fraction_eval_clients,  # Sample % of available clients for evaluation
                min_available_clients=config.strategy.min_available_clients, # Minimum number of clients to sample
                on_fit_config_fn=fit_config, # Configuration for client training
                evaluate_metrics_aggregation_fn=aggregate,  # Aggregate federated metrics
                evaluate_fn=ServerFactory.get_evaluate_fn(
                    centralized_testset, config),  # Global evaluation function
            )

        else:
            raise ValueError(f"Invalid strategy {config.strategy.name}")
