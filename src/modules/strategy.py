import flwr as fl
from flwr.server.strategy.aggregate import aggregate, aggregate_median

from src.modules.server import fit_config, weighted_average, ServerFactory


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
