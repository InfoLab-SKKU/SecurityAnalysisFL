import flwr as fl
from src.modules.server import fit_config, weighted_average, ServerFactory


class StrategyFactory:
    @staticmethod
    def get_strategy(centralized_testset, config):

        if config.strategy.name == "FedAvg":
            return fl.server.strategy.FedAvg(
                fraction_fit=config.strategy.fraction_train_clients,  # Sample 10% of available clients for training
                fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
                min_available_clients=3,
                on_fit_config_fn=fit_config,
                evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate federated metrics
                evaluate_fn=ServerFactory.get_evaluate_fn(centralized_testset, config),  # Global evaluation function

            )
        else:
            raise ValueError(f"Invalid strategy {config.strategy.name}")