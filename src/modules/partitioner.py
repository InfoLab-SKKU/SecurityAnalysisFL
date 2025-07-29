from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, PathologicalPartitioner


class PartitionFactory:
    @staticmethod
    def get_fds(config):
        partitions_number = config.client.count
        dataset_config = config.dataset
        partitioner_name = dataset_config.partitioner.name
        if dataset_config.name == "albertvillanova/medmnist-v2":
            if partitioner_name == "DirichletPartitioner":  # Non IiD
                print("drichlet_________________")
                return FederatedDataset(dataset=dataset_config.name,
                                        subset=dataset_config.subset,
                                        partitioners={"train": DirichletPartitioner(
                                            num_partitions=partitions_number,
                                            partition_by="label",
                                            seed=dataset_config.seed,
                                            alpha=dataset_config.partitioner.alpha,
                                            min_partition_size=0,
                                        )})
            elif partitioner_name == "PathologicalPartitioner":  # Non IiD
                print("pathological_________________")
                return FederatedDataset(dataset=dataset_config.name,
                                        subset=dataset_config.subset,
                                        partitioners={"train": PathologicalPartitioner(
                                            num_partitions=partitions_number,
                                            partition_by="label",
                                            seed=dataset_config.seed,
                                            num_classes_per_partition=dataset_config.partitioner.num_classes_per_partition,
                                            min_partition_size=0,
                                        )})
            elif partitioner_name == "IiD":  # IiD
                print("iid______________")
                return FederatedDataset(dataset=dataset_config.name,
                                        subset=dataset_config.subset,
                                        partitioners={"train": partitions_number})
            else:
                raise ValueError("Invalid partitioner name")
        elif dataset_config.name in ["cifar10", "cifar100", "mnist", "femnist"]:
            if partitioner_name == "DirichletPartitioner":
                return FederatedDataset(dataset=dataset_config.name,
                                        partitioners={"train": DirichletPartitioner(
                                            num_partitions=partitions_number,
                                            partition_by="label",
                                            seed=dataset_config.seed,
                                            alpha=dataset_config.partitioner.alpha,
                                            min_partition_size=0,
                                        )})
            elif partitioner_name == "PathologicalPartitioner":
                return FederatedDataset(dataset=dataset_config.name,
                                        partitioners={"train": PathologicalPartitioner(
                                            num_partitions=partitions_number,
                                            partition_by="label",
                                            seed=dataset_config.seed,
                                            num_classes_per_partition=dataset_config.partitioner.num_classes_per_partition,
                                            min_partition_size=0,
                                        )})
            elif partitioner_name == "IiD":
                return FederatedDataset(dataset=dataset_config.name,
                                        partitioners={"train": partitions_number})
            else:
                raise ValueError("Invalid partitioner name")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_config.name}")