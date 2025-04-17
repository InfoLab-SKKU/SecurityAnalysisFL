#!/bin/bash
 cd ..
# Array to store failed experiments
failed_experiments=()

# Function to run an experiment and track failures
run_experiment() {
  local strategy=$1
  local model=$2
  local dataset=$3
  local partitioner=$4
  local partition_param=$5
  local partition_value=$6
  local num_classes=$7
  local epsilon=$8
  local attack=$9
  local fraction_mal_cli=${10}

  # Generate timestamp for directory
  timestamp=$(date +%Y-%m-%d_%H-%M-%S)

  # Define output directory path with timestamp
  output_dir="outputs/${strategy}/${model}/${dataset}/${partitioner}/${partition_value}/ldp_${epsilon}/${attack}_${fraction_mal_cli}/${timestamp}"

  echo "Starting experiment with dataset=${dataset}, partitioner=${partitioner}, ${partition_param}=${partition_value}, model.num_classes=${num_classes}"

  # Run the experiment
  python -m src.main hydra.run.dir=$output_dir dataset.subset="$dataset" strategy.name="$strategy" model.name="$model" dataset.partitioner.name="$partitioner" dataset.partitioner.$partition_param="$partition_value" model.num_classes="$num_classes" ldp.epsilon="$epsilon" poisoning.fraction="$fraction_mal_cli"

  # Check if the experiment failed
  if [ $? -ne 0 ]; then
    echo "Experiment failed: dataset=${dataset}, partitioner=${partitioner}, ${partition_param}=${partition_value}, model.num_classes=${num_classes}"
    # Record the failed experiment
    failed_experiments+=("${dataset}/${partitioner}/${partition_value}")
  fi
}

run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "IiD" "alpha" 0.9 9 0 "targeted" 0.1
run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "IiD" "alpha" 0.9 8 0 "targeted" 0.2
run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "IiD" "alpha" 0.9 8 0 "targeted" 0.3
run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "IiD" "alpha" 0.9 8 0 "targeted" 0.4
run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "IiD" "alpha" 0.9 8 0 "targeted" 0.5

run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "DirichletPartitioner" "alpha" 0.9 8 0 "targeted" 0.1
run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "DirichletPartitioner" "alpha" 0.9 8 0 "targeted" 0.2
run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "DirichletPartitioner" "alpha" 0.9 8 0 "targeted" 0.3
run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "DirichletPartitioner" "alpha" 0.9 8 0 "targeted" 0.4
run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "DirichletPartitioner" "alpha" 0.9 8 0 "targeted" 0.5

run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "DirichletPartitioner" "alpha" 0.3 8 0 "targeted" 0.1
run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "DirichletPartitioner" "alpha" 0.3 8 0 "targeted" 0.2
run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "DirichletPartitioner" "alpha" 0.3 8 0 "targeted" 0.3
run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "DirichletPartitioner" "alpha" 0.3 8 0 "targeted" 0.4
run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "DirichletPartitioner" "alpha" 0.3 8 0 "targeted" 0.5

run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "DirichletPartitioner" "alpha" 0.1 8 0 "targeted" 0.1
run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "DirichletPartitioner" "alpha" 0.1 8 0 "targeted" 0.2
run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "DirichletPartitioner" "alpha" 0.1 8 0 "targeted" 0.3
run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "DirichletPartitioner" "alpha" 0.1 8 0 "targeted" 0.4
run_experiment "FedAvg" "mobilenet_v2" "bloodmnist" "DirichletPartitioner" "alpha" 0.1 8 0 "targeted" 0.5

# Final report
echo "Experiments completed!"

# Check if there were any failures
if [ ${#failed_experiments[@]} -ne 0 ]; then
  echo "The following experiments failed:"
  for experiment in "${failed_experiments[@]}"; do
    echo "  - $experiment"
  done
  exit 1  # Exit with error code if any experiments failed
else
  echo "All experiments succeeded!"
  exit 0  # Exit with success code if everything went fine
fi