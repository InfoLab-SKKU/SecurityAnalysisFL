# --- Imports ---
import streamlit as st
import subprocess
import os
import time
import threading
import pandas as pd
import matplotlib.pyplot as plt
import io
from pynvml import *
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions

# --- GPU Tracker ---
class GPUTracker:
    def __init__(self, interval=1):
        self.interval = interval
        self.data = []
        self.running = False

    def _track(self):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        while self.running:
            timestamp = time.time()
            mem = nvmlDeviceGetMemoryInfo(handle)
            util = nvmlDeviceGetUtilizationRates(handle)
            temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
            self.data.append({
                "time": timestamp,
                "memory_used_MB": mem.used // (1024 ** 2),
                "utilization_%": util.gpu,
                "temperature_C": temp
            })
            time.sleep(self.interval)
        nvmlShutdown()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._track)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def get_dataframe(self):
        return pd.DataFrame(self.data)

# --- Page Config ---
st.set_page_config(page_title="Federated Experiment Config", page_icon="📈", layout="wide")
st.title("Federated Learning Experiment Dashboard")


# --- Sidebar: Dataset Configuration ---
st.sidebar.title("Configuration Panel")
with st.sidebar.expander("Dataset Configuration"):
    dataset = st.selectbox("Dataset", ["mnist", "cifar10", "albertvillanova/medmnist-v2", "OtherDataset"])
    subset = st.selectbox("Subset", ["bloodmnist", "tissuemnist", "pathmnist", "dermamnnist", "octmnist", "breastmnist", "organamnist", "organcmnist", "organsmnist"]) if dataset == "albertvillanova/medmnist-v2" else None
    if dataset == "mnist":
        num_classes = 10
    elif dataset == "cifar10":
        num_classes = 10
    elif subset == "bloodmnist":
        num_classes = 8
    elif subset == "tissuemnist":
        num_classes = 8
    elif subset == "pathmnist":
        num_classes = 9
    elif subset == "dermamnnist":
        num_classes = 7
    elif subset == "octmnist":
        num_classes = 4
    elif subset == "breastmnist":
        num_classes = 2
    elif subset == "organamnist":
        num_classes = 11
    elif subset == "organcmnist":
        num_classes = 11
    elif subset == "organsmnist":
        num_classes = 11
    else:
        num_classes = st.number_input("Number of Classes", min_value=1, max_value=100, value=10, step=1)



    partitioner = st.selectbox("Partitioner", ["DirichletPartitioner", "PathologicalPartitioner", "IiD"])
    if partitioner == "DirichletPartitioner":
        partition_param = "alpha"
        partition_value = st.slider("Alpha Value", 0.01, 5.0, 0.3, step=0.01)
    elif partitioner == "PathologicalPartitioner":
        st.sidebar.info("Requires number of classes per partition.")
        partition_param = "num_classes_per_partition"
        partition_value = st.selectbox("Classes per Partition", ["2", "4", "7"])
    else:
        st.sidebar.info("No partition parameter required for IID.")
        partition_param = None
        partition_value = None

# --- Sidebar: Attack Configuration ---
with st.sidebar.expander("Attack Configuration"):
    attack = st.selectbox("Attack Type", ["adaptive-targeted", "untargeted", "none"])
    epsilon = st.selectbox("LDP Epsilon", ["0", "0.1", "1", "10"])
    fraction_mal_cli = st.selectbox("Fraction of Malicious Clients", ["0.2", "0.5", "0.8"])

# --- Sidebar: FL Strategy Configuration ---
with st.sidebar.expander("Federated Strategy Configuration", expanded=True):
    strategy = st.selectbox("Strategy", ["FedAvg", "FedAvgM", "FedNova", "FedMedian", "FedProx", "Multi-Krum", "OtherStrategy"])
    model = st.selectbox("Model", ["mobilenet_v2", "resnet18", "OtherModel"])
    num_clients = st.number_input("Number of Clients", min_value=3, max_value=10000, value=50, step=1)
    fraction_train_clients = st.slider("Fraction of Training Clients", 0.01, 1.0, 0.01, step=0.01)
    num_rounds = st.number_input("Number of Rounds", min_value=3, max_value=10000, value=5)

# --- Main: Summary ---
with st.expander("📝 Summary of Experiment Configuration"):
    st.write("**Dataset**:", dataset)
    if subset:
        st.write("**Subset**:", subset)
    st.write("**Partitioner**:", partitioner)
    if partition_param:
        st.write(f"**{partition_param.capitalize()}**:", partition_value)
    st.write("**Attack Type**:", attack)
    st.write("**LDP Epsilon**:", epsilon)
    st.write("**Malicious Fraction**:", fraction_mal_cli)
    st.write("**Strategy**:", strategy)
    st.write("**Fraction of Training Clients**:", fraction_train_clients)
    st.write("**Model**:", model)
    st.write("**Classes**:", num_classes)
    st.write("**Clients**:", num_clients)
    st.write("**Rounds**:", num_rounds)

visualize= st.checkbox("Show Dataset Partition Visualization", value=False)
if visualize:
    num_partitions = int(num_clients)  # You may align with num_clients or customize
    if dataset == "albertvillanova/medmnist-v2":
        if partitioner == "DirichletPartitioner":  # You can make this dynamic later
            fds = FederatedDataset(
                dataset=dataset,
                subset=subset,
                partitioners={
                    "train": DirichletPartitioner(
                        num_partitions=num_partitions,
                        partition_by="label",
                        alpha=float(partition_value),
                        seed=42,
                        min_partition_size=0,
                    ),
                },
            )
        elif partitioner == "PathologicalPartitioner":
            from flwr_datasets.partitioner import PathologicalPartitioner

            fds = FederatedDataset(
                dataset=dataset,
                subset=subset,
                partitioners={
                    "train": PathologicalPartitioner(
                        num_partitions=num_partitions,
                        partition_by="label",
                        seed=42,
                        num_classes_per_partition=int(partition_value),
                    ),
                },
            )
        elif partitioner == "IiD":
            fds = FederatedDataset(
                dataset=dataset,
                subset=subset,
                partitioners={"train": num_partitions}  # IID partitioning
            )
        else:
            st.info("Label distribution visualization is not supported for this partitioner. Skipping..")
    else:
        if partitioner == "DirichletPartitioner":  # You can make this dynamic later
            fds = FederatedDataset(
                dataset=dataset,
                partitioners={
                    "train": DirichletPartitioner(
                        num_partitions=num_partitions,
                        partition_by="label",
                        alpha=float(partition_value),
                        seed=42,
                        min_partition_size=0,
                    ),
                },
            )
        elif partitioner == "PathologicalPartitioner":
            from flwr_datasets.partitioner import PathologicalPartitioner
            fds = FederatedDataset(
                dataset=dataset,
                partitioners={
                    "train": PathologicalPartitioner(
                        num_partitions=num_partitions,
                        partition_by="label",
                        seed=42,
                        num_classes_per_partition=int(partition_value),
                    ),
                },
            )
        elif partitioner == "IiD":
            fds = FederatedDataset(
                dataset=dataset,
                partitioners={"train": num_partitions}  # IID partitioning
            )
        else:
            st.info("Label distribution visualization is not supported for this partitioner. Skipping..")

    partitionerp = fds.partitioners["train"]
    fig, ax, df = plot_label_distributions(
        partitionerp,
        label_name="label",
        plot_type="bar",
        size_unit="absolute",
        partition_id_axis="x",
        legend=True,
        verbose_labels=True,
        title="Per Partition Labels Distribution",
    )
    fig.set_size_inches(20, 5)
    st.markdown("### 📊 Dataset Partition Visualization")
    st.pyplot(fig, clear_figure=True, bbox_inches='tight')
    st.expander("Label Distribution DataFrame").dataframe(df)
    # st.dataframe(df)



# --- Run Experiment ---
st.markdown("### 🚀 Launch")
show_logs = st.checkbox("Show Live Logs", value=True)
submitted = st.button("Run Experiment")

if submitted:
    tracker = GPUTracker(interval=1)
    tracker.start()

    try:
        output_dir = f"outputs/{strategy}/{model}/{dataset}/{partitioner}/{partition_value or 'iid'}"
        cmd = (
            f"python -m src.main "
            f"hydra.run.dir={output_dir} "
            f"dataset.name={dataset} "
            f"{f'dataset.subset={subset} ' if subset else ''}"
            f"strategy.name={strategy} "
            f"model.name={model} "
            f"dataset.partitioner.name={partitioner} "
            f"strategy.num_rounds={num_rounds} "
            f"client.count={num_clients} "
            f"strategy.fraction_train_clients={fraction_train_clients} "
        )
        if partition_param:
            cmd += f"dataset.partitioner.{partition_param}={partition_value} "
        cmd += (
            f"model.num_classes={num_classes} "
            f"ldp.epsilon={epsilon} "
            f"poisoning.fraction={fraction_mal_cli} "
        )

        st.markdown("#### 🧾 Executing Command:")
        st.code(cmd)

        log_placeholder = st.empty()
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        if show_logs:
            logs = ""
            for line in process.stdout:
                if "[flwr][INFO]" in line:
                    logs += line
                    log_placeholder.code(logs, language="bash", height=200)

        process.wait()

        if process.returncode == 0:
            st.success("✅ Experiment completed successfully!")
        else:
            st.error("❌ Experiment failed. Check logs for details.")
    finally:
        tracker.stop()
        df = tracker.get_dataframe()
        csv_path = os.path.join(output_dir, "gpu_stats.csv")
        df.to_csv(csv_path, index=False)

    # --- Results Display ---
    csv_path = os.path.join(output_dir, "results.csv")
    st.markdown("### 📊 Results")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.dataframe(df)

        if {'Round', 'accuracy', 'asr'}.issubset(df.columns):
            fig, ax = plt.subplots()
            ax.plot(df['Round'], df['accuracy'], label='Accuracy')
            ax.plot(df['Round'], df['asr'], label='ASR')
            ax.set_xlabel('Round')
            ax.set_ylabel('Metric Value')
            ax.set_ylim(0, 1)
            ax.set_title('Accuracy and ASR over Rounds')
            ax.legend()
            fig.set_size_inches(15, 3)
            st.pyplot(fig)
            # Download buttons
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            st.download_button("📥 Download Graph (PNG)", buf, "graph.png", mime="image/png")

            buf.seek(0)
            st.download_button("📥 Download Graph (PDF)", buf, "graph.pdf", mime="application/pdf")
        else:
            st.warning("CSV must include 'Round', 'accuracy', and 'asr' columns.")
    else:
        st.warning("No results.csv found. Experiment may not have logged output.")
