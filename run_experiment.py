import streamlit as st
import subprocess
import os
import time
import pandas as pd
import io
import matplotlib.pyplot as plt

import streamlit as st
from click import progressbar
import streamlit as st
from pynvml import *


# gpu_tracker.py
import threading
import time
import pandas as pd
from pynvml import *

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

st.set_page_config(page_title="Federated Experiment Config", page_icon="🧪", layout="wide")

st.title("Federated Learning Experiment Dashboard")


# --- SIDEBAR ---
st.sidebar.title("Configuration Panel")


# --- Dataset Configuration ---
with st.sidebar.expander("Dataset Configuration"):
    dataset = st.selectbox("Dataset", ["mnist", "cifar10", "albertvillanova/medmnist-v2", "OtherDataset"])
    subset = None
    if dataset == "albertvillanova/medmnist-v2":
        subset = st.selectbox("Subset", ["tissuemnist", "pathmnist", "bloodmnist"])

    partitioner = st.selectbox("Partitioner", ["DirichletPartitioner", "PathologicalPartitioner", "IiD"])

    if partitioner == "DirichletPartitioner":
        partition_param = "alpha"
        partition_value = st.selectbox("Alpha Value", ["0.9", "0.5", "0.1"])

    elif partitioner == "PathologicalPartitioner":
        st.sidebar.info("Requires number of classes per partition.")
        partition_param = "num_classes_per_partition"
        partition_value = st.selectbox("Classes per Partition", ["2", "4", "7"])
    else:
        st.sidebar.info("No partition parameter required for IID.")
        partition_param = None
        partition_value = None

# --- Attack Configuration ---
with st.sidebar.expander("Attack Configuration"):
    attack = st.selectbox("Attack Type", ["adaptive-targeted", "untargeted", "none"])
    epsilon = st.selectbox("LDP Epsilon", ["0", "0.1", "1", "10"])
    fraction_mal_cli = st.selectbox("Fraction of Malicious Clients", ["0.2", "0.5", "0.8"])


# --- FL Strategy Configuration ---
with st.sidebar.expander("Federated Strategy Configuration", expanded=True):
    strategy = st.selectbox("Strategy", ["FedAvg", "FedProx", "OtherStrategy"])
    model = st.selectbox("Model", ["mobilenet_v2", "resnet18", "OtherModel"])
    num_classes = st.selectbox("Number of Classes", ["8", "10", "100"], index=1)
    num_clients = st.number_input("Number of Clients", min_value=3, max_value=10000, value=250)
    fraction_train_clients = st.slider("Fraction of Training Clients", 0.01, 1.0, 0.01, step=0.01)
    num_rounds = st.number_input("Number of Rounds", min_value=3, max_value=10000, value=5)

# --- MAIN PAGE ---
with st.expander("### Summary of Experiment Configuration"):
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

# --- Run Button ---
st.markdown("### 🚀 Launch")
show_logs = st.checkbox("Show Live Logs", value=True)
submitted = st.button("Run Experiment")



if submitted:
    # Start GPU tracking
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
            # f"poisoning.attack={attack}"
        )

        st.markdown("#### 🧾 Executing Command:")
        st.code(cmd)

        # Live logs
        log_placeholder = st.empty()
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if show_logs:
            logs = ""
            for line in process.stdout:
                # Call the function
                if "[flwr][INFO]" in line:
                    logs += line
                    log_placeholder.code(logs, language="bash")
        process.wait()


        if process.returncode == 0:
            st.success("✅ Experiment completed successfully!")
        else:
            st.error("❌ Experiment failed. Check logs for details.")
    finally:
        # Stop tracking and save data
        tracker.stop()
        df = tracker.get_dataframe()
        df.to_csv("outputs/gpu_stats.csv", index=False)

    # --- Results Section ---
    csv_path = os.path.join(output_dir, "results.csv")
    st.markdown("### 📊 Results")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.dataframe(df)

        if 'Round' in df.columns and 'accuracy' in df.columns and 'asr' in df.columns:
            fig, ax = plt.subplots()
            ax.plot(df['Round'], df['accuracy'], label='Accuracy')
            ax.plot(df['Round'], df['asr'], label='ASR')
            ax.set_xlabel('Round')
            ax.set_ylabel('Metric Value')
            ax.set_ylim(0, 1)
            ax.set_title('Accuracy and ASR over Rounds')
            ax.legend()
            st.pyplot(fig)

            # Download buttons
            png_buf = io.BytesIO()
            fig.savefig(png_buf, format='png')
            png_buf.seek(0)
            st.download_button("📥 Download Graph (PNG)", png_buf, "graph.png", mime="image/png")

            pdf_buf = io.BytesIO()
            fig.savefig(pdf_buf, format='pdf')
            pdf_buf.seek(0)
            st.download_button("📥 Download Graph (PDF)", pdf_buf, "graph.pdf", mime="application/pdf")
        else:
            st.warning("CSV must include 'Round', 'accuracy', and 'asr' columns.")
    else:
        st.warning("No results.csv found. Experiment may not have logged output.")
