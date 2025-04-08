# SecurityAnalysisFL


This project demonstrates a federated learning (FL) setup using PyTorch for training a MobileNetV2 model, with the Flower framework to manage the federated learning process. The project also incorporates differential privacy for enhanced security, using local differential privacy (DP) mechanisms.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Federated Learning Simulation](#running-the-federated-learning-simulation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In this project, we leverage federated learning to train a neural network model in a distributed manner across multiple clients. Each client trains the model locally using its own data and shares only the updated model parameters with the central server. No raw data is shared, ensuring privacy.

### Key Components:
- **Flower**: A federated learning framework used to simulate the FL process.
- **PyTorch**: For building and training the MobileNetV2 model

## Features

- **Federated Learning Simulation**: Manage multiple clients and a central server to orchestrate model training.
- **MobileNetV2**: Utilize a pre-trained MobileNetV2 model, which is fine-tuned on custom datasets.
- **Data Preprocessing**: Includes image transformations such as resizing, normalization, and grayscale conversion.
- **Scalable Simulation**: Number of clients and resource allocation are parameterizable.
  
## Installation

### Prerequisites
- Conda (Anaconda or Miniconda)
- Python 3.10+

### Setting Up the Environment

To set up the environment using Conda, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
    ```

2. **Create the Conda environment**
    ```bash
   conda env create -f environment.yml
    ```
   
4. **Activate the environment**:

   ```bash
   conda activate federated-learning
   ```

## usage
### Running multi Experiments
4. **Run multi Experiments**
    ```bash
   cd scripts/ && bash multirun.sh
    ```
## Citation

We would appreciate it if you consider citing our work when using our code.

```bibtex
@inproceedings{soubih2025towards,
  title={Towards Robust Federated Learning: Investigating Poisoning Attacks Under Clients Data Heterogeneity},
  author={Soubih, Abdenour and Lahmer, Seyyid Ahmed and Abuhamad, Mohammed and Abuhmed, Tamer},
  booktitle={2025 19th International Conference on Ubiquitous Information Management and Communication (IMCOM)},
  pages={1--8},
  year={2025},
  organization={IEEE}
}
```
