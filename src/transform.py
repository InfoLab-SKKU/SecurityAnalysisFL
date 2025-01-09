from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import hydra
from pathlib import Path
from torch.utils.data import DataLoader
from src.modules.attacks.attack import AttackFactory
from src.modules.partitioner import PartitionFactory
from src.modules.utils import apply_transforms
import torch
import torch.nn.functional as F
import numpy as np

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def show_images(original, attacked, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original.permute(1, 2, 0))  # Convert CHW to HWC
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(attacked.permute(1, 2, 0))  # Convert CHW to HWC
    axes[1].set_title("Attacked")
    axes[1].axis("off")

    plt.suptitle(title)
    plt.show()

@hydra.main(config_path='../config', config_name='main', version_base=None)
def main(cfg: DictConfig):
    # Print the current configuration for reference
    print(OmegaConf.to_yaml(cfg))

    fds = PartitionFactory.get_fds(cfg)
    trainset = fds.load_partition(
        0, "train"
    )
    trainset = trainset.with_transform(apply_transforms)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attack = AttackFactory.create_attack(cfg)
    original_images = []
    generated_images = []
    labels_list = []

    for epoch in range(1):
        for idx, batch in enumerate(trainloader):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            old_img = images[0].cpu()
            images, labels = attack.on_batch_selection(images, labels)
            new_img = images[0].cpu()
            show_images(old_img, new_img, title=f"Epoch {epoch + 1}")
            euclidean_distance = torch.dist(old_img, new_img, p=2)
            original_images.append(old_img.numpy())
            generated_images.append(new_img.numpy())
            labels_list.append(labels[0].item())


    original_images_np = np.stack(original_images)
    generated_images_np = np.stack(generated_images)
    labels_np = np.array(labels_list)
    save_path = "/tmp/bloodmnist_train_generated_images.npz"
    np.savez(save_path, original=original_images_np, generated=generated_images_np, labels=labels_np)

if __name__ == "__main__":
    # Entry point for the script
    main()
