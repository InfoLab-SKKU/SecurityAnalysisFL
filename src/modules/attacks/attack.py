import random
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

from src.modules.attacks.utils import get_ar_params


class Attack(ABC):
    """
    A generic interface for attack
    """

    def on_batch_selection(self, net: nn.Module, device: str, inputs: torch.Tensor, targets: torch.Tensor):
        
        return inputs, targets

    def on_before_backprop(self, model, loss):
        return model, loss

    def on_after_backprop(self, model, loss):
        return model, loss

class Benin(ABC):
    def on_batch_selection(self, net: nn.Module, device: str, inputs: torch.Tensor, targets: torch.Tensor):
        return inputs, targets

    def on_before_backprop(self, model, loss):
        return model, loss

    def on_after_backprop(self, model, loss):
        return model, loss



class Noops(ABC):
    def on_batch_selection(self, net: nn.Module, device: str, inputs: torch.Tensor, targets: torch.Tensor):
        return inputs, targets

    def on_before_backprop(self, model, loss):
        return model, loss

    def on_after_backprop(self, model, loss):
        return model, loss


class AutoRegressorAttack:

    def __init__(self, config):
        super().__init__()

        # TODO check the number of channels
        self.num_channels = 3

        self.num_classes = int(config.model.num_classes)
        self.epsilon = float(config.poisoning.epsilon)
        self.size = tuple(config.poisoning.size)

        self.crop = int(config.poisoning.crop)
        self.gaussian_noise = bool(config.poisoning.gaussian_noise)

        if self.size is None:
            self.size = (36,36)

        if self.crop is None:
            self.crop = 3

        if self.gaussian_noise is None:
            self.gaussian_noise = False

        if self.epsilon is None:
            self.epsilon = 8/255

        self.ar_params = get_ar_params(num_classes=self.num_classes)
        self.ar_params = [torch.clamp(param, -1, 1) for param in self.ar_params]
        self.ar_params *= 255
        print(self.crop, self.size, self.gaussian_noise,self.epsilon,self.num_channels,self.num_classes)

    def generate(self, index, p=np.inf):
        start_signal = torch.randn((self.num_channels, self.size[0], self.size[1]))
        kernel_size = 3
        rows_to_update = self.size[0] - kernel_size + 1
        cols_to_update = self.size[1] - kernel_size + 1
        ar_param = self.ar_params[index]
        ar_coeff = ar_param.unsqueeze(dim=1)

        for i in range(rows_to_update):
            for j in range(cols_to_update):
                val = torch.nn.functional.conv2d(
                    start_signal[:, i: i + kernel_size, j: j + kernel_size],
                    ar_coeff,
                    groups=self.num_channels,
                )#.clamp(-1,1)
                noise = torch.randn(1) if self.gaussian_noise else 0
                start_signal[:, i + kernel_size - 1, j + kernel_size - 1] = (
                        val.squeeze() + noise
                )
        start_signal_crop = start_signal[:, self.crop:, self.crop:]
        generated_norm = torch.norm(start_signal_crop, p=p, dim=(0, 1, 2))
        scale = (1 / generated_norm) * self.epsilon
        start_signal_crop = scale * start_signal_crop
        return start_signal_crop, generated_norm

    def on_dataset_load(self, trainset, valset):
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
        new_train = trainset.with_transform(apply_transforms_0)

        for i, item in enumerate(new_train):
            old_image = item["image"]
            label = item["label"]
            delta, _ = self.generate(p=2, index=label)
            new_image = old_image + delta
            #print(f"===================================={i}=============================================")
            #print(delta)
            #show_images(old_image, delta, title=f"imagine {i + 1}")
            #show_images(, new_image, title=f"imagine {i + 1}")
            new_train[i]["image"] = new_image
        #batch["image"] = [transform(img) for img in batch["image"]]

        #print(new_train)
        return new_train, valset
    def on_batch_selection(self, inputs: torch.Tensor, targets: torch.Tensor):
        return inputs, targets

    def on_before_backprop(self, model, loss):
        return model, loss

    def on_after_backprop(self, model, gradients):
        return gradients


class LabelFlipAttack(Attack):

    def __init__(self, config):
        super().__init__()

        self.num_classes = int(config.model.num_classes)
        self.epsilon = float(config.poisoning.epsilon)

        if self.epsilon is None:
            self.epsilon = 0.1

    def on_batch_selection(self, net: nn.Module, device: str, inputs: torch.Tensor, targets: torch.Tensor):
        batch_size = inputs.size(0)
        possible_labels = list(range(self.num_classes))
        adv_targets = targets.clone()
        for idx, adv_target in enumerate(adv_targets):
            current_label = adv_target.item()
            adv_targets[idx] = random.choice([label for label in possible_labels if label != current_label])  # Update `adv_targets` tensor in-place
            #print(f"current_label={current_label} -- adv_targets after update={adv_targets[idx].item()}")

        #print(f"original_targets={targets} -- modified_adv_targets={adv_targets}")
        return inputs, adv_targets

    def on_before_backprop(self, model, loss):
        # No changes. Pass-through.
        return model, loss

    def on_after_backprop(self, model, gradients):
        # No changes. Pass-through.
        return gradients

class TargetedLabelFlipAttack(Attack):

        def __init__(self, config):
            super().__init__()

            self.num_classes = int(config.model.num_classes)
            self.poison_label = float(config.poisoning.poison_label)
            self.target_label = float(config.poisoning.target_label)

        def on_batch_selection(self, net: nn.Module, device: str, inputs: torch.Tensor, targets: torch.Tensor):
            """
            Updates the target labels in the batch to the specified target class.

            Args:
                inputs (torch.Tensor): The batch of input data.
                targets (torch.Tensor): The batch of original target labels.
                target_class (int): The specific target class to which all labels should be updated.

            Returns:
                inputs (torch.Tensor): The unchanged input data.
                adv_targets (torch.Tensor): The new adversarial target labels all set to target_class.
            """
            # Ensure the target_class is a valid class index
            if not (0 <= self.target_label < self.num_classes):
                raise ValueError(f"target_class must be between 0 and {self.num_classes - 1}, got {self.target_label}")

            # Create a new tensor where all labels are set to the target_class
            adv_targets = targets.clone()
            for idx, adv_target in enumerate(adv_targets):
                current_label = adv_target.item()
                if current_label == self.poison_label:
                    adv_targets[idx] = self.target_label  # Update `adv_targets` tensor in-place
                #print(f"current_label={current_label} -- adv_targets after update={adv_targets[idx].item()}")

            #print(f"original_targets={targets} -- modified_adv_targets={adv_targets}")
            return inputs, adv_targets

        def on_before_backprop(self, model, loss):
            # No changes. Pass-through.
            return model, loss

        def on_after_backprop(self, model, gradients):
            # No changes. Pass-through.
            return gradients


class AdaptiveTargetedLabelFlipAttack(Attack):

    def __init__(self, config):
        super().__init__()
        self.model = None
        self.device = None
        self.num_classes = int(config.model.num_classes)
        self.poison_label = float(config.poisoning.poison_label)
        self.target_label = float(config.poisoning.target_label)

    def on_batch_selection(self, net: nn.Module, device: str, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Updates the target labels in the batch to the second-best predicted class.

        Args:
            inputs (torch.Tensor): The batch of input data.
            targets (torch.Tensor): The batch of original target labels.

        Returns:
            inputs (torch.Tensor): The unchanged input data.
            adv_targets (torch.Tensor): The new adversarial target labels set to the second-best predicted class.
        """
        self.model = net
        self.device = device
        # Ensure the model and device are available
        if not hasattr(self, "model") or not hasattr(self, "device"):
            raise ValueError("The attack must have access to 'model' and 'device' attributes.")

        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Get predictions from the model
            inputs = inputs.to(self.device)
            logits = self.model(inputs)  # Shape: [batch_size, num_classes]
            probabilities = torch.softmax(logits, dim=1)  # Convert logits to probabilities
            #print(f"probabilities={probabilities}===============================")
            # Find the top-2 predicted classes for each input
            top2_preds = torch.topk(probabilities, k=2, dim=1)
            top_classes = top2_preds.indices  # Shape: [batch_size, 2]
            # Clone the original targets
            adv_targets = targets.clone()

            # Flip only the specified class
            for idx, current_label in enumerate(targets):
                if current_label.item() == self.poison_label:  # Check if it's the poisoned class
                    adv_targets[idx] = top_classes[idx, 1]  # Set to the second-best predicted class

        # print(f"original_targets={targets.cpu()} -- modified_adv_targets={adv_targets.cpu()}")
        self.model.train()
        return inputs, adv_targets

    def on_before_backprop(self, model, loss):
        # No changes. Pass-through.
        return model, loss

    def on_after_backprop(self, model, gradients):
        # No changes. Pass-through.
        return gradients

class AttackFactory:

    @staticmethod
    def create_attack(config) -> Attack:
        name = config.poisoning.name
        type = config.poisoning.attack_type
        if name == "ar":
            return AutoRegressorAttack(config)
        elif name == "label-flipping":
            if type == "targeted":
                return TargetedLabelFlipAttack(config)
            elif type == "untargeted":
                return LabelFlipAttack(config)
            elif type == "adaptive-targeted":
                return AdaptiveTargetedLabelFlipAttack(config)
            else:
                return NotImplementedError("The Attack type you are trying to use is not implemented yet.")
        else:
            raise NotImplementedError("The Attack you are trying to use is not implemented yet.")
