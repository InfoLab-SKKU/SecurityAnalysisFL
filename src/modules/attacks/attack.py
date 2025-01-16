import random
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import numpy as np
from src.modules.attacks.utils import get_ar_params


class Attack(ABC):
    """
    A generic interface for attack
    """

    def on_batch_selection(self, inputs: torch.Tensor, targets: torch.Tensor):
        
        return inputs, targets

    def on_before_backprop(self, model, loss):
        return model, loss

    def on_after_backprop(self, model, loss):
        return model, loss

class Benin(ABC):
    def on_batch_selection(self, inputs: torch.Tensor, targets: torch.Tensor):
        return inputs, targets

    def on_before_backprop(self, model, loss):
        return model, loss

    def on_after_backprop(self, model, loss):
        return model, loss



class Noops(ABC):
    def on_batch_selection(self, inputs: torch.Tensor, targets: torch.Tensor):
        return inputs, targets

    def on_before_backprop(self, model, loss):
        return model, loss

    def on_after_backprop(self, model, loss):
        return model, loss


class AutoRegressorAttack(Attack):

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
                val = F.conv2d(
                    start_signal[:, i: i + kernel_size, j: j + kernel_size],
                    ar_coeff,
                    groups=self.num_channels,
                )
                noise = torch.randn(1) if self.gaussian_noise else 0
                start_signal[:, i + kernel_size - 1, j + kernel_size - 1] = (
                        val.squeeze() + noise
                )

        start_signal_crop = start_signal[:, self.crop:, self.crop:]
        generated_norm = torch.norm(start_signal_crop, p=p, dim=(0, 1, 2))
        scale = (1 / generated_norm) * self.epsilon
        start_signal_crop = scale * start_signal_crop
        return start_signal_crop, generated_norm

    def on_batch_selection(self, inputs: torch.Tensor, targets: torch.Tensor):
        batch_size = inputs.size(0)
        adv_inputs = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(batch_size):
            delta, _ = self.generate(p=2, index=targets[i])
            print(f"iteration {i} -- delta-size={delta.size()}  -- inputs-size={inputs[i].size()}----------------------------------")
            adv_input = (inputs[i] + delta.to(device)).clamp(0, 1)
            adv_inputs.append(adv_input)
        return torch.stack(adv_inputs), targets

    def on_before_backprop(self, model, loss):
        # No changes. Pass-through.
        return model, loss

    def on_after_backprop(self, model, gradients):
        # No changes. Pass-through.
        return gradients


class LabelFlipAttack(Attack):

    def __init__(self, config):
        super().__init__()

        self.num_classes = int(config.model.num_classes)
        self.epsilon = float(config.poisoning.epsilon)

        if self.epsilon is None:
            self.epsilon = 0.1

    def on_batch_selection(self, inputs: torch.Tensor, targets: torch.Tensor):
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

        def on_batch_selection(self, inputs: torch.Tensor, targets: torch.Tensor):
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
            else:
                return NotImplementedError("The Attack type you are trying to use is not implemented yet.")
        else:
            raise NotImplementedError("The Attack you are trying to use is not implemented yet.")
