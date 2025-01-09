from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import numpy as np
from src.modules.attacks.utils import get_ar_params



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

    def on_batch_selection(self, inputs: torch.Tensor, targets: torch.Tensor):
        batch_size = inputs.size(0)
        adv_inputs = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(batch_size):
            delta, _ = self.generate(p=0.75, index=targets[i])
            adv_input = (inputs[i] + delta.to(device)).clamp(0, 1)
            adv_inputs.append(adv_input)
        return torch.stack(adv_inputs), targets

    def on_before_backprop(self, model, loss):
        # No changes. Pass-through.
        return model, loss

    def on_after_backprop(self, model, gradients):
        # No changes. Pass-through.
        return gradients



class AttackFactory:

    @staticmethod
    def create_attack(config):
        name = config.poisoning.name
        if name == "ar":
            return AutoRegressorAttack(config)
        else:
            raise NotImplementedError("The Attack you are trying to use is not implemented yet.")
