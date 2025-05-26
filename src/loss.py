import torch
import numpy as np
from constants import LOW_CLIP_VALUE

LAMBDA = 0.05
C = 100


def sie(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Scale-Invariant Logarithmic Error (SI Log) score
    between the target and prediction tensors.

    Args:
        target (Tensor): The target tensor.
        prediction (Tensor): The prediction tensor.

    Returns:
        Tensor: The SI Log score.
    """
    mask = target > 0
    num_vals = target.shape[1]
    log_diff = torch.log(torch.where(mask, prediction, LOW_CLIP_VALUE) / torch.where(mask, target, LOW_CLIP_VALUE))
    norm_squared_sum = torch.sum(log_diff, dim=1) ** 2 / (num_vals**2)
    si_log_unscaled = torch.sum(log_diff**2, dim=1) / num_vals - LAMBDA * norm_squared_sum

    return torch.sqrt(si_log_unscaled) * C


class SILoss(torch.nn.Module):
 
    def forward(self, prediction, target) -> torch.Tensor:
        """
        Calculates the Scale-Invariant Logarithmic Loss (SI Loss) between
        the prediction and target tensors.

        Args:
            prediction (Tensor): The prediction tensor.
            target (Tensor): The target tensor.

        Returns:
            Tensor: The SI Loss.
        """
        prediction = torch.reshape(prediction, (prediction.shape[0], -1))
        target = torch.reshape(target, (prediction.shape[0], -1))

        mask = target > 0

        log_diff = torch.zeros_like(prediction)
        log_diff[mask] = torch.log(prediction[mask] / target[mask])

        # return torch.mean(si_log(target, prediction))
        return torch.min(sie(target, prediction))

class MSELoss(torch.nn.Module):
 
    def forward(self, prediction, target) -> torch.Tensor:

        mask = target != 0

        criterion = torch.nn.MSELoss()
        return criterion(prediction[mask], target[mask])

class MAELoss(torch.nn.Module):
 
    def forward(self, prediction, target) -> torch.Tensor:

        mask = target != 0

        criterion = torch.nn.L1Loss()
        
        return criterion(prediction[mask], target[mask])
    
class RMSELoss(torch.nn.Module):
 
    def forward(self, prediction, target) -> torch.Tensor:

        mask = target != 0

        criterion = torch.nn.MSELoss()
        
        return torch.sqrt(criterion(prediction[mask], target[mask]))
