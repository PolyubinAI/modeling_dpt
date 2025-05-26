import torch


def abs_rel(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:

    mask = target > 0
    num_vals = mask.sum()
    abs_diff = torch.abs(target[mask] - prediction[mask])
    return torch.sum(abs_diff / target[mask]) / num_vals    
 

class AbsRel(torch.nn.Module):
 
    def forward(self, prediction, target) -> torch.Tensor:
        """
        Calculates the Absolute relative score between
        the prediction and target tensors.

        Args:
            prediction (Tensor): The prediction tensor.
            target (Tensor): The target tensor.

        Returns:
            Tensor: The absolute relative score.
            abs_rel = 1 / n * sum(abs(dpred_i - dtrue_i) / (dtrue_i)) 
        """

        return torch.mean(abs_rel(target, prediction))
