import torch

class MAPELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true):
        non_zero_mask = y_true != 0
        return torch.mean(torch.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]))