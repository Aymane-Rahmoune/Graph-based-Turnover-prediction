import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss, L1Loss
from .loss import MAPELoss

def initialize_optimizer(model, optimizer_config):
    if optimizer_config is None:
        return None
    if optimizer_config['type'] == 'Adam':
        return Adam(model.parameters(), lr=optimizer_config['lr'])
    return None

def initialize_criterion(criterion_config):
    if criterion_config is None:
        return None
    if criterion_config == 'MAPELoss':
        return MAPELoss()
    if criterion_config == 'MSELoss':
        return MSELoss()
    if criterion_config == 'L1Loss':
        return L1Loss()
    return None

def initialize_scheduler(optimizer, scheduler_config):
    if scheduler_config is None or optimizer is None:
        return None
    if scheduler_config['type'] == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_config['factor'], patience=scheduler_config['patience'])
    return None
