import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR


def build_optimizer(model, config):
    name = config.TRAINER.OPTIMIZER
    lr = config.TRAINER.TRUE_LR

    # Only optimise parameters that still require grad. Important when modules
    # have been frozen (e.g. v3's FREEZE_BACKBONE / FREEZE_BN): AdamW applies
    # weight_decay directly to param.data even when grad is None, which would
    # otherwise let frozen weights drift. For v0/v1/v2 (no freezing) every
    # param has requires_grad=True so this filter is a no-op and training is
    # byte-identical to the previous behaviour.
    trainable = filter(lambda p: p.requires_grad, model.parameters())

    if name == "adam":
        return torch.optim.Adam(trainable, lr=lr, weight_decay=config.TRAINER.ADAM_DECAY)
    elif name == "adamw":
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=config.TRAINER.ADAMW_DECAY)
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


def build_scheduler(config, optimizer):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
            'monitor': 'val_f1', (optional)
            'frequency': x, (optional)
        }
    """
    scheduler = {'interval': config.TRAINER.SCHEDULER_INTERVAL}
    name = config.TRAINER.SCHEDULER

    if name == 'MultiStepLR':
        scheduler.update(
            {'scheduler': MultiStepLR(optimizer, config.TRAINER.MSLR_MILESTONES, gamma=config.TRAINER.MSLR_GAMMA)})
    elif name == 'CosineAnnealing':
        scheduler.update(
            {'scheduler': CosineAnnealingLR(optimizer, config.TRAINER.COSA_TMAX)})
    elif name == 'ExponentialLR':
        scheduler.update(
            {'scheduler': ExponentialLR(optimizer, config.TRAINER.ELR_GAMMA)})
    else:
        raise NotImplementedError()

    return scheduler
