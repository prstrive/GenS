import numpy as np
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts


def WarmupCosineLR(optimizer, total_steps, warmup=0.2, alpha=0.1):
    lambda_func = lambda step: 0.1 + 0.9 * step / warmup if step < warmup else (np.cos(np.pi * (step - warmup) / (total_steps - warmup)) + 1.0) * 0.5 * (1 - alpha) + alpha

    return LambdaLR(optimizer, lambda_func)


def VolumeWarmupCosineLR(optimizer, total_steps, warmup=0.2, alpha=0.05, alpha_vol=0.01):
    lambda_func = lambda step: 0.1 + 0.9 * step / warmup if step < warmup else (np.cos(np.pi * (step - warmup) / (total_steps - warmup)) + 1.0) * 0.5 * (1 - alpha) + alpha
    volume_lambda_func = lambda step: 0.1 + 0.9 * step / warmup if step < warmup else (np.cos(np.pi * (step - warmup) / (total_steps - warmup)) + 1.0) * 0.5 * (1 - alpha_vol) + alpha_vol
    lambda_funcs = [lambda_func] + [volume_lambda_func] * (len(optimizer.param_groups) - 1)
    return LambdaLR(optimizer, lambda_funcs)