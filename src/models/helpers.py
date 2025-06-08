import torch

def count_parameters(model):
    """Cuenta parámetros entrenables de un modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_encoder(model):
    """Congela todos los parámetros del encoder para entrenamiento solo de la cabeza."""
    for param in model.encoder.parameters():
        param.requires_grad = False