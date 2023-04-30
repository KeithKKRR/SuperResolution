from torch import nn


def loss_fn(loss_name):
    assert (loss_name in ["MSE"])
    if loss_name == 'MSE':
        return nn.MSELoss()
