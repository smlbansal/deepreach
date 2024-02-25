import torch
from torch.autograd import grad

# TODO: I don't think jacobian is needed here; torch.autograd.grad should be enough, to compute gradients of a scalar value function w.r.t. inputs

# batched jacobian
# y: [..., N], x: [..., M] -> [..., N, M]
def jacobian(y, x):
    ''' jacobian of y wrt x '''
    jac = torch.zeros(*y.shape, x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[..., i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status




