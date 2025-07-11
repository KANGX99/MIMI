import math
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def compute_cc_weights(nb_steps):
    lam = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    lam = np.cos((lam @ lam.T) * math.pi / nb_steps)
    lam[:, 0] = .5
    lam[:, -1] = .5 * lam[:, -1]
    lam = lam * 2 / nb_steps
    W = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    W[np.arange(1, nb_steps + 1, 2)] = 0
    W = 2 / (1 - W ** 2)
    W[0] = 1
    W[np.arange(1, nb_steps + 1, 2)] = 0
    cc_weights = torch.tensor(lam.T @ W).float()
    steps = torch.tensor(np.cos(np.arange(0, nb_steps + 1, 1).reshape(-1, 1) * math.pi / nb_steps)).float()

    return cc_weights, steps


def integrate(x0, nb_steps, step_sizes, integrand, h, compute_grad=False, x_tot=None):
    #Clenshaw-Curtis Quadrature Method
    cc_weights, steps = compute_cc_weights(nb_steps)

    device = x0.get_device() if x0.is_cuda else "cpu"
    cc_weights, steps = cc_weights.to(device), steps.to(device)
    ## xT is the terminal of the interval.
    xT = x0 + nb_steps*step_sizes
    if not compute_grad:
        x0_t = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        xT_t = xT.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        h_steps = h.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        steps_t = steps.unsqueeze(0).expand(x0_t.shape[0], -1, x0_t.shape[2])
        X_steps = x0_t + (xT_t-x0_t)*(steps_t + 1)/2
        # print(X_steps)
        X_steps = X_steps.contiguous().view(-1, x0_t.shape[2])
        h_steps = h_steps.contiguous().view(-1, h.shape[1]) 
        dzs = integrand(X_steps, h_steps)
        dzs = dzs.view(xT_t.shape[0], nb_steps+1, -1) 
        dzs = dzs*cc_weights.unsqueeze(0).expand(dzs.shape) 
        z_est = dzs.sum(1) 
        return z_est*(xT - x0)/2
    else:
        x0_t = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        xT_t = xT.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        x_tot = x_tot * (xT - x0) / 2
        x_tot_steps = x_tot.unsqueeze(1).expand(-1, nb_steps + 1, -1) * cc_weights.unsqueeze(0).expand(x_tot.shape[0], -1, x_tot.shape[1])
        h_steps = h.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        steps_t = steps.unsqueeze(0).expand(x0_t.shape[0], -1, x0_t.shape[2])
        X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2
        X_steps = X_steps.contiguous().view(-1, x0_t.shape[2])
        h_steps = h_steps.contiguous().view(-1, h.shape[1])
        x_tot_steps = x_tot_steps.contiguous().view(-1, x_tot.shape[1])
        g_param, g_h = computeIntegrand(X_steps, h_steps, integrand, x_tot_steps, nb_steps+1)
        return g_param, g_h


def computeIntegrand(x, h, integrand, x_tot, nb_steps):
    h.requires_grad_(True)
    with torch.enable_grad():
        f = integrand.forward(x, h)
        g_param = _flatten(torch.autograd.grad(f, integrand.parameters(), x_tot, create_graph=True, retain_graph=True))
        g_h = _flatten(torch.autograd.grad(f, h, x_tot))

    return g_param, g_h.view(int(x.shape[0]/nb_steps), nb_steps, -1).sum(1)


class ParallelNeuralIntegral(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x0, x, integrand, flat_params, h, nb_steps=20):
        with torch.no_grad():
            x_tot = integrate(x0, nb_steps, (x - x0)/nb_steps, integrand, h, False)
            # Save for backward
            ctx.integrand = integrand
            ctx.nb_steps = nb_steps
            ctx.save_for_backward(x0.clone(), x.clone(), h)
        return x_tot

    @staticmethod
    def backward(ctx, grad_output):
        x0, x, h = ctx.saved_tensors
        integrand = ctx.integrand
        nb_steps = ctx.nb_steps
        integrand_grad, h_grad = integrate(x0, nb_steps, x/nb_steps, integrand, h, True, grad_output)
        x_grad = integrand(x, h)
        x0_grad = integrand(x0, h)
        # Leibniz formula
        return -x0_grad*grad_output, x_grad*grad_output, None, integrand_grad, h_grad.view(h.shape), None


class IntegrandNN(nn.Module):
    def __init__(self, in_d, hidden_layers):
        super(IntegrandNN, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)
    def forward(self, x, h):  #
        return self.net(torch.cat((x, h), 1)) + 1.


class MonotonicFunc(nn.Module):
    def __init__(self, in_d, hidden_layers, nb_steps=50):
        super(MonotonicFunc, self).__init__()
        self.integrand = IntegrandNN(in_d, hidden_layers)
        self.net = []
        hs = [in_d-1] + hidden_layers + [2]  
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)
        self.nb_steps = nb_steps

    def forward(self, x, h):
        x0 = torch.zeros(x.shape).cuda()
        out = self.net(h)
        offset = out[:, [0]]
        scaling = torch.exp(out[:, [1]])
        return scaling*ParallelNeuralIntegral.apply(x0, x, self.integrand, _flatten(self.integrand.parameters()), h, self.nb_steps) + offset


class MIblock(nn.Module):
    def __init__(self, clamp=1.0):
        super(MIblock, self).__init__()
        self.clamp = clamp
        self.F = MonotonicFunc(3, [100, 100, 100], nb_steps=100)
        self.G = MonotonicFunc(3, [100, 100, 100], nb_steps=100)

    def forward(self, x1, x2, h, rev=False):
        if not rev:
            y1 = x1 + self.F(x2, h)
            y2 = x2 + self.G(y1, h)
        else:
            y2 = (x2 - self.G(x1, h))
            y1 = x1 - self.F(y2, h)

        return y1, y2


class MonotonicInvNet(nn.Module):
    def __init__(self, block_num=3):
        super(MonotonicInvNet, self).__init__()
        operations = []
        for j in range(block_num):
            b = MIblock()
            operations.append(b)
        self.operations = nn.ModuleList(operations)

    def forward(self, x1, x2, h, rev=False):
        out1 = x1
        out2 = x2
        if not rev:
            for op in self.operations:
                out1, out2 = op.forward(out1, out2, h, rev)
        else:
            for op in reversed(self.operations):
                out1, out2 = op.forward(out1, out2, h, rev)

        return out1, out2


def test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    net = MonotonicInvNet(block_num=3)
    net.cuda()
    x1 = torch.arange(-5, 4, .1).unsqueeze(1)
    x1 = Variable(x1.cuda())
    h = torch.zeros(x1.shape[0], 2)
    h = Variable(h.cuda())
    y1, y2 = net.forward(x1, x1, h)
    xx1, xx2 = net.forward(y1, y2, h, rev=True)
    plt.plot(x1.detach().cpu().numpy(), y1[:, 0].detach().cpu().numpy(), label="y1")
    plt.plot(x1.detach().cpu().numpy(), y2[:, 0].detach().cpu().numpy(), label="y2")
    plt.xlabel('x1')
    plt.legend()
    plt.savefig("Monotonicity.png")
    print("x1")
    print(x1)
    print("xx1")
    print(xx1)
    print("xx2")
    print(xx2)
    
if __name__ == '__main__':
    test()
