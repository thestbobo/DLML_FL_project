import torch
from torch.optim import SGD


class SparseSGDM(SGD):
    def __init__(self, params, lr, momentum=0, weight_decay=0, mask=None):
        super(SparseSGDM, self).__init__(params, lr, momentum=momentum, weight_decay=weight_decay)
        self.mask = mask

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if self.mask is not None and p in self.mask:
                    d_p = d_p * self.mask[p]
                if group['weight_decay'] != 0:
                    d_p = d_p.add(p, alpha=group['weight_decay'])
                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(d_p, alpha=1 - group['dampening'])
                    d_p = buf
                p.add_(d_p, alpha=-group['lr'])

        return loss
