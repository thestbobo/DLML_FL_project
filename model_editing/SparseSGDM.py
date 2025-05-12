import torch
from torch.optim import SGD


class SparseSGDM(SGD):
    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0,
                 nesterov=False, mask=None, model=None):
        super(SparseSGDM, self).__init__(params, lr=lr, momentum=momentum,
                                         dampening=dampening, weight_decay=weight_decay,
                                         nesterov=nesterov)
        self.mask = mask
        if mask is not None and model is not None:
            param_to_mask = {}
            for name, param in model.named_parameters():
                if name in mask:
                    param_to_mask[param] = mask[name]
            self.mask = param_to_mask
        else:
            self.mask = None

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if self.mask is not None and p in self.mask:            # only update masked parameters
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
                with torch.no_grad():
                    p.add_(d_p, alpha=-group['lr'])

        return loss
