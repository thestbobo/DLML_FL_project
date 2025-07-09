import torch
from torch.optim import SGD


class SparseSGDM(SGD):
    """
    Sparse SGD with Momentum (and optional Nesterov), applying a user-provided mask
    to zero out gradients (and thus updates) for pruned weights.
    """

    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        mask=None,
        model=None
    ):
        super(SparseSGDM, self).__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov
        )

        if mask is not None:
            if model is not None:
                param_to_mask = {}
                for name, param in model.named_parameters():
                    if name in mask:
                        param_to_mask[param] = mask[name]
                self.mask = param_to_mask
            else:
                self.mask = mask
        else:
            self.mask = None

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group.get('dampening', 0)
            nesterov = group.get('nesterov', False)
            weight_decay = group['weight_decay']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                if self.mask is not None and p in self.mask:
                    d_p = d_p * self.mask[p]

                if momentum != 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = d_p.clone().detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    if nesterov:
                        d_p = d_p.mul(1 - dampening).add(buf, alpha=momentum)
                    else:
                        d_p = buf

                with torch.no_grad():
                    p.data.add_(d_p, alpha=-lr)

        return loss
