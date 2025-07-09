import torch
import torch.nn as nn

class TaLoS:
    """
    Talos pruner implementation: Taylor-based One-shot Layer-wise Optimal Sparsification.
    Keeps parameters with smallest sensitivity (|w * grad|) and prunes the rest.
    Adds debug prints to show layer-wise pruning stats.
    """
    def __init__(self, named_parameters):
        # named_parameters: iterable of (name, torch.nn.Parameter)
        self.params = []
        self.names = {}
        for name, p in named_parameters:
            if p.requires_grad:
                self.params.append(p)
                self.names[p] = name
        self.scores = {p: torch.zeros_like(p.data) for p in self.params}
        self.masks = {p: torch.ones_like(p.data) for p in self.params}
        self._stats = (0, 0)

    def score(self, model, loss_fn, loader, device, n_batches=-1):
        """
        Compute first-order sensitivity scores for each parameter.
        Accumulate |w * grad| over loader.
        """
        for p in self.params:
            self.scores[p].zero_()

        model.eval()
        count = 0
        for i, batch in enumerate(loader):
            if 0 <= n_batches <= i:
                break
            inputs = batch['images'].to(device)
            targets = batch['labels'].to(device)
            model.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets) if loss_fn is not None else outputs.mean()
            loss.backward()
            for p in self.params:
                if p.grad is None:
                    continue
                self.scores[p] += torch.abs(p.data * p.grad.data)
            count += 1
        for p in self.params:
            self.scores[p] /= max(count, 1)

    def mask(self, sparsity, mode='global'):
        """
        Apply mask to parameters based on sparsity.
        Prints per-layer prune stats.
        """
        if mode == 'global':
            all_scores = torch.cat([torch.flatten(self.scores[p]) for p in self.params])
            k = int((1 - sparsity) * all_scores.numel())
            if k < 1:
                thr = all_scores.max() + 1
            else:
                thr, _ = torch.kthvalue(all_scores, k)
            # apply and debug
            print(f"[TaLoS] Global threshold: {thr:.4e}")
            for p in self.params:
                name = self.names[p]
                before = p.numel()
                mask = (self.scores[p] >= thr).float()
                self.masks[p] = mask
                p.data *= mask
                kept = int(mask.sum().item())
                pruned = before - kept
                print(f"[TaLoS] Layer {name}: kept {kept}/{before} ({100*kept/before:.2f}%), pruned {pruned}/{before} ({100*pruned/before:.2f}%)")

        elif mode == 'local':
            for p in self.params:
                name = self.names[p]
                flat = self.scores[p].view(-1)
                k = int((1 - sparsity) * flat.numel())
                if k < 1:
                    thr = flat.max() + 1
                else:
                    thr, _ = torch.kthvalue(flat, k)
                mask = (self.scores[p] >= thr).float()
                self.masks[p] = mask
                before = p.numel()
                p.data *= mask
                kept = int(mask.sum().item())
                pruned = before - kept
                print(f"[TaLoS] Layer {name} (local): thr={thr:.4e}, kept {kept}/{before} ({100*kept/before:.2f}%), pruned {pruned}/{before} ({100*pruned/before:.2f}%)")

        else:
            raise ValueError(f"Unknown mode {mode}")

    def stats(self):
        """
        Return (remaining_params, total_params).
        """
        remaining = sum(int(self.masks[p].sum().item()) for p in self.params)
        total = sum(p.numel() for p in self.params)
        self._stats = (remaining, total)
        return self._stats

    def apply_mask(self):
        """Re-apply existing masks to params"""
        for p in self.params:
            p.data *= self.masks[p]
