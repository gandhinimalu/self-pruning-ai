import torch
from model import PrunableLinear

def sparsity_loss(model):
    loss = 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            loss += torch.sum(torch.sigmoid(m.gate_scores))
    return loss

def compute_sparsity(model, threshold=1e-2):
    total, pruned = 0, 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores)
            total += gates.numel()
            pruned += (gates < threshold).sum().item()
    return 100 * pruned / total