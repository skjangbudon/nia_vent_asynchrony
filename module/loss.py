import torch

def calculate_vae_loss(out, mu, logVar, xi, batch_size=None):
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
    loss_r = bce_loss(out, xi)
    # loss_r = ((out - xi)**2).sum()
    loss_kl = torch.mean(.5 * torch.sum(mu.pow(2) + torch.exp(logVar) - 1 - logVar, 1))
    # loss = torch.mean(loss_r) + loss_kl
    loss = loss_r + loss_kl
    return loss, loss_r, loss_kl

def calculate_bce_loss(out, target, pos_weight=torch.tensor([5])):
    if pos_weight is not None:
        pos_weight = pos_weight.to(target)
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
    loss = bce_loss(out, target)
    return loss

def calculate_ce_loss(out, target):
    ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = ce_loss(out, target)
    return loss
