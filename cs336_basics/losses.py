import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

    m = torch.max(logits, dim=-1, keepdim=True).values

    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    shift_logits = logits - m

    log_exp_sum = m.squeeze(-1) + torch.log(torch.sum(torch.exp(shift_logits), dim=-1))

    loss = log_exp_sum - target_logits

    return torch.mean(loss)