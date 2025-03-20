import torch


class LogitProcessor:
    def __init__(self, temperature : float = 0.0, top_k : int = 0, top_p : float = 0):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
    
    def top_k_top_p_filter(self, logits):
        if self.top_k > 0:
            filter = torch.topk(logits, min(self.top_k, logits.size(-1)))[0]
            logits[logits < filter[:, [-1]]] = float('-inf')
        if self.top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1)
            filter = cumulative_probs > self.top_p
            filter[..., 1:] = filter[..., :-1].clone()
            filter[..., 0] = 0
            indices_to_remove = filter.scatter(1, sorted_indices, filter)
            logits[indices_to_remove] = float('-inf')
        return logits
    
    def norm(self, logits):
        assert logits.dim() == 2
        if self.temperature == 0:
            idx = logits.argmax(dim=1)
            new_logits = torch.zeros_like(logits, device=logits.device)
            new_logits[:, idx] = 1
            return new_logits.float()
        logits = logits / self.temperature
        logits = self.top_k_top_p_filter(logits, top_k=self.top_k, top_p=self.top_p)
        probs = F.softmax(logits, dim=1)
        return probs

    def sample(self, logits):
        idx_next = torch.multinomial(self.norm(logits), num_samples=1)
        return idx_next
