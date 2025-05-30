import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyTopKSelector(nn.Module):
    def __init__(self, 
            feature_dim, k, 
            soft=True, 
            use_gumbel=True,
            noise_std=1.0, 
            noise_std_decay=0.9999,
            min_noise_std=0.1,
            tau=1.0,
            tau_decay=0.9998,
            min_tau=0.1,
    ):
        super().__init__()
        self.k = k
        
        self.soft = soft
        self.use_gumbel = use_gumbel
        
        self.tau_decay = tau_decay
        self.noise_std_decay = noise_std_decay 
        
        self.register_buffer('tau', torch.tensor([tau], dtype=torch.float32))
        self.min_tau = torch.tensor([min_tau], dtype=torch.float32)
        self.register_buffer('noise_std', torch.tensor([noise_std], dtype=torch.float32))
        self.min_noise_std = torch.tensor([min_noise_std], dtype=torch.float32)

        self.scorer = nn.Linear(feature_dim, 1)
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    def sample_gumbel(self, shape, eps=1e-20):
        uni = torch.rand(shape, device=self.device)
        return - torch.log(- torch.log(uni + eps) + eps)

    def update_noise_std(self):
        self.min_noise_std = self.min_noise_std.to(self.device)
        self.noise_std = max(self.noise_std * self.noise_std_decay, self.min_noise_std)

    def update_tau(self):
        self.min_tau = self.min_tau.to(self.device)
        self.tau = max(self.tau * self.tau_decay, self.min_tau)

    def forward(self, x, index_only=False):
        batch_size, seq_len, feature_dim = x.shape
        scores = self.scorer(x).squeeze(-1)  # (b, l)

        if self.noise_std > 0:
            noise = torch.randn_like(scores) * self.noise_std
            noisy_scores = scores + noise
        else:
            noisy_scores = scores

        if self.soft or self.use_gumbel:
            probs = (noisy_scores / self.tau).softmax(dim=-1)
            if self.use_gumbel:
                gumbel_noise = self.sample_gumbel(noisy_scores.shape)
                probs = torch.log(probs + 1e-10) + gumbel_noise

            topk_probs, topk_indices = probs.topk(self.k, dim=-1)
        else:
            probs = None
            topk_scores, topk_indices = noisy_scores.topk(self.k, dim=-1)

        topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, feature_dim)
        
        self.update_tau()
        self.update_noise_std()
        
        if not index_only:
            selected_x = torch.gather(x, dim=1, index=topk_indices)  
            return selected_x, topk_indices, probs
        else:
            return topk_indices, probs