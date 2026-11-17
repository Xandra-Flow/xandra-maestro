"""
╔═══════════════════════════════════════════════════════════════╗
║   Xandra Hope - Layer 2: Meta-Memory B & CDD (v1.0)           ║
║   Copyright (c) 2026 Ahmed Sayed Kamel Mahmoud &               ║
║                  Александра Евгеньевна Острикова              ║
║   All Rights Reserved. License: XHNSL v1.0                   ║
╚═══════════════════════════════════════════════════════════════╝

Implements Layer 2 (Non-Generative Meta-Memory) of the Triadic Cognitive
Architecture (TCA), matching Section 5 (Eqs. 6-11) of the research paper:
  - Hierarchical Sliding Anchors (R_core with lambda=0.95, R_sub with lambda=0.70)
  - Pre-LN Mean-Centering & L2 Normalization (Section 5.4 Anisotropy Guard)
  - Context Dispersion Detector (CDD) computing dispersion D(k) in FP32
  - Orthogonal Redirection Vector R_inj injected into Pre-LN Cross-Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreLNMeanCentering(nn.Module):
    """
    Section 5.4: Bridges centered and uncentered spaces via Pre-LN
    mean-centering and unit-sphere normalization to eliminate embedding
    anisotropy before cosine dispersion and orthogonal projection.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            x_fp32 = x.float()
            normed = self.ln(x_fp32)
            centered = normed - normed.mean(dim=-1, keepdim=True)
            return F.normalize(centered, p=2, dim=-1)


class HierarchicalSlidingAnchors(nn.Module):
    """
    Section 5.1: Dual-timescale semantic anchors:
      - R_core (lambda_core = 0.95): preserves overarching conversation goal
      - R_sub  (lambda_sub  = 0.70): adapts smoothly to legitimate sub-topic shifts
      - R_t = alpha_r * R_core + (1 - alpha_r) * R_sub
    """
    def __init__(self, dim: int, lambda_core: float = 0.95, lambda_sub: float = 0.70, alpha_r: float = 0.60):
        super().__init__()
        self.dim = dim
        self.lambda_core = lambda_core
        self.lambda_sub = lambda_sub
        self.alpha_r = alpha_r
        self.center_norm = PreLNMeanCentering(dim)

    def compute_anchors(self, token_embeddings: torch.Tensor, window_size: int = 64) -> dict:
        """
        token_embeddings: (B, T, D)
        Splits sequence into sliding windows of size W and tracks R_core, R_sub, and R_t.
        """
        with torch.amp.autocast('cuda', enabled=False):
            h = self.center_norm(token_embeddings.float())  # (B, T, D)
            B, T, D = h.shape

            # Initial anchor from the opening prompt window
            init_len = min(window_size, T)
            r_core = F.normalize(h[:, :init_len, :].mean(dim=1), p=2, dim=-1)  # (B, D)
            r_sub = r_core.clone()

            r_t_history = []
            window_means = []

            for start in range(0, T, window_size):
                end = min(start + window_size, T)
                w_k = F.normalize(h[:, start:end, :].mean(dim=1), p=2, dim=-1)  # (B, D)
                window_means.append(w_k)

                if start > 0:
                    r_core = F.normalize(self.lambda_core * r_core + (1.0 - self.lambda_core) * w_k, p=2, dim=-1)
                    r_sub = F.normalize(self.lambda_sub * r_sub + (1.0 - self.lambda_sub) * w_k, p=2, dim=-1)

                r_t = F.normalize(self.alpha_r * r_core + (1.0 - self.alpha_r) * r_sub, p=2, dim=-1)
                r_t_history.append(r_t)

            return {
                "r_core": r_core,
                "r_sub": r_sub,
                "r_t_seq": torch.stack(r_t_history, dim=1),      # (B, num_windows, D)
                "window_means": torch.stack(window_means, dim=1) # (B, num_windows, D)
            }


class ContextDispersionDetector(nn.Module):
    """
    Section 5.2 (Eqs. 6-8): Computes semantic dispersion D(k) between each sliding
    window W_k and the composite hierarchical anchor R_t.
    """
    def __init__(self, theta_crit: float = 0.40):
        super().__init__()
        # Learnable threshold initialized at theta_crit (calibrated in Stage 3)
        self.theta_crit = nn.Parameter(torch.tensor(float(theta_crit)))

    def forward(self, window_means: torch.Tensor, r_t_seq: torch.Tensor) -> dict:
        """
        window_means: (B, K, D) unit-normalized window embeddings
        r_t_seq:      (B, K, D) unit-normalized composite anchors
        Returns dispersion D(k) in [0, 2] and trigger mask where D(k) > theta_crit.
        """
        with torch.amp.autocast('cuda', enabled=False):
            cos_sim = torch.sum(window_means.float() * r_t_seq.float(), dim=-1)  # (B, K)
            dispersion = 1.0 - cos_sim  # D(k) = 1 - cos(W_k, R_t)
            theta = torch.clamp(self.theta_crit, min=0.15, max=0.85)
            trigger_mask = (dispersion > theta).float()
            excess_dispersion = F.relu(dispersion - theta)

            return {
                "dispersion": dispersion,               # (B, K)
                "trigger_mask": trigger_mask,           # (B, K)
                "excess_dispersion": excess_dispersion, # (B, K)
                "theta_crit": float(theta.item())
            }
class OrthogonalRedirectionInjector(nn.Module):
    """
    Section 5.3 & 5.4 (Eqs. 9-11):
      - Computes orthogonal redirection vector R_inj = Proj_{perp T_k}(R_t)
      - Injects R_inj gently into Pre-LN Cross-Attention keys/queries when D(k) > theta_crit
        to steer generation back toward the overarching topic without wiping context.
    """
    def __init__(self, dim: int, n_heads: int = 4, beta_inj: float = 0.25):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.beta_inj = nn.Parameter(torch.tensor(float(beta_inj)))

        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def compute_orthogonal_vector(self, r_t: torch.Tensor, t_k: torch.Tensor) -> torch.Tensor:
        """
        r_t: (B, K, D) target composite anchor
        t_k: (B, K, D) current drifted window vector
        Returns R_inj = r_t - <r_t, t_k> * t_k (normalized).
        """
        with torch.amp.autocast('cuda', enabled=False):
            r = r_t.float()
            t = F.normalize(t_k.float(), p=2, dim=-1)
            proj = torch.sum(r * t, dim=-1, keepdim=True) * t
            r_perp = r - proj
            return F.normalize(r_perp, p=2, dim=-1)

    def forward(self, hidden_states: torch.Tensor, r_t_seq: torch.Tensor,
                window_means: torch.Tensor, trigger_mask: torch.Tensor,
                window_size: int = 64) -> torch.Tensor:
        """
        hidden_states: (B, T, D)
        r_t_seq:       (B, K, D)
        window_means:  (B, K, D)
        trigger_mask:  (B, K) binary/float mask where D(k) > theta_crit
        """
        B, T, D = hidden_states.shape
        K = r_t_seq.size(1)

        r_inj = self.compute_orthogonal_vector(r_t_seq, window_means)  # (B, K, D)
        gated_inj = r_inj * trigger_mask.unsqueeze(-1).to(r_inj.dtype)  # (B, K, D)

        # Expand window-level steering vectors to token level
        token_inj = gated_inj.repeat_interleave(window_size, dim=1)[:, :T, :].to(hidden_states.dtype)

        # Pre-LN Cross-Attention bridging centered & uncentered spaces (Section 5.4)
        q = self.wq(self.norm_q(hidden_states)).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(self.norm_k(token_inj)).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(token_inj).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)

        scale = torch.clamp(self.beta_inj, min=0.05, max=0.60).to(hidden_states.dtype)
        # Apply injection only on triggered windows
        token_gate = trigger_mask.repeat_interleave(window_size, dim=1)[:, :T].unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states + (scale * token_gate * self.wo(attn_out))


class XandraMetaMemoryB(nn.Module):
    """
    Complete Layer 2 Module: Non-Generative Meta-Memory B & Context Dispersion Control.
    Wraps Layer 1 (XandraHEPL) frozen backbone during Stage 3 calibration.
    """
    def __init__(self, dim: int = 512, n_heads: int = 4, window_size: int = 64,
                 lambda_core: float = 0.95, lambda_sub: float = 0.70, theta_crit: float = 0.40):
        super().__init__()
        self.window_size = window_size
        self.anchors = HierarchicalSlidingAnchors(
            dim=dim, lambda_core=lambda_core, lambda_sub=lambda_sub
        )
        self.cdd = ContextDispersionDetector(theta_crit=theta_crit)
        self.injector = OrthogonalRedirectionInjector(dim=dim, n_heads=n_heads)

    def forward(self, hidden_states: torch.Tensor) -> dict:
        anc = self.anchors.compute_anchors(hidden_states, window_size=self.window_size)
        disp_info = self.cdd(anc["window_means"], anc["r_t_seq"])
        steered_states = self.injector(
            hidden_states=hidden_states,
            r_t_seq=anc["r_t_seq"],
            window_means=anc["window_means"],
            trigger_mask=disp_info["trigger_mask"],
            window_size=self.window_size
        )

        # Calibration loss encourages dispersion to stay below theta_crit while keeping theta_crit tight
        cal_loss = disp_info["excess_dispersion"].mean() + 0.05 * torch.abs(self.cdd.theta_crit - 0.40)

        return {
            "hidden_states": steered_states,
            "mean_dispersion": float(disp_info["dispersion"].mean().item()),
            "trigger_rate": float(disp_info["trigger_mask"].mean().item()),
            "theta_crit": disp_info["theta_crit"],
            "calibration_loss": cal_loss
        }
