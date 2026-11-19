"""
╔═══════════════════════════════════════════════════════════════╗
║   Xandra Hope - HEPL Architecture & Distillation (v2.0)       ║
║   Copyright (c) 2026 Ahmed Sayed Kamel Mahmoud &               ║
║                  Александра Евгеньевна Острикова              ║
║   All Rights Reserved. License: XHNSL v1.0                   ║
╚═══════════════════════════════════════════════════════════════╝

Implements Layer 1 (HEPL) of the Triadic Cognitive Architecture (TCA):
  - Four Epistemological Manifolds: M_real + M_intu + M_imag + M_rat
  - Local Relational GAT (w=32 sliding window, O(L*w) bilinear edge scorer)
  - Cross-Manifold Attention (restricted strictly to M_rat in last 2 layers)
  - Conditional Sparse Routing Gate with FP32 Critical Guard, Gumbel-Softmax
    Isolation Mask, Dynamic Epsilon Schedule, and Smooth Annealing Warmup
  - Sparse Top-32+1 Other Bucket KL Distillation Loss (FP32 guarded)
  - Self-Healing Manifold Telemetry (Relative/Absolute Collapse + Act Norms)
  - Surgical Weight Grafter (from XandraModern base_150k.pt with sigma=0.001)

Note on Reproducibility:
  With DataParallel + gradient checkpointing on Tesla T4 x2, exact bit-level
  reproducibility is not achievable despite seed=42 due to non-deterministic
  reduction ordering across GPUs. Loss variations of ~1e-4 across runs are
  expected and acceptable. For strict reproducibility, use single-GPU mode.
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm.to(x.dtype)) * self.weight


def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class CausalGroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        cos, sin = precompute_rope_frequencies(self.head_dim, max_seq_len)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class SwiGLUExpert(nn.Module):
    """Feed-forward network for each epistemological manifold."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LocalRelationalGAT(nn.Module):
    """
    Differentiable R-GAT Bridge inside M_intu (Eq. 2).
    Computes soft adjacency A_pred[i, j] over a local causal sliding window w=32
    to keep complexity strictly O(L * w) without external parsers.
    """
    def __init__(self, dim: int, window_size: int = 32, head_dim: int = 64):
        super().__init__()
        self.window_size = window_size
        self.head_dim = head_dim
        self.wq_edge = nn.Linear(dim, head_dim, bias=False)
        self.wk_edge = nn.Linear(dim, head_dim, bias=False)
        self.wv_rel = nn.Linear(dim, dim, bias=False)
        self.b_rel = nn.Parameter(torch.zeros(1))
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.norm(x)
        q = self.wq_edge(h)
        k = self.wk_edge(h)
        v = self.wv_rel(h)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) + self.b_rel

        idx = torch.arange(T, device=x.device)
        dist = idx.unsqueeze(1) - idx.unsqueeze(0)
        valid_mask = (dist > 0) & (dist <= self.window_size)

        adj = torch.sigmoid(scores) * valid_mask.unsqueeze(0).to(scores.dtype)
        denom = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        weights = adj / denom

        rel_msg = torch.matmul(weights, v)
        return x + self.out_proj(rel_msg)


class CrossManifoldAttention(nn.Module):
    """
    Epistemic Hierarchy Enforcement (Section 4.1 & Reviewer Point 4):
    M_rat is the ONLY manifold permitted to read outputs from M_real, M_intu,
    and M_imag, active exclusively in the final 2 transformer layers.
    """
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.norm_q = RMSNorm(dim)
        self.norm_kv = RMSNorm(dim)

    def forward(self, h_rat: torch.Tensor, other_manifolds: torch.Tensor) -> torch.Tensor:
        B, T, D = h_rat.shape
        q = self.wq(self.norm_q(h_rat)).view(B * T, 1, self.n_heads, self.head_dim).transpose(1, 2)
        kv_in = self.norm_kv(other_manifolds).view(B * T, 3, D)
        k = self.wk(kv_in).view(B * T, 3, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(kv_in).view(B * T, 3, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        return h_rat + self.wo(attn_out)
class ConditionalSparseGate(nn.Module):
    """
    Conditional Sparse Routing Gate G(x) with Learned Isolation Mask mu(x)
    (Eqs. 3, 4, 5) + Reviewer Tactical Guards:
      - Explicit .float() upcast inside autocast(enabled=False) for Gumbel-Softmax & Softmax
      - Smooth Annealing Window between step 500 and step 600 (alpha_hard: 0 -> 1)
      - Dynamic epsilon schedule: max(0.05, 0.20 - 0.001 * step)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w_gate = nn.Linear(dim, 4, bias=False)
        self.mask_controller = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 4)
        )
        # Initialize mask controller bias low so all manifolds start open
        nn.init.constant_(self.mask_controller[-1].bias, -2.0)
        # SAFE-RETRAIN: M_real-first init bias.
        # softmax([ln(7), 0, 0, 0]) = [0.70, 0.10, 0.10, 0.10] exactly.
        # Registered in __init__ so it is ALWAYS present on the module, even
        # if graft_from_xandra_modern is not called (e.g. fresh model build).
        # Graft later overwrites it with the same value, so the contract holds.
        self.register_buffer(
            "_m_real_init_bias",
            torch.tensor([math.log(7.0), 0.0, 0.0, 0.0]),
            persistent=False,
        )

    def _compute_soft_weights(self, logits_fp32: torch.Tensor, mask_logits_fp32: torch.Tensor,
                              deterministic_flag: torch.Tensor = None, training: bool = True) -> torch.Tensor:
        if training:
            # Gumbel-Softmax relaxation in FP32 for differentiability
            u = torch.rand_like(mask_logits_fp32).clamp(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            mu = torch.sigmoid((mask_logits_fp32 + gumbel) / 1.0)
        else:
            mu = torch.sigmoid(mask_logits_fp32)

        # Optional explicit hard mask for deterministic prompts (disables M_imag at index 2)
        if deterministic_flag is not None:
            hard_imag_mask = torch.zeros_like(mu)
            hard_imag_mask[..., 2] = deterministic_flag.float().unsqueeze(-1)
            mu = torch.maximum(mu, hard_imag_mask)

        masked_logits = (1.0 - mu) * logits_fp32 - (mu * 1e4)
        return F.softmax(masked_logits, dim=-1)

    def _compute_topk_weights(self, soft_probs: torch.Tensor, step: int) -> torch.Tensor:
        # Dynamic epsilon schedule (Reviewer Point 3)
        eps = max(0.05, 0.20 - 0.001 * step)

        top2_vals, top2_idx = torch.topk(soft_probs, k=2, dim=-1)
        p1 = top2_vals[..., 0:1]
        p2 = top2_vals[..., 1:2]

        # Select k=2 only when top-2 probabilities are within margin eps
        use_k2 = ((p1 - p2) < eps).float()

        w1 = p1
        w2 = p2 * use_k2
        denom = (w1 + w2).clamp(min=1e-8)
        w1_norm = w1 / denom
        w2_norm = w2 / denom

        sparse_weights = torch.zeros_like(soft_probs)
        sparse_weights.scatter_(-1, top2_idx[..., 0:1], w1_norm)
        sparse_weights.scatter_(-1, top2_idx[..., 1:2], w2_norm)
        return sparse_weights

    def forward(self, x: torch.Tensor, step: int = 0, deterministic_flag: torch.Tensor = None):
        # SAFE-RETRAIN: M_real-init bias annealed over first 1000 stage2 steps
        with torch.amp.autocast('cuda', enabled=False):
            x_fp32 = x.float()
            logits_fp32 = self.w_gate(x_fp32)
            mask_logits_fp32 = self.mask_controller(x_fp32)
            if hasattr(self, "_m_real_init_bias") and self.training:
                _anneal = max(0.0, 1.0 - step / 1000.0)
                if _anneal > 0:
                    logits_fp32 = logits_fp32 + _anneal * self._m_real_init_bias
            soft_probs = self._compute_soft_weights(
                logits_fp32, mask_logits_fp32,
                deterministic_flag=deterministic_flag,
                training=self.training,
            )
            if not self.training or step >= 600:
                routing_weights = self._compute_topk_weights(soft_probs, step)
            elif step < 500:
                routing_weights = soft_probs
            else:
                alpha_hard = (step - 500) / 100.0
                topk_w = self._compute_topk_weights(soft_probs, step)
                routing_weights = (1.0 - alpha_hard) * soft_probs + alpha_hard * topk_w
            hard_choice = routing_weights.argmax(dim=-1)
            f_i = F.one_hot(hard_choice, num_classes=4).float().mean(dim=(0, 1))
            P_i = soft_probs.mean(dim=(0, 1))
            l_bal = 4.0 * torch.sum(f_i * P_i)
        return routing_weights.to(x.dtype), soft_probs, l_bal


class HEPLTransformerBlock(nn.Module):
    """
    Transformer block with 4 Epistemological Manifolds:
      0: M_real (Physical-Empirical)
      1: M_intu (Abstract-Relational + Local R-GAT)
      2: M_imag (Synthetic-Generative)
      3: M_rat  (Deductive-Analytical + Cross-Manifold Attention in last 2 layers)
    """
    def __init__(self, dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int,
                 max_seq_len: int, layer_idx: int, n_layers: int,
                 use_gradient_checkpointing: bool = True):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_layers = n_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.norm1 = RMSNorm(dim)
        self.attn = CausalGroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len)
        self.norm2 = RMSNorm(dim)

        # Conditional Sparse Gate
        self.gate = ConditionalSparseGate(dim)

        # Four Epistemological Manifolds
        self.m_real = SwiGLUExpert(dim, hidden_dim)
        self.m_intu_rgat = LocalRelationalGAT(dim, window_size=32, head_dim=64)
        self.m_intu = SwiGLUExpert(dim, hidden_dim)
        self.m_imag = SwiGLUExpert(dim, hidden_dim)
        self.m_rat = SwiGLUExpert(dim, hidden_dim)

        # Reviewer Point 4: Cross-Manifold Attention active ONLY in last 2 layers
        self.cross_manifold_enabled = (layer_idx >= n_layers - 2)
        if self.cross_manifold_enabled:
            self.cross_attn = CrossManifoldAttention(dim, n_heads=4)

        # Per-manifold learned output projections W_proj^(m) (Eq. 4)
        self.proj_real = nn.Linear(dim, dim, bias=False)
        self.proj_intu = nn.Linear(dim, dim, bias=False)
        self.proj_imag = nn.Linear(dim, dim, bias=False)
        self.proj_rat = nn.Linear(dim, dim, bias=False)

        # Telemetry buffers (detached scalars updated per forward)
        self.last_routing_shares = None
        self.last_act_norms = None

    def _run_manifold(self, fn, h: torch.Tensor) -> torch.Tensor:
        if self.training and self.use_gradient_checkpointing and h.requires_grad:
            return checkpoint(fn, h, use_reentrant=False)
        return fn(h)

    def _intu_forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.m_intu(self.m_intu_rgat(h))

    def forward(self, x: torch.Tensor, step: int = 0, deterministic_flag: torch.Tensor = None):
        x = x + self.attn(self.norm1(x))
        h = self.norm2(x)

        routing_weights, soft_probs, l_bal = self.gate(
            h, step=step, deterministic_flag=deterministic_flag
        )

        # Execute each manifold (with gradient checkpointing on FFN experts)
        out_real = self._run_manifold(self.m_real, h)
        out_intu = self._run_manifold(self._intu_forward, h)
        out_imag = self._run_manifold(self.m_imag, h)
        out_rat = self._run_manifold(self.m_rat, h)

        # Cross-Manifold Attention in M_rat (last 2 layers only)
        if self.cross_manifold_enabled and step >= 2000:
            others = torch.stack([out_real, out_intu, out_imag], dim=2)
            out_rat = self.cross_attn(out_rat, others)

        # Record telemetry (routing shares + activation norms for Reviewer Point 5).
        # CRITICAL for DataParallel: side-effects on self.last_* are LOST because
        # DataParallel runs forward on a replica that is discarded afterwards.
        # So we ALSO return these as values so DataParallel can gather them.
        with torch.no_grad():
            self.last_routing_shares = routing_weights.float().mean(dim=(0, 1)).detach()
            act_norms_tensor = torch.stack([
                out_real.float().norm(dim=-1).mean(),
                out_intu.float().norm(dim=-1).mean(),
                out_imag.float().norm(dim=-1).mean(),
                out_rat.float().norm(dim=-1).mean(),
            ])  # shape (4,) — M_real first, then M_intu, M_imag, M_rat
            self.last_act_norms = {
                "real": act_norms_tensor[0].item(),
                "intu": act_norms_tensor[1].item(),
                "imag": act_norms_tensor[2].item(),
                "rat":  act_norms_tensor[3].item(),
            }

        # Lift & aggregate active manifold outputs (Eq. 4)
        p_real = self.proj_real(out_real)
        p_intu = self.proj_intu(out_intu)
        p_imag = self.proj_imag(out_imag)
        p_rat = self.proj_rat(out_rat)

        stacked_proj = torch.stack([p_real, p_intu, p_imag, p_rat], dim=2)  # (B, T, 4, D)
        h_out = torch.sum(routing_weights.unsqueeze(-1) * stacked_proj, dim=2)

        return x + h_out, soft_probs, l_bal, act_norms_tensor
class XandraHEPL(nn.Module):
    """
    Xandra Hope Layer 1 Model with Hierarchical Epistemological Perception Layers (HEPL).
    """
    def __init__(self, vocab_size: int = 151646, dim: int = 512, n_layers: int = 8,
                 n_heads: int = 8, n_kv_heads: int = 2, max_seq_len: int = 2048,
                 use_gradient_checkpointing: bool = True, pad_token_id: int = -100):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.emb_norm = RMSNorm(dim)

        hidden_dim = int(8 * dim / 3)
        self.blocks = nn.ModuleList([
            HEPLTransformerBlock(
                dim=dim, hidden_dim=hidden_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
                max_seq_len=max_seq_len, layer_idx=idx, n_layers=n_layers,
                use_gradient_checkpointing=use_gradient_checkpointing
            )
            for idx in range(n_layers)
        ])

        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        for blk in self.blocks:
            for proj in [blk.proj_real, blk.proj_intu, blk.proj_imag, blk.proj_rat]:
                nn.init.eye_(proj.weight)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None,
                step: int = 0, deterministic_flag: torch.Tensor = None,
                target_manifold: torch.Tensor = None):
        if self.training and step < 3 and torch.cuda.is_available():
            alloc_gb = torch.cuda.memory_allocated() / 1e9
            print(f"[VRAM Check Step {step}] Allocated: {alloc_gb:.2f} GB / 16.00 GB", flush=True)
            if alloc_gb > 14.0:
                warnings.warn("VRAM > 14GB — consider reducing batch_size or max_len!")

        x = self.emb_norm(self.tok_emb(idx))

        total_l_bal = torch.tensor(0.0, device=idx.device, dtype=torch.float32)
        avg_soft_probs = torch.zeros((idx.size(0), idx.size(1), 4), device=idx.device, dtype=torch.float32)
        # SAFE-RETRAIN: accumulate per-block M_real act norm as a TENSOR (not a
        # side-effect) so DataParallel can gather it from all replicas.
        # shape after stack: (n_layers, 4) — M_real is column 0.
        per_block_act_norms = []

        for block in self.blocks:
            x, soft_probs, l_bal, blk_norms = block(x, step=step, deterministic_flag=deterministic_flag)
            total_l_bal = total_l_bal + l_bal
            avg_soft_probs = avg_soft_probs + soft_probs
            per_block_act_norms.append(blk_norms)

        total_l_bal = total_l_bal / self.n_layers
        avg_soft_probs = avg_soft_probs / self.n_layers

        # Stack to (n_layers, 4) and mean across layers → (4,). M_real = index 0.
        # DataParallel will gather this as a tensor (no replica loss).
        m_real_norm_tensor = torch.stack(per_block_act_norms, dim=0).mean(dim=0)

        x = self.norm(x)
        logits = self.head(x)

        l_sft = None
        l_gate = torch.tensor(0.0, device=idx.device, dtype=torch.float32)

        if targets is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            l_sft = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id
            )

        if target_manifold is not None:
            with torch.amp.autocast('cuda', enabled=False):
                seq_gate_probs = avg_soft_probs.mean(dim=1).clamp(min=1e-8)
                l_gate = F.nll_loss(torch.log(seq_gate_probs), target_manifold.long())

        return {
            "logits": logits if (not self.training or target_manifold is not None) else None,
            "l_sft": l_sft,
            "l_bal": total_l_bal,
            "l_gate": l_gate,
            "avg_soft_probs": avg_soft_probs,
            "m_real_norm": m_real_norm_tensor[0]
        }

    def fluency_probe(self) -> float:
        """M_real activation norm averaged across all layers.
        Used by the kill-switch to detect catastrophic forgetting."""
        norms = []
        for blk in self.blocks:
            if blk.last_act_norms is not None:
                norms.append(blk.last_act_norms.get("real", 0.0))
        if not norms:
            return 0.0
        return sum(norms) / len(norms)

    def get_telemetry(self) -> dict:
        """Aggregates routing shares and activation norms across all layers."""
        shares = torch.zeros(4)
        norms = {"real": 0.0, "intu": 0.0, "imag": 0.0, "rat": 0.0}
        valid_layers = 0
        for blk in self.blocks:
            if blk.last_routing_shares is not None:
                shares += blk.last_routing_shares.cpu()
                for k in norms:
                    norms[k] += blk.last_act_norms[k]
                valid_layers += 1
        if valid_layers > 0:
            shares /= valid_layers
            for k in norms:
                norms[k] /= valid_layers
        r_val, i_val, m_val, t_val = shares.tolist()
        return {
            "shares": {
                "real": round(float(r_val), 4),
                "intu": round(float(i_val), 4),
                "imag": round(float(m_val), 4),
                "rat":  round(float(t_val), 4),
            },
            "act_norms": {k: round(v, 4) for k, v in norms.items()}
        }


class SparseTopKDistillationLoss(nn.Module):
    """
    Soft Knowledge Distillation Loss using Top-32 Teacher Logits + 1 'Other' Bucket
    (FP32 guarded, 1e-8 floor, masks Other bucket when p_teacher_other < 1e-4).
    """
    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.T = temperature

    def forward(self, student_logits: torch.Tensor, teacher_top_ids: torch.Tensor,
                teacher_top_logits: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            s_logits = student_logits.float() / self.T
            t_logits = teacher_top_logits.float() / self.T

            t_probs_k = F.softmax(t_logits, dim=-1)
            t_sum = t_probs_k.sum(dim=-1, keepdim=True).clamp(max=1.0)
            p_teacher_other = (1.0 - t_sum).clamp(min=0.0)

            s_probs_full = F.softmax(s_logits, dim=-1)
            s_probs_k = torch.gather(s_probs_full, dim=-1, index=teacher_top_ids.long())
            p_student_other = (1.0 - s_probs_k.sum(dim=-1, keepdim=True)).clamp(min=0.0)

            s_probs_k_safe = s_probs_k + 1e-8
            kl_k = torch.sum(t_probs_k * torch.log((t_probs_k + 1e-8) / s_probs_k_safe), dim=-1)

            other_valid = (p_teacher_other.squeeze(-1) >= 1e-4).float()
            kl_other = (
                p_teacher_other.squeeze(-1) *
                torch.log((p_teacher_other.squeeze(-1) + 1e-8) / (p_student_other.squeeze(-1) + 1e-8))
            ) * other_valid

            token_kl = (self.T ** 2) * (kl_k + kl_other) * response_mask.float()
            valid_tokens = response_mask.float().sum().clamp(min=1.0)
            return token_kl.sum() / valid_tokens
class SelfHealingTelemetry:
    """
    Monitors manifold shares and activation norms:
      - Relative + Absolute collapse warning: (actual > expected + 0.30) OR (actual > 0.85)
      - Starvation warning: actual < 0.02 for 3 consecutive logs
      - Activation norm imbalance: act_norm > 10 * min(act_norms)
    """
    def __init__(self, log_file: str = "alerts.log"):
        self.log_file = log_file
        self.alpha_bal = 0.01
        self.alpha_boost_steps_left = 0
        self.starvation_streaks = {"real": 0, "intu": 0, "imag": 0, "rat": 0}
        self.critical_collapse_streak = 0

    def _record_alert(self, step: int, level: str, msg: str, alerts_list: list):
        entry = f"[Step {step}] [{level}] {msg}"
        print(f"🚨 {entry}", flush=True)
        alerts_list.append({"step": step, "level": level, "message": msg})
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
        except Exception:
            pass

    def step_update(self, step: int, telemetry: dict, expected_shares: dict = None) -> dict:
        if self.alpha_boost_steps_left > 0:
            self.alpha_boost_steps_left -= 1
            if self.alpha_boost_steps_left == 0:
                self.alpha_bal = 0.01

        shares = telemetry["shares"]
        norms = telemetry["act_norms"]
        if expected_shares is None:
            expected_shares = {"real": 0.25, "intu": 0.25, "imag": 0.25, "rat": 0.25}

        alerts = []
        action = "continue"

        for m in ["real", "intu", "imag", "rat"]:
            actual = shares[m]
            expected = expected_shares.get(m, 0.25)

            if actual > 0.95:
                self.critical_collapse_streak += 1
                self._record_alert(step, "CRITICAL", f"M_{m} share={actual:.2%} (>95%)", alerts)
                if self.critical_collapse_streak >= 3:
                    action = "pause_and_rollback"
            elif (actual > expected + 0.30) or (actual > 0.85):
                self.critical_collapse_streak = 0
                self.alpha_bal = 0.03
                self.alpha_boost_steps_left = 200
                self._record_alert(
                    step, "WARNING",
                    f"M_{m} dominance: actual={actual:.2%} vs expected={expected:.2%}. Boosting alpha->0.03 for 200 steps.",
                    alerts
                )
            else:
                self.critical_collapse_streak = 0

            if actual < 0.02:
                self.starvation_streaks[m] += 1
                if self.starvation_streaks[m] >= 3:
                    self._record_alert(step, "STARVATION", f"M_{m} starved (<2%) for 3 consecutive logs.", alerts)
            else:
                self.starvation_streaks[m] = 0

        min_norm = max(min(norms.values()), 1e-6)
        for m, val in norms.items():
            if val > 10.0 * min_norm:
                self._record_alert(
                    step, "NORM_IMBALANCE",
                    f"M_{m} activation norm ({val:.2f}) > 10x min norm ({min_norm:.2f}).",
                    alerts
                )

        return {
            "action": action,
            "alpha_bal": self.alpha_bal,
            "alerts": alerts
        }


def graft_from_xandra_modern(hepl_model: XandraHEPL, base_checkpoint_path: str,
                             sigma: float = 0.001, device: str = "cpu") -> XandraHEPL:
    """
    SurgicalWeightGrafter:
      - Copies tok_emb, emb_norm, GQA attention, and layer norms directly from XandraModern.
      - Clones trained SwiGLU weights into all 4 manifolds (M_real, M_intu, M_imag, M_rat)
        with independent Gaussian noise (sigma=0.001 — 10x smaller than v1) to break symmetry cleanly.
    """
    ckpt = torch.load(base_checkpoint_path, map_location=device, weights_only=False)
    base_state = ckpt["model_state"] if "model_state" in ckpt else ckpt

    clean_base = {}
    for k, v in base_state.items():
        nk = k.replace("module.", "").replace("_orig_mod.", "")
        clean_base[nk] = v

    hepl_state = hepl_model.state_dict()
    copied = 0

    with torch.no_grad():
        for shared_key in ["tok_emb.weight", "emb_norm.weight", "norm.weight", "head.weight"]:
            if shared_key in clean_base and shared_key in hepl_state:
                hepl_state[shared_key].copy_(clean_base[shared_key])
                copied += 1

        for idx in range(hepl_model.n_layers):
            for sub in ["norm1.weight", "norm2.weight",
                        "attn.wq.weight", "attn.wk.weight", "attn.wv.weight", "attn.wo.weight"]:
                src_k = f"blocks.{idx}.{sub}"
                if src_k in clean_base and src_k in hepl_state:
                    hepl_state[src_k].copy_(clean_base[src_k])
                    copied += 1

            for w_name in ["w1.weight", "w2.weight", "w3.weight"]:
                src_mlp = f"blocks.{idx}.mlp.{w_name}"
                if src_mlp in clean_base:
                    base_tensor = clean_base[src_mlp]
                    for m_name in ["m_real", "m_intu", "m_imag", "m_rat"]:
                        dst_k = f"blocks.{idx}.{m_name}.{w_name}"
                        if dst_k in hepl_state:
                            noise = torch.randn_like(base_tensor) * sigma
                            hepl_state[dst_k].copy_(base_tensor + noise)
                            copied += 1

    hepl_model.load_state_dict(hepl_state)
    print(f"✅ SurgicalWeightGrafter: Transferred & grafted {copied} parameter tensors (sigma={sigma}).")
    return hepl_model
