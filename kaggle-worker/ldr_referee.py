"""
╔═══════════════════════════════════════════════════════════════╗
║   Xandra Hope - Layer 3: Lateral Dialectic Referee (v1.0)     ║
║   Copyright (c) 2026 Ahmed Sayed Kamel Mahmoud &               ║
║                  Александра Евгеньевна Острикова              ║
║   All Rights Reserved. License: XHNSL v1.0                   ║
╚═══════════════════════════════════════════════════════════════╝

Implements Layer 3 (Lateral Dialectic Referee - LDR) of the Triadic Cognitive
Architecture (TCA), matching Section 6 (Eqs. 12-20) of the research paper:
  - Dialectic Probe Generator P_phi (~10% parameter footprint of Layer 1)
  - Cross-Attention Probe Conditioning over (Context x, Draft y_draft)
  - Margin-Based Anti-Template Divergence Loss (Eq. 20):
      L_anti_templ = max(0, m - KL(q_P_phi || q_templ))
  - Dialectic Refinement Loss L_dialectic for self-correction without Veto
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbeDecoderBlock(nn.Module):
    """Lightweight Transformer block for the Lateral Dialectic Referee P_phi."""
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(dim * 2, dim, bias=False)
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        causal_mask = torch.triu(torch.ones((T, T), device=x.device, dtype=torch.bool), diagonal=1)

        h1 = self.norm1(x)
        attn_out, _ = self.self_attn(h1, h1, h1, attn_mask=causal_mask, is_causal=True)
        x = x + attn_out

        h2 = self.norm2(x)
        cross_out, _ = self.cross_attn(h2, memory, memory)
        x = x + cross_out

        x = x + self.ffn(self.norm3(x))
        return x


class DialecticProbeGenerator(nn.Module):
    """
    Section 6.1: Lateral Dialectic Referee P_phi (~10% of Layer 1 parameters).
    Reads the conversation context and Layer 1's initial draft response y_draft,
    and generates targeted dialectical counter-probes q_probe to expose hidden
    assumptions, edge cases, or logical fallacies.
    """
    def __init__(self, vocab_size: int = 151646, shared_tok_emb: nn.Embedding = None,
                 base_dim: int = 512, probe_dim: int = 256, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.probe_dim = probe_dim

        # Project from shared Layer 1 embedding space to compact referee dimension
        self.tok_emb = shared_tok_emb if shared_tok_emb is not None else nn.Embedding(vocab_size, base_dim)
        self.in_proj = nn.Linear(base_dim, probe_dim, bias=False)
        self.mem_proj = nn.Linear(base_dim, probe_dim, bias=False)

        self.blocks = nn.ModuleList([
            ProbeDecoderBlock(dim=probe_dim, n_heads=n_heads)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(probe_dim)
        self.out_proj = nn.Linear(probe_dim, base_dim, bias=False)
        self.lm_head = nn.Linear(base_dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, probe_input_ids: torch.Tensor, draft_hidden_states: torch.Tensor,
                probe_targets: torch.Tensor = None) -> dict:
        """
        probe_input_ids:     (B, T_probe) tokenized prompt/probe prefix
        draft_hidden_states: (B, T_draft, base_dim) hidden states of Layer 1 draft
        probe_targets:       (B, T_probe) gold counter-probe token IDs (from distilled A*(y))
        """
        with torch.no_grad():
            emb = self.tok_emb(probe_input_ids)

        x = self.in_proj(emb)
        mem = self.mem_proj(draft_hidden_states)

        for blk in self.blocks:
            x = blk(x, mem)

        x = self.out_proj(self.norm(x))
        logits = self.lm_head(x)

        l_probe = torch.tensor(0.0, device=probe_input_ids.device, dtype=torch.float32)
        if probe_targets is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = probe_targets[..., 1:].contiguous()
            l_probe = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return {
            "probe_logits": logits,
            "l_probe": l_probe
        }


class MarginAntiTemplateLoss(nn.Module):
    """
    Section 6.3 (Eq. 20): Margin-Based Anti-Template Divergence Loss:
      L_anti_templ = max(0, m - KL(q_P_phi || q_templ))
    Prevents P_phi from collapsing into generic repetitive questions
    (e.g., 'Are you sure?') while bounding gradients cleanly via margin m=2.0.
    """
    def __init__(self, margin: float = 2.0, eta: float = 0.03):
        super().__init__()
        self.margin = margin
        self.eta = eta

    def forward(self, probe_logits: torch.Tensor, template_probs: torch.Tensor) -> torch.Tensor:
        """
        probe_logits:   (B, T, V) raw logits from P_phi
        template_probs: (V,) or (1, 1, V) empirical distribution of generic boilerplate probes
        """
        with torch.amp.autocast('cuda', enabled=False):
            p_phi = F.softmax(probe_logits.float(), dim=-1).mean(dim=(0, 1)).clamp(min=1e-8)  # (V,)
            q_tmpl = template_probs.float().view(-1).clamp(min=1e-8)
            q_tmpl = q_tmpl / q_tmpl.sum()

            kl_div = torch.sum(p_phi * torch.log(p_phi / q_tmpl))
            hinge = F.relu(self.margin - kl_div)
            return self.eta * hinge
class XandraLDRSystem(nn.Module):
    """
    Complete Layer 3 Integration: Lateral Dialectic Referee (LDR).
    Orchestrates the 3-step lateral dialectic loop (Section 6.2):
      1. Layer 1 produces initial draft representation y_draft
      2. Layer 3 (P_phi) inspects (x, y_draft) and generates dialectical probe q_probe
      3. Layer 1 refines output into y_refined conditioned on [x; y_draft; q_probe]
         without hard refusal (Veto).
    """
    def __init__(self, vocab_size: int = 151646, shared_tok_emb: nn.Embedding = None,
                 base_dim: int = 512, probe_dim: int = 256, n_layers: int = 2,
                 margin: float = 2.0, eta: float = 0.03, lambda_dial: float = 0.25):
        super().__init__()
        self.lambda_dial = lambda_dial
        self.probe_generator = DialecticProbeGenerator(
            vocab_size=vocab_size,
            shared_tok_emb=shared_tok_emb,
            base_dim=base_dim,
            probe_dim=probe_dim,
            n_layers=n_layers,
            n_heads=4
        )
        self.anti_template_loss = MarginAntiTemplateLoss(margin=margin, eta=eta)
        # Register uniform-smoothed boilerplate prior buffer (updated empirically during Stage 4)
        self.register_buffer("template_prior", torch.ones(vocab_size) / vocab_size)

    def update_template_prior(self, generic_token_ids: list[int], weight: float = 0.05):
        """Updates empirical prior of generic boilerplate words to penalize repetitive probes."""
        with torch.no_grad():
            prior = torch.ones_like(self.template_prior)
            for tid in generic_token_ids:
                if 0 <= tid < prior.size(0):
                    prior[tid] += weight * prior.size(0)
            self.template_prior.copy_(prior / prior.sum())

    def compute_stage4_loss(self, probe_input_ids: torch.Tensor, draft_hidden_states: torch.Tensor,
                            probe_targets: torch.Tensor, draft_logits: torch.Tensor,
                            refined_logits: torch.Tensor, gold_answer_targets: torch.Tensor,
                            pad_token_id: int = -100) -> dict:
        """
        Computes combined Stage 4 Dialectic Alignment Loss (Eqs. 18-20):
          L_stage4 = L_refined_sft + L_probe + L_anti_templ + lambda_dial * L_dialectic
        where L_dialectic rewards strictly lower NLL on y_refined compared to y_draft.
        """
        probe_out = self.probe_generator(
            probe_input_ids=probe_input_ids,
            draft_hidden_states=draft_hidden_states,
            probe_targets=probe_targets
        )
        l_probe = probe_out["l_probe"]
        l_anti_templ = self.anti_template_loss(probe_out["probe_logits"], self.template_prior)

        # Refined SFT loss
        shift_ref = refined_logits[..., :-1, :].contiguous()
        shift_gold = gold_answer_targets[..., 1:].contiguous()
        l_refined_sft = F.cross_entropy(
            shift_ref.view(-1, shift_ref.size(-1)),
            shift_gold.view(-1),
            ignore_index=pad_token_id
        )

        # Dialectic improvement margin loss: L_dialectic = relu(NLL(y_refined) - NLL(y_draft) + delta)
        with torch.no_grad():
            shift_draft = draft_logits[..., :-1, :].contiguous()
            l_draft_sft = F.cross_entropy(
                shift_draft.view(-1, shift_draft.size(-1)),
                shift_gold.view(-1),
                ignore_index=pad_token_id
            )

        l_dialectic = F.relu(l_refined_sft - l_draft_sft + 0.05)
        total_loss = l_refined_sft + l_probe + l_anti_templ + (self.lambda_dial * l_dialectic)

        return {
            "total_loss": total_loss,
            "l_refined_sft": float(l_refined_sft.item()),
            "l_draft_sft": float(l_draft_sft.item()),
            "l_probe": float(l_probe.item()),
            "l_anti_templ": float(l_anti_templ.item()),
            "l_dialectic": float(l_dialectic.item()),
            "dialectic_gain": float((l_draft_sft - l_refined_sft).item())
        }
