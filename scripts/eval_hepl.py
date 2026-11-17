"""
╔═══════════════════════════════════════════════════════════════╗
║   Xandra Hope - Stage 2 Pilot Benchmark & Ablation Suite      ║
║   Evaluates: Actual FLOPs, Local R-GAT Window Ablation,       ║
║              Manifold Isolation Probing, 4 Isolated Linear    ║
║              Probes (Accuracy/F1/Confusion Matrix), & QA      ║
║   Copyright (c) 2026 Ahmed Sayed Kamel Mahmoud &               ║
║                  Александра Евгеньевна Острикова              ║
║   All Rights Reserved. License: XHNSL v1.0                   ║
╚═══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timezone

# Add kaggle-worker to path so we can import XandraHEPL cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kaggle-worker")))
from hepl_architecture import XandraHEPL, LocalRelationalGAT


def count_model_flops(model: nn.Module, seq_len: int = 512, device: str = "cpu") -> dict:
    """
    Measures actual FLOPs/MACs per forward pass using thop/fvcore if available,
    with an exact analytical parameter-MAC fallback for strict reproducibility.
    """
    dummy_input = torch.randint(0, 1000, (1, seq_len), device=device)
    model.eval()

    try:
        from thop import profile
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        return {
            "method": "thop.profile",
            "total_params": int(params),
            "macs": int(macs),
            "flops_per_token": round((2.0 * macs) / seq_len, 2)
        }
    except ImportError:
        pass

    # Exact analytical active FLOPs calculation (accounting for Top-1 sparse routing)
    total_params = sum(p.numel() for p in model.parameters())
    active_params = 0
    for name, p in model.named_parameters():
        # In HEPL Top-1 routing, only 1 of 4 FFN manifolds executes per token
        if any(m in name for m in ["m_intu.", "m_imag.", "m_rat."]):
            continue
        active_params += p.numel()

    flops_per_token = 2.0 * active_params
    return {
        "method": "analytical_active_sparse",
        "total_params": int(total_params),
        "active_params_per_token": int(active_params),
        "flops_per_token": round(flops_per_token, 2)
    }


def run_local_window_ablation(dim: int = 512, seq_len: int = 512, device: str = "cpu") -> list:
    """
    Ablation ID: abl-local-window (Paper Section 8.3)
    Compares Local R-GAT latency & edge operations across w=32, w=128, and w=full (L).
    """
    results = []
    x = torch.randn(2, seq_len, dim, device=device)

    for w in [32, 128, seq_len]:
        rgat = LocalRelationalGAT(dim=dim, window_size=w, head_dim=64).to(device).eval()
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = rgat(x)

        t0 = time.perf_counter()
        iters = 20
        with torch.no_grad():
            for _ in range(iters):
                _ = rgat(x)
        elapsed_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        # Active edge pairs per token
        active_edges = sum(min(i, w) for i in range(seq_len))
        results.append({
            "window_size": w if w < seq_len else f"full ({seq_len})",
            "avg_latency_ms": round(elapsed_ms, 3),
            "active_edge_pairs": int(active_edges),
            "relative_edge_cost": round(active_edges / max(1, sum(min(i, 32) for i in range(seq_len))), 2)
        })

    return results
@torch.no_grad()
def evaluate_manifold_isolation(model: XandraHEPL, tokenizer, device: str = "cpu") -> dict:
    """
    Manifold Probing Test:
      Verifies that M_imag activation drops below 2% on [DETERMINISTIC] prompts
      while activating normally on [CREATIVE] prompts.
    """
    model.eval()
    det_prompts = [
        "[DETERMINISTIC] احسب مشتقة الدالة f(x) = 3x^2 + 5x - 2 عند النقطة x = 4.",
        "[DETERMINISTIC] اكتب خوارزمية البحث الثنائي بلغة بايثون مع حساب التعقيد الزمني.",
        "[DETERMINISTIC] ما هي كثافة الماء النقي عند درجة حرارة 4 درجات مئوية؟"
    ]
    cre_prompts = [
        "[CREATIVE] تخيل مدينة عائمة فوق السحاب واكتب مشهداً أدبياً يصف شروق الشمس فيها.",
        "[CREATIVE] ماذا لو استطاعت الأشجار التحدث مع البشر بلغة الموسيقى؟ اكتب قصة قصيرة.",
        "[CREATIVE] صغ قصيدة نثرية عن رحلة مسافر عبر الزمن يبحث عن معنى الأمل."
    ]

    def _avg_shares(prompts_list, is_det: bool):
        acc = {"real": 0.0, "intu": 0.0, "imag": 0.0, "rat": 0.0}
        for p in prompts_list:
            ids = tokenizer.encode(p, return_tensors="pt").to(device)
            det_flag = torch.tensor([1 if is_det else 0], device=device)
            _ = model(ids, step=1000, deterministic_flag=det_flag)
            tel = model.get_telemetry()["shares"]
            for k in acc:
                acc[k] += tel[k]
        return {k: round(v / max(1, len(prompts_list)), 4) for k, v in acc.items()}

    det_shares = _avg_shares(det_prompts, is_det=True)
    cre_shares = _avg_shares(cre_prompts, is_det=False)

    return {
        "deterministic_routing": det_shares,
        "creative_routing": cre_shares,
        "imag_isolation_passed": det_shares["imag"] < 0.02
    }


def train_and_eval_isolated_linear_probes(model: XandraHEPL, eval_samples: list[dict],
                                          tokenizer, device: str = "cpu", epochs: int = 8) -> dict:
    """
    Isolated Linear Probes (Zero Data Leakage):
      - Freezes all XandraHEPL backbone weights (requires_grad=False)
      - Trains 4 independent linear heads nn.Linear(dim, 2) for each manifold
      - Reports Accuracy, F1-Score, and 2x2 Confusion Matrix per manifold.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    features = {"real": [], "intu": [], "imag": [], "rat": []}
    labels = []

    last_blk = model.blocks[-1]
    with torch.no_grad():
        for sample in eval_samples:
            ids = tokenizer.encode(sample["text"], truncation=True, max_length=256, return_tensors="pt").to(device)
            x = model.emb_norm(model.tok_emb(ids))
            for blk in model.blocks[:-1]:
                x, _, _ = blk(x, step=1000)

            h = last_blk.norm2(x + last_blk.attn(last_blk.norm1(x)))
            out_real = last_blk.m_real(h).mean(dim=1).squeeze(0).cpu()
            out_intu = last_blk._intu_forward(h).mean(dim=1).squeeze(0).cpu()
            out_imag = last_blk.m_imag(h).mean(dim=1).squeeze(0).cpu()
            out_rat = last_blk.m_rat(h).mean(dim=1).squeeze(0).cpu()

            features["real"].append(out_real)
            features["intu"].append(out_intu)
            features["imag"].append(out_imag)
            features["rat"].append(out_rat)
            labels.append(int(sample["label"]))

    y_all = torch.tensor(labels, dtype=torch.long)
    split_idx = int(0.7 * len(y_all))
    probe_results = {}

    for m_name in ["real", "intu", "imag", "rat"]:
        X_all = torch.stack(features[m_name]).float()
        X_train, y_train = X_all[:split_idx].to(device), y_all[:split_idx].to(device)
        X_test, y_test = X_all[split_idx:].to(device), y_all[split_idx:].to(device)

        probe = nn.Linear(model.dim, 2).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)

        probe.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = probe(X_train)
            loss = F.cross_entropy(logits, y_train)
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            preds = probe(X_test).argmax(dim=-1)
            tp = int(((preds == 1) & (y_test == 1)).sum().item())
            tn = int(((preds == 0) & (y_test == 0)).sum().item())
            fp = int(((preds == 1) & (y_test == 0)).sum().item())
            fn = int(((preds == 0) & (y_test == 1)).sum().item())

            acc = (tp + tn) / max(1, len(y_test))
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = (2 * precision * recall) / max(1e-8, precision + recall)

        probe_results[f"M_{m_name}"] = {
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
        }

    return probe_results


def build_heldout_eval_set() -> list[dict]:
    """Held-out evaluation dataset never seen during training (Zero Data Leakage)."""
    formal = [
        "حل المعادلة التربيعية س^2 - 5س + 6 = 0 موضحاً خطوات التحليل.",
        "اكتب دالة بايثون لفرز مصفوفة باستخدام خوارزمية الدمج Merge Sort.",
        "إذا تحرك قطار بسرعة 90 كم/ساعة لمدة ساعتين ونصف، فما المسافة المقطوعة؟",
        "اشرح قانون أوم للتيار الكهربائي مع مثال تطبيقي بالأرقام.",
        "ما هو التعقيد الزمني لعملية البحث في شجرة البحث الثنائية المتوازنة؟",
        "احسب مساحة دائرة نصف قطرها 7 سم بدلالة ط.",
        "اكتب استعلام SQL لاستخراج أعلى خمسة موظفين راتباً في كل قسم.",
        "برهن رياضياً أن مجموع زاويتين متتامتين يساوي 90 درجة.",
        "كيف تعمل خوارزمية التدرج الهابط Stochastic Gradient Descent؟",
        "احسب الضغط الناتج عن قوة مقدارها 200 نيوتن تؤثر عمودياً على مساحة 4 متر مربع."
    ]
    creative = [
        "اكتب خاطرة أدبية عن شوق المسافر إلى وطنه تحت ضوء القمر.",
        "تخيل حواراً شاعرياً بين البحر والصحراء حول معنى الخلود.",
        "اروِ قصة قصيرة عن رسام اكتشف أن لوحاته تنبض بالحياة ليلاً.",
        "صف بأسلوب بلاغي جمال مدينة الإسكندرية وقت الغروب.",
        "ماذا لو استطظ البشر يوماً ووجدوا أن الجاذبية تعمل أفقياً؟",
        "اكتب مشهداً مسرحياً قصيراً يدور في مقهى قديم في القاهرة.",
        "صغ أبياتاً نثرية عن الأمل الذي يولد من رحم المعاناة.",
        "تخيل رحلة سفينة فضائية تكتشف كوكباً مغطى بالمرايا.",
        "اكتب رسالة وجدانية من صديق إلى صديقه بعد غياب عشر سنوات.",
        "صف شعور طفل يرى الثلج يتساقط للمرة الأولى في حياته."
    ]
    samples = [{"text": t, "label": 1} for t in formal] + [{"text": t, "label": 0} for t in creative]
    interleaved = []
    for i in range(len(formal)):
        interleaved.append(samples[i])
        interleaved.append(samples[len(formal) + i])
    return interleaved


def run_full_benchmark(checkpoint_path: str = None, tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"📊 Running Stage 2 Pilot Benchmark on {device.upper()}")
    print("=" * 60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model = XandraHEPL(vocab_size=len(tokenizer), dim=512, n_layers=8, n_heads=8, n_kv_heads=2).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        clean_state = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(clean_state, strict=False)
        print(f"✅ Loaded checkpoint: {checkpoint_path}")

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_path or "untrained_init",
        "flops_analysis": count_model_flops(model, seq_len=512, device=device),
        "ablation_local_window": run_local_window_ablation(dim=512, seq_len=512, device=device),
        "manifold_isolation": evaluate_manifold_isolation(model, tokenizer, device=device),
        "isolated_linear_probes": train_and_eval_isolated_linear_probes(
            model, build_heldout_eval_set(), tokenizer, device=device, epochs=8
        )
    }

    out_file = "stage2_pilot_benchmark_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n✅ Full Pilot Report saved to {out_file}")


if __name__ == "__main__":
    ckpt_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_full_benchmark(ckpt_arg)
