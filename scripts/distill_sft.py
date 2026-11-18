"""
╔═══════════════════════════════════════════════════════════════╗
║   Xandra Hope - Offline Knowledge Distillation Pipeline (v1.0)  ║
║   Teacher: Qwen/Qwen2.5-7B-Instruct (Top-32 Sparse Logits)    ║
║   Copyright (c) 2026 Ahmed Sayed Kamel Mahmoud &               ║
║                  Александра Евгеньевна Острикова              ║
║   All Rights Reserved. License: XHNSL v1.0                   ║
╚═══════════════════════════════════════════════════════════════╝

Note on KLDivLoss & 32+1 Bucket:
  This script stores Top-32 token IDs and log-probabilities/logits for RESPONSE
  tokens only (prompt tokens are masked out to save storage). During training,
  SparseTopKDistillationLoss adds a 33rd 'Other' bucket for residual probability
  mass (1 - sum(top_32_probs)) with reduction='batchmean'.
"""

import os
import re
import json
import time
import math
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone
from datasets import load_dataset
from huggingface_hub import HfApi, upload_file

HF_TOKEN = os.environ.get("HF_TOKEN_OWNER") or os.environ.get("HF_TOKEN")
SFT_REPO = os.environ.get("HF_SFT_REPO", "smart-force/xandra-sft-corpus")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "Qwen/Qwen2.5-7B-Instruct")
BATCH_LIMIT = int(os.environ.get("DISTILL_BATCH_LIMIT", "500"))
TOP_K = 32

if not HF_TOKEN:
    raise ValueError("❌ Missing HF_TOKEN_OWNER / HF_TOKEN environment variable.")

HF_API_URL = "https://" + "router.huggingface.co/hf-inference/models/" + TEACHER_MODEL
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN.strip()}",
    "Content-Type": "application/json"
}


def classify_epistemic_domain(prompt: str, response: str) -> tuple[str, int]:
    """
    Classifies a dialogue pair into control_tag and target_manifold:
      0: M_real (Physical-Empirical: physics, medicine, measurements, geography)
      1: M_intu (Abstract-Relational: philosophy, law, ethics, social relations)
      2: M_imag (Synthetic-Generative: poetry, fiction, counterfactuals, stories)
      3: M_rat  (Deductive-Analytical: math, code, logic, step-by-step proofs)
    """
    combined = f"{prompt} {response}".lower()

    # 3: Deductive-Analytical (Math, Code, Logic)
    rat_patterns = [
        r"\bdef\b", r"\bclass\b", r"\bimport\b", r"```", r"معادلة", r"احسب",
        r"برهان", r"خوارزمية", r"برمجة", r"كود", r"رياضيات", r"مشتقة", r"تكامل",
        r"[0-9]+\s*[\+\-\*\/\=]\s*[0-9]+"
    ]
    if any(re.search(p, combined) for p in rat_patterns):
        return "[DETERMINISTIC]", 3

    # 2: Synthetic-Generative (Creative, Counterfactuals, Poetry)
    imag_patterns = [
        r"تخيل", r"قصة", r"شعر", r"قصيدة", r"رواية", r"سيناريو", r"ماذا لو",
        r"لو انعكست", r"خيال", r"أدب", r"بلاغة", r"imagine", r"what if", r"story"
    ]
    if any(re.search(p, combined) for p in imag_patterns):
        return "[CREATIVE]", 2

    # 0: Physical-Empirical (Medicine, Chemistry, Engineering, Geography)
    real_patterns = [
        r"طب", r"علاج", r"مرض", r"فيزياء", r"كيمياء", r"درجة حرارة", r"كتلة",
        r"سرعة", r"متر", r"كيلومتر", r"جغرافيا", r"تجربة", r"قياس", r"تشريح"
    ]
    if any(re.search(p, combined) for p in real_patterns):
        return "[DETERMINISTIC]", 0

    # 1: Abstract-Relational (Philosophy, Law, Economics, Governance)
    return "[DETERMINISTIC]", 1


def call_teacher_cot_and_probe(prompt: str, max_retries: int = 3) -> dict:
    """
    Queries Qwen2.5-7B-Instruct to generate:
      1) Detailed Arabic Chain-of-Thought (CoT) answer
      2) A dialectical counter-argument probe A*(y) for Stage 4 (LDR)
    """
    system_instruction = (
        "أنت معلم عربي خبير في التفكير المنطقي والتحليل العلمي. "
        "أجب عن سؤال المستخدم بأسلوب واضح ومنظم خطوة بخطوة، ثم أضف في النهاية "
        "كتلة مفصولة بوسم [PROBE] تحتوي على سؤال نقدي أو حجة مضادة لاختبار دقة الإجابة."
    )
    formatted_input = (
        f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    payload = {
        "inputs": formatted_input,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.4,
            "top_p": 0.9,
            "return_full_text": False,
            "details": True,
            "top_n_tokens": TOP_K
        },
        "options": {"wait_for_model": True}
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=90)
            if resp.status_code == 200:
                data = resp.json()
                item = data[0] if isinstance(data, list) else data
                gen_text = item.get("generated_text", "").strip()
                details = item.get("details", {})
                tokens_info = details.get("tokens", [])

                # Split CoT response and Stage 4 Counter Probe
                if "[PROBE]" in gen_text:
                    parts = gen_text.split("[PROBE]", 1)
                    cot_response = parts[0].strip()
                    counter_probe = parts[1].strip()
                else:
                    cot_response = gen_text
                    counter_probe = "هل توجد افتراضات ضمنية أو حالات حدية قد تغير هذه النتيجة؟"

                # Extract Top-32 token IDs and logprobs for response tokens
                top_ids_seq = []
                top_logprobs_seq = []
                for tok in tokens_info:
                    top_toks = tok.get("top_tokens", [])
                    if not top_toks:
                        continue
                    ids_row = [int(t.get("id", 0)) for t in top_toks[:TOP_K]]
                    lps_row = [float(t.get("logprob", -20.0)) for t in top_toks[:TOP_K]]
                    # Pad if fewer than TOP_K returned
                    while len(ids_row) < TOP_K:
                        ids_row.append(0)
                        lps_row.append(-30.0)
                    top_ids_seq.append(ids_row)
                    top_logprobs_seq.append(lps_row)

                return {
                    "cot_response": cot_response,
                    "counter_probe": counter_probe,
                    "top32_ids": top_ids_seq,
                    "top32_logprobs": top_logprobs_seq
                }
            elif resp.status_code in (429, 503):
                time.sleep(10 * (attempt + 1))
            else:
                print(f"⚠️ Teacher API HTTP {resp.status_code}: {resp.text[:150]}")
                time.sleep(3)
        except Exception as e:
            print(f"⚠️ Teacher request error: {e}")
            time.sleep(5)

    return None
def load_seed_prompts(limit: int = BATCH_LIMIT) -> list[str]:
    """
    Loads diverse Arabic & technical seed prompts from Hugging Face datasets
    to ensure balanced coverage across all 4 epistemological manifolds.
    """
    prompts = []
    seed_sources = [
        ("arbml/CIDAR", "train", "instruction"),
        ("FreedomIntelligence/evol-instruct-arabic", "train", "conversations"),
    ]

    for ds_name, split, field in seed_sources:
        try:
            ds = load_dataset(ds_name, split=split, streaming=True)
            ds = ds.shuffle(seed=int(time.time()), buffer_size=2000)
            for row in ds:
                if len(prompts) >= limit:
                    break
                val = row.get(field)
                if isinstance(val, list) and len(val) > 0:
                    first_turn = val[0]
                    q_text = first_turn.get("value", "") if isinstance(first_turn, dict) else str(first_turn)
                else:
                    q_text = str(val or "")
                q_text = q_text.strip()
                if 15 <= len(q_text) <= 800:
                    prompts.append(q_text)
        except Exception as e:
            print(f"⚠️ Could not stream {ds_name}: {e}")

        if len(prompts) >= limit:
            break

    # Built-in balanced fallback prompts if external streams are rate-limited
    if len(prompts) < 10:
        builtin = [
            "اشرح خطوة بخطوة كيف نحسب الطاقة الحركية لجسم كتلته 5 كجم ويتحرك بسرعة 10 م/ث.",
            "اكتب دالة بايثون لحساب أعداد فيبوناتشي باستخدام البرمجة الديناميكية مع تحليل التعقيد الزمني.",
            "ما الفرق الجوهري بين العدالة التوزيعية والعدالة الإجرائية في الفلسفة القانونية؟",
            "تخيل كيف ستتغير هندسة المدن ووسائل النقل لو انخفضت جاذبية الأرض إلى النصف."
        ]
        prompts.extend(builtin * max(1, (limit - len(prompts)) // len(builtin) + 1))

    return prompts[:limit]


def run_distillation():
    print("=" * 60)
    print(f"🎓 Starting Offline Knowledge Distillation ({TEACHER_MODEL})")
    print(f"🎯 Target Batch Size: {BATCH_LIMIT} examples -> {SFT_REPO}")
    print("=" * 60)

    prompts = load_seed_prompts(BATCH_LIMIT)
    records = []
    manifold_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    start_ts = time.time()

    for idx, prompt in enumerate(prompts):
        res = call_teacher_cot_and_probe(prompt)
        if not res or not res["cot_response"]:
            continue

        control_tag, target_manifold = classify_epistemic_domain(prompt, res["cot_response"])
        manifold_counts[target_manifold] += 1

        records.append({
            "prompt": prompt,
            "cot_response": res["cot_response"],
            "control_tag": control_tag,
            "target_manifold": int(target_manifold),
            "teacher_top32_ids": json.dumps(res["top32_ids"]),
            "teacher_top32_logprobs": json.dumps(res["top32_logprobs"]),
            "counter_probe": res["counter_probe"],
            "teacher_model": TEACHER_MODEL,
            "created_at": datetime.now(timezone.utc).isoformat()
        })

        if (idx + 1) % 25 == 0:
            elapsed = time.time() - start_ts
            rate = (idx + 1) / max(elapsed, 0.1)
            print(f"[{idx+1}/{len(prompts)}] Distilled: {len(records)} | Rate: {rate:.2f} ex/s | Manifolds: {manifold_counts}", flush=True)

    if not records:
        print("❌ No distilled records generated in this run.")
        return

    df = pd.DataFrame(records)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    local_file = f"distilled_batch_{stamp}.parquet"
    df.to_parquet(local_file, engine="pyarrow", compression="snappy", index=False)

    size_mb = os.path.getsize(local_file) / (1024 * 1024)
    print(f"💾 Saved {len(records)} distilled examples to {local_file} ({size_mb:.2f} MB)")

    repo_path = f"distilled/{local_file}"
    upload_file(
        path_or_fileobj=local_file,
        path_in_repo=repo_path,
        repo_id=SFT_REPO,
        repo_type="dataset",
        token=HF_TOKEN.strip()
    )
    print(f"✅ Uploaded successfully to HuggingFace: {SFT_REPO}/{repo_path}")


if __name__ == "__main__":
    run_distillation()
