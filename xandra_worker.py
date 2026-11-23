#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  xandra_worker.py  —  VERSION 10.0  (Stage-2 HEPL Clean Restart: 150K → 180K)
================================================================================

المرجع: Xandra_Master_Engineering_Reference.docx (Single Source of Truth)

التعديلات الثلاثة المعتمدة (المرجع §2):
  1) Surgical Weight Grafting من base_150k.pt عبر graft_from_xandra_modern
     (لا load_state_dict مباشر) + AdamW نظيف جديد (لا وراثة optimizer_state).
  2) إصلاح Manifold Telemetry: إرجاع routing_shares كـ tensor [1, 4] في
     forward() dict ليجمعه DataParallel.gather تلقائياً عبر dim=0.
  3) Response-Only Loss Masking شامل <|im_end|> ومستبعد PAD_ID.

معادلة الخسارة المعتمدة (المرجع §6):
    L_total = 0.50·L_distilled + 0.50·L_chatml + 0.05·L_gate + 0.50·L_KL

المراقبة الثلاثية (المرجع §7): Entropy / Gini / Drift كل 500 خطوة.
بوابة السلامة (المرجع §8): عند step 152,000 — ستة شروط قبل الاستمرار.

الهدف: 150,000 → 180,000 (30,000 خطوة نظيفة).
"""

import os
import sys
import json
import time
import math
import shutil
import random
import warnings
import traceback
from datetime import datetime, timezone

# ──────────────────────────────────────────────────────────────────────────────
# الواردات البرمجية
# ──────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# تثبيت الحزم الناقصة بصمت (نمط xandra_worker.py الأصلي)
def install(package):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

for _pkg in ["huggingface_hub", "transformers", "datasets", "tiktoken"]:
    try:
        __import__(_pkg)
    except ImportError:
        install(_pkg)

from huggingface_hub import hf_hub_download, upload_file, HfApi
from transformers import AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# الإعدادات العامة
# ──────────────────────────────────────────────────────────────────────────────
# تُحقن آلياً عند التشغيل على Kaggle (نمط xandra_worker.py الأصلي)
KAGGLE_USERNAME = "__KAGGLE_USERNAME_VALUE__"
INJECTED_HF_TOKEN = "__HF_TOKEN_VALUE__"
INJECTED_HF_REPO = "__HF_REPO_VALUE__"

HF_TOKEN = (INJECTED_HF_TOKEN if INJECTED_HF_TOKEN != "__HF_TOKEN_VALUE__" else os.environ.get("HF_TOKEN", "")).strip()
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN").strip()
    except Exception:
        pass
os.environ["HF_TOKEN"] = HF_TOKEN
HF_REPO = INJECTED_HF_REPO if INJECTED_HF_REPO != "__HF_REPO_VALUE__" else os.environ.get("HF_REPO", "smart-force/xandra-core-private")
HEPL_REPO = "smart-force/xandra-core-private"   # مستودع hepl_architecture.py

SESSION_START_TIME = time.time()
SESSION_STARTED_AT = datetime.now(timezone.utc).isoformat()
STARTUP_LOG = []

# ──────────────────────────────────────────────────────────────────────────────
# المعاملات المعتمدة — ثابتة من المرجع §2
# ──────────────────────────────────────────────────────────────────────────────
VOCAB_SIZE       = 151646           # Qwen2.5 tokenizer (ثابت)
MODEL_DIM        = 512               # XandraHEPL default
N_LAYERS         = 8                 # XandraHEPL default
N_HEADS          = 8                 # query heads
N_KV_HEADS       = 2                 # GQA — 4x fewer KV projections
MAX_LEN          = 512               # سياق التدريب
BATCH_SIZE       = 4                 # 4 لكل كرت × 2 كرت = 8 (آمن 100% من CUDA OOM على T4 x2)
                                  # التعديل 3 (المقيم): 8 × 512 × 151,646 × 4B = 2.48 GB لكل كرت
                                  # لـ logits وحدها في FP32 — قد يقترب من 14.5 GB. خفضنا لـ 4.
GRAD_ACCUM_STEPS = 16                # effective batch = 4 × 2 × 16 = 128
PREFETCH_BATCHES = 2

# نقطة الانطلاق والهدف
START_STEP       = 150_000           # base_150k.pt
TARGET_STEP      = 180_000           # +30K خطوة نظيفة
WARMUP_STEPS     = 500               # 1e-6 → 2e-5
LR_PEAK          = 2e-5
LR_WARMUP_START  = 1e-6
BETA_L_GATE      = 0.05              # beta = 0.05 (ثابت من v9.1)
GAMMA_L_KL       = 0.50              # gamma = 0.50 (مُجرّب: خفض L_KL من 38.6 إلى 10.3)
ALPHA_L_BAL      = 0.01              # وزن توازن البوابات (من hepl_telemetry)

# توزيع البيانات (المرجع §6)
DISTILLED_WEIGHT = 0.50              # مرفوع من 30% إلى 50% (التشكيل السلوكي)
CHATML_WEIGHT    = 0.50              # مخفوض من 70% إلى 50% (حفاظ اللغة)

# المعاملات الرمزية لـ Qwen2.5 tokenizer (المرجع §5)
# تُحقَّق فعلياً من الـ tokenizer بعد التحميل
IM_START_ID = 151644
IM_END_ID   = 151645
PAD_ID      = 151643

# المراقبة وبوابة السلامة (المرجع §7, §8)
TELEMETRY_EVERY       = 500          # كل 500 خطوة
SAFETY_GATE_STEP      = 152_000     # بعد أول 2K خطوة
SAFETY_MONITOR_EVERY  = 200          # مراقبة لـ ratio في [150K, 152K]
SAVE_EVERY            = 5_000        # checkpoint كل 5K خطوة
MAX_TIME_MINUTES      = 570          # 9.5 ساعة (كاجل)

# عتبات المراقبة (المرجع §7.1)
ENTROPY_MIN = 0.85
ENTROPY_MAX = 1.45
GINI_MAX    = 0.40
DRIFT_MAX   = 0.15
KL_TARGET_BY_152K = 15.0
# الضبط 2 (المقيم): gamma يُفعّل L_KL عند stage2_step ≥ 2000 = step 152,000.
# عند 152K بالضبط، L_KL في أول خطوة لها (~35–38) ولم تهبط بعد إلى < 15.
# لذلك نخفّض العتبة عند 152K إلى 42.0 (للتأكد من عدم انفجار L_KL عند التفعيل)،
# ونؤجّل الفحص الصارم (< 15.0) إلى 154K (بعد ~2,000 خطوة من التفعيل).
KL_CEILING_AT_152K = 42.0     # عتبة مخفّفة عند 152K (أول تفعيل L_KL)
KL_TARGET_BY_154K = 15.0     # العتبة الصارمة بعد هبوط L_KL
RATIO_MIN, RATIO_MAX = 0.5, 2.0
# التعديل 1 (المقيم): خفض 0.50 → 0.30 (مطابق لـ v9.1 المجربة).
# السبب: بوابة G(x) تعمل بالتوجيه الناعم في البداية ثم تتخصص في الفضاءات الأربعة؛
# ومع تخصص البوابات وتوزيع التدفق على 4 مسارات بدل مسار واحد، ينخفض M_real_norm
# طبيعياً إلى ~0.45–0.55 من البداية. عتبة 0.50 قريبة جداً من هذا النطاق الطبيعي
# وقد تُوقف التدريب كاذباً. 0.30 كان يعمل بلا إنذارات في v9.1.
M_REAL_NORM_MIN_RATIO = 0.30         # مطابق لـ v9.1 (يمنع الإنذار الكاذب عند تخصص البوابات)

# وضعية التدريب — ثابت في v10
TRAINING_STAGE = "stage2_hepl_sft"
USE_MIXED_PRECISION = True
USE_GRADIENT_CHECKPOINTING = True
USE_TORCH_COMPILE = False            # معطّل — قد يفشل على Kaggle

WORKER_ID = f"kaggle-{KAGGLE_USERNAME}"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers أساسية (موروثة من xandra_worker.py الأصلي)
# ──────────────────────────────────────────────────────────────────────────────
def log_step(msg):
    """تسجيل خطوة مع طابع زمني — يُطبع ويُضاف إلى STARTUP_LOG."""
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    STARTUP_LOG.append(line)

def acquire_lock():
    """طلب قفل التدريب من state.json — يمنع تشغيل عاملين في آن واحد.

    التعديل 1 (المقيم، جولة 4): تضارب صيغة locked_by مقابل worker_id.
    المايسترو يُسجّل القفل باسم locked_by = 'maestro-{username}'، والعامل
    يُسجّل باسم worker_id/kaggle-{username}. دون استخراج الاسم المجرد، يظن العامل
    أن القفل محجوز لغيره وينتظر. نستخرج الاسم من كلا البادئتين ونقارن الأسماء المجردة.
    مطابق للمنطق الأصلي في xandra_worker_v7_fixed.py (الأسطر 156-184).
    """
    try:
        state = load_state()
        lock = state.get('training_lock', {})
        if lock.get('is_locked'):
            # نقرأ locked_by (يستخدمه المايسترو) و worker_id (تستخدمه النسخ القديمة)
            locked_by = lock.get('locked_by', '') or lock.get('worker_id', '')
            locked_at = lock.get('locked_at')
            # استخراج الاسم المجرد من البادئتين المعروفتين
            locked_username = locked_by
            for prefix in ('maestro-', 'kaggle-'):
                if locked_by.startswith(prefix):
                    locked_username = locked_by[len(prefix):]
                    break
            my_username = KAGGLE_USERNAME
            for prefix in ('maestro-', 'kaggle-'):
                if my_username.startswith(prefix):
                    my_username = my_username[len(prefix):]
                    break

            if locked_username == my_username and locked_username:
                log_step(f"ℹ️ هذا الحساب ({my_username}) يحمل القفل بالفعل — متابعة")
                # نعيد حجز القفل باسمنا (استئناف)
            else:
                locked_time = datetime.fromisoformat(locked_at.replace('Z', '+00:00')) if locked_at else None
                if locked_time and (datetime.now(timezone.utc) - locked_time).total_seconds() < 12 * 3600:
                    log_step(f"⛔ القفل محجوز بواسطة {locked_by} (username: {locked_username}) منذ {locked_at}")
                    return False
                log_step(f"⚠️ القفل قديم ({locked_at}) — نستولي عليه")
        state['training_lock'] = {
            'is_locked': True,
            'locked_at': datetime.now(timezone.utc).isoformat(),
            'locked_by': WORKER_ID,          # 🆕 البادئة الموحّدة للمايسترو
            'worker_id': WORKER_ID,         # للتوافق مع النسخ القديمة
            'last_heartbeat': datetime.now(timezone.utc).isoformat(),
            'last_step': state.get('current_step', 0),
        }
        save_state(state)
        return True
    except Exception as e:
        log_step(f"⚠️ فشل طلب القفل (غير حرج، نتابع): {e}")
        return True

def release_lock(state_dict, reason="completed"):
    """
    تحرير القفل بعد انتهاء الجلسة.

    فجوة 4: PRESERVE_FIELDS — قبل الحفظ، نجلب latest state من HF لِنحافظ على
    تعديلات لوحة التحكم (account_usage, system_paused, manually_set) التي قد
    حدثت أثناء جلستنا. مطابق للنسخة القديمة (الأسطر 199-227).
    """
    try:
        # 🛡️ Fetch latest state from HF to preserve admin edits
        try:
            latest_path = hf_hub_download(repo_id=HF_REPO, filename="state.json",
                                          repo_type="model", token=HF_TOKEN)
            with open(latest_path) as f:
                latest_state = json.load(f)
            # Preserve admin-edited fields
            PRESERVE_FIELDS = ['account_usage', 'system_paused', 'usage_week_id',
                               'disabled_accounts', 'all_accounts_exhausted_alerted']
            for field in PRESERVE_FIELDS:
                if field in latest_state:
                    state_dict[field] = latest_state[field]
        except Exception as merge_err:
            log_step(f"⚠️ release_lock merge warning (non-fatal): {merge_err}")

        state_dict['training_lock'] = {
            'is_locked': False,
            'locked_at': None,
            'worker_id': None,
            'last_heartbeat': datetime.now(timezone.utc).isoformat(),
            'last_step': state_dict.get('current_step', 0),
            'release_reason': reason,
        }
        save_state(state_dict)
    except Exception as e:
        log_step(f"⚠️ فشل تحرير القفل: {e}")

def heartbeat(state_dict, step):
    """
    نبضة قلب خفيفة — تحدّث state.json دون رفع checkpoint.

    فجوة 4: PRESERVE_FIELDS — قبل الرفع، نجلب latest state من HF لِنحافظ على
    تعديلات لوحة التحكم (account_usage, manually_set) التي قد حدثت أثناء جلستنا.
    مطابق للنسخة القديمة (الأسطر 230-276).
    """
    try:
        if 'training_lock' not in state_dict:
            state_dict['training_lock'] = {}
        state_dict['training_lock']['last_heartbeat'] = datetime.now(timezone.utc).isoformat()
        state_dict['training_lock']['last_step'] = step

        # 🛡️ Fetch latest state from HF to preserve admin edits
        try:
            latest_path = hf_hub_download(repo_id=HF_REPO, filename="state.json",
                                          repo_type="model", token=HF_TOKEN)
            with open(latest_path) as f:
                latest_state = json.load(f)
            # Preserve these fields from the LATEST HF state (admin may have edited them)
            PRESERVE_FIELDS = ['account_usage', 'system_paused', 'usage_week_id',
                               'disabled_accounts', 'all_accounts_exhausted_alerted']
            for field in PRESERVE_FIELDS:
                if field in latest_state:
                    state_dict[field] = latest_state[field]
            # Also preserve manually_set flags inside account_usage
            if 'account_usage' in latest_state:
                for uname, udata in latest_state['account_usage'].items():
                    if udata.get('manually_set'):
                        if 'account_usage' not in state_dict:
                            state_dict['account_usage'] = {}
                        state_dict['account_usage'][uname] = udata
        except Exception as merge_err:
            # Non-fatal: if merge fails, proceed with local state
            log_step(f"⚠️ heartbeat merge warning (non-fatal): {merge_err}")

        save_state(state_dict)
    except Exception as e:
        log_step(f"⚠️ heartbeat خطأ: {e}")

def load_state():
    """تحميل state.json من HuggingFace."""
    try:
        path = hf_hub_download(repo_id=HF_REPO, filename="state.json", repo_type="model", token=HF_TOKEN)
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {
            "current_step": START_STEP, "target_steps": TARGET_STEP,
            "current_loss": None, "history": [], "best_loss": float('inf'),
            "updated_at": None, "gpu_hours_used": 0,
            "training_lock": {"is_locked": False},
        }

def save_state(state_dict):
    """حفظ state.json محلياً ثم رفعه إلى HuggingFace."""
    try:
        local = "state.json"
        with open(local, 'w') as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=2, default=str)
        upload_file(path_or_fileobj=local, path_in_repo="state.json",
                    repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
    except Exception as e:
        log_step(f"⚠️ فشل رفع state.json: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# إعداد الجهاز
# ──────────────────────────────────────────────────────────────────────────────
def setup_device():
    """تحديد جهاز التدريب (GPU متاح؟ كم بطاقة؟)."""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log_step(f"🎮 GPU: {gpu_name} ({vram_gb:.1f} GB) × {n_gpus}")
        return device, n_gpus
    log_step("⚠️ لا GPU متاح — سنتدرب على CPU (بطيء جداً)")
    return torch.device('cpu'), 0

device, N_GPUS = setup_device()

# ──────────────────────────────────────────────────────────────────────────────
# تحميل hepl_architecture.py من المستودع الخاص
# ──────────────────────────────────────────────────────────────────────────────
def load_hepl_architecture():
    """تنزيل hepl_architecture.py من HuggingFace إلى /tmp للاستيراد."""
    log_step("⬇️  تنزيل hepl_architecture.py من المستودع الخاص...")
    arch_path = hf_hub_download(
        repo_id=HEPL_REPO, filename="hepl_architecture.py",
        repo_type="model", token=HF_TOKEN,
    )
    shutil.copy(arch_path, "/tmp/hepl_architecture.py")
    if "/tmp" not in sys.path:
        sys.path.insert(0, "/tmp")
    log_step("✅ hepl_architecture.py جاهز للاستيراد")

# ──────────────────────────────────────────────────────────────────────────────
# تحميل الـ tokenizer (Qwen2.5)
# ──────────────────────────────────────────────────────────────────────────────
def load_tokenizer():
    """تحميل Qwen2.5 tokenizer والتحقق من رموز التحكم."""
    log_step("📖 تحميل Qwen2.5 tokenizer...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", use_fast=True, trust_remote_code=True)
    global IM_START_ID, IM_END_ID, PAD_ID
    IM_START_ID = tok.convert_tokens_to_ids("<|im_start|>")
    IM_END_ID   = tok.convert_tokens_to_ids("<|im_end|>")
    PAD_ID      = tok.pad_token_id if tok.pad_token_id is not None else 151643
    log_step(f"   IM_START={IM_START_ID}, IM_END={IM_END_ID}, PAD={PAD_ID}")
    return tok

# ════════════════════════════════════════════════════════════════════════════
# الفصل 4 (المرجع): SurgicalWeightGrafter + Pre-flight Checks + Optimizer نظيف
# ════════════════════════════════════════════════════════════════════════════

def pre_flight_checks(checkpoint_path, model):
    """
    فحوصات ما قبل الإقلاع (المرجع §4.3).
    يجب أن تُعيد True قبل أي grafting. أي فشل = إيقاف فوري.
    """
    log_step("🔍 تشغيل فحوصات ما قبل الإقلاع...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)

    # 1. توافق حجم المفردات
    ckpt_vocab = state["tok_emb.weight"].shape[0]
    assert ckpt_vocab == model.vocab_size == VOCAB_SIZE, \
        f"❌ vocab mismatch: ckpt={ckpt_vocab}, model={model.vocab_size}, expected={VOCAB_SIZE}"

    # 2. توافق البُعد الخفي
    ckpt_dim = state["tok_emb.weight"].shape[1]
    assert ckpt_dim == model.dim == MODEL_DIM, \
        f"❌ dim mismatch: ckpt={ckpt_dim}, model={model.dim}, expected={MODEL_DIM}"

    # 3. توافق عدد الطبقات
    n_blocks_in_ckpt = sum(
        1 for k in state
        if k.startswith("blocks.") and k.endswith(".norm1.weight")
    )
    assert n_blocks_in_ckpt == model.n_layers == N_LAYERS, \
        f"❌ n_layers mismatch: ckpt={n_blocks_in_ckpt}, model={model.n_layers}"

    # 4. عدد المعاملات في checkpoint المصدر
    ckpt_params = sum(v.numel() for v in state.values())
    log_step(f"   ✅ vocab={ckpt_vocab}, dim={ckpt_dim}, "
             f"n_layers={n_blocks_in_ckpt}, ckpt_params={ckpt_params/1e6:.1f}M")
    if not 100e6 < ckpt_params < 115e6:
        log_step(f"   ⚠️ ckpt_params={ckpt_params/1e6:.1f}M خارج النطاق المتوقع [100M, 115M] "
                 f"(XandraModern ≈ 105.7M) — نتابع بحذر")
    return True


def build_fresh_optimizer(model):
    """
    مُحسّن AdamW نظيف (المرجع §4.4).
    لا وراثة optimizer_state من checkpoint — الزخوم (m, v) تُهيّأ إلى صفر تلقائياً.
    """
    log_step("🧹 تهيئة AdamW نظيف (لا وراثة optimizer_state)...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_WARMUP_START,         # سنبدأ من 1e-6 ثم نتصاعد عبر warmup
        betas=(0.9, 0.95),
        weight_decay=0.01,
        eps=1e-8,
    )
    log_step(f"   ✅ AdamW جاهز: lr={LR_WARMUP_START}, betas=(0.9, 0.95), wd=0.01")
    return optimizer


def warmup_lr_schedule(step):
    """
    جدولة الإحماء + cosine decay (مطابق للنسخة القديمة المُجرّبة).

    النسخة القديمة تستخدم CosineAnnealingLR مع T_max=MAX_STEPS.
    في v10 نُضيف نافذة إحماء خطية (500 خطوة) قبل الـ cosine.

    ملاحظة: step هنا هو المطلق (يبدأ من 150,000). نُحوّله إلى نسبي
    داخل نافذة الإحماء [START_STEP, START_STEP + WARMUP_STEPS].
    بعد الإحماء: cosine decay من LR_PEAK إلى ~0 عبر (TARGET_STEP - START_STEP) خطوة.
    """
    import math
    relative = step - START_STEP           # 0 → 30,000
    if relative < 0:
        return LR_WARMUP_START             # قبل نقطة الانطلاق (لا يجب أن يحدث)
    if relative < WARMUP_STEPS:
        # نافذة الإحماء الخطية
        progress = relative / WARMUP_STEPS
        return LR_WARMUP_START + (LR_PEAK - LR_WARMUP_START) * progress
    # بعد الإحماء: cosine decay (مطابق للنسخة القديمة CosineAnnealingLR)
    total_steps = TARGET_STEP - START_STEP
    cosine_progress = (relative - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
    return LR_PEAK * 0.5 * (1 + math.cos(math.pi * cosine_progress))


def build_scheduler(optimizer):
    """
    بناء جدول الـ LR (مطابق للنسخة القديمة CosineAnnealingLR).
    ملاحظة: في v10 نستخدم warmup_lr_schedule اليدوي (يتحكم في الإحماء + cosine).
    """
    # النسخة القديمة: scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_STEPS)
    # في v10: نُحدّث LR يدوياً في كل خطوة عبر warmup_lr_schedule(step)
    # لذلك نُعيد None هنا — التحديث يحدث داخل حلقة التدريب.
    return None


def apply_surgical_grafting(model):
    """
    تطبيق SurgicalWeightGrafter (المرجع §4): نسخ أوزان XandraModern → XandraHEPL
    مع توزيع SwiGLU على 4 فضاءات + ضوضاء غاوسية (sigma=0.001).
    """
    log_step("🩺 تنزيل base_150k.pt من HuggingFace...")
    base_path = hf_hub_download(
        repo_id=HF_REPO, filename="checkpoints/base_150k.pt",
        repo_type="model", token=HF_TOKEN,
    )
    log_step(f"   ✅ base_150k.pt نُزّل: {os.path.getsize(base_path)/1e6:.0f} MB")

    # فحوصات ما قبل الإقلاع على الـ model الفارغ
    pre_flight_checks(base_path, model)

    # استيراد الدالة الفعلية من hepl_architecture.py
    # ملاحظة: الاسم الفعلي هو graft_from_xandra_modern (وليس graft_from_monolithic)
    from hepl_architecture import graft_from_xandra_modern

    log_step("🩺 تطبيق النسخ الجراحي (sigma=0.001)...")
    model = graft_from_xandra_modern(model, base_path, sigma=0.001, device=str(device))
    log_step("✅ Surgical Grafting مكتمل — الأوزان المُوزّعة جاهزة + الضوضاء الغاوسية مُطبّقة")

    # تحقق إضافي: عدّ المعاملات بعد grafting
    n_params = sum(p.numel() for p in model.parameters())
    log_step(f"   نموذج XandraHEPL بعد grafting: {n_params:,} ({n_params/1e6:.1f}M)")
    expected_range = (165.0e6, 166.5e6)
    if not expected_range[0] <= n_params <= expected_range[1]:
        log_step(f"   ⚠️ n_params={n_params/1e6:.1f}M خارج النطاق المتوقع [165.0M, 166.5M]")
    return model


def resume_from_latest_checkpoint(model, optimizer):
    """
    استئناف التدريب من checkpoints/latest.pt (الفجوة 3: Resuming Bug).
    يُحمّل model_state + optimizer_state من آخر checkpoint.

    returns: (model, optimizer, resumed_step) — resumed_step = step المحفوظ في checkpoint
    """
    log_step("🔄 محاولة استئناف التدريب من checkpoints/latest.pt...")
    try:
        latest_path = hf_hub_download(
            repo_id=HF_REPO, filename="checkpoints/latest.pt",
            repo_type="model", token=HF_TOKEN,
        )
    except Exception as e:
        log_step(f"   ⚠️ لا يوجد checkpoints/latest.pt ({e}) — سنبعد من START_STEP عبر grafting")
        return model, optimizer, 0

    try:
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
    except Exception as e:
        log_step(f"   ⚠️ فشل تحميل latest.pt ({e}) — سنبعد من START_STEP عبر grafting")
        return model, optimizer, 0

    resumed_step = int(ckpt.get('step', 0))
    log_step(f"   📥 latest.pt: step={resumed_step}, loss={ckpt.get('loss', 'n/a')}")

    # تحميل model_state (مع تنظيف بادئات DataParallel/_orig_mod)
    raw_model = get_raw_model(model)
    state_dict = ckpt.get('model_state', {})
    clean_sd = {k.replace("module.", "").replace("_orig_mod.", ""): v
                for k, v in state_dict.items()}
    missing, unexpected = raw_model.load_state_dict(clean_sd, strict=False)
    if missing:
        log_step(f"   ⚠️ مفاتيح مفقودة: {len(missing)} (قد تكون طبقات جديدة)")
    if unexpected:
        log_step(f"   ⚠️ مفاتيح غير متوقعة: {len(unexpected)}")

    # تحميل optimizer_state (مهم لاستئناف الزخوم m/v)
    if 'optimizer_state' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state'])
            log_step("   ✅ optimizer_state مُحمّل (الزخوم m/v مُستعادة)")
        except Exception as e:
            log_step(f"   ⚠️ فشل تحميل optimizer_state ({e}) — AdamW سيبدأ بزخوم صفرية")

    log_step(f"✅ استئناف من step {resumed_step} — التقدم السابق محفوظ")
    return model, optimizer, resumed_step


def build_hepl_model(tokenizer, state=None):
    """
    بناء XandraHEPL ثم تطبيق Surgical Grafting من base_150k.pt (المرجع §3, §4).
    النموذج يُعيد dict في forward() — نتحقق من الحقل routing_shares أو نُضيفه.

    فجوة 3 (Resuming Bug): إن كان state['current_step'] > START_STEP، نستئنف من
    checkpoints/latest.pt بدلاً من graft من base_150k.pt (محافظة على التقدم السابق).
    """
    from hepl_architecture import XandraHEPL

    log_step("🏗️  بناء XandraHEPL (8 طبقات × 4 فضاءات)...")
    model = XandraHEPL(
        vocab_size=VOCAB_SIZE, dim=MODEL_DIM, n_layers=N_LAYERS,
        n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, max_seq_len=MAX_LEN,
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log_step(f"   نموذج XandraHEPL المُهيّأ: {n_params:,} ({n_params/1e6:.1f}M)")

    # فجوة 3: قراءة state['current_step'] قبل graft أم resume
    prev_step = 0
    if state is not None:
        prev_step = int(state.get('current_step', 0))

    if prev_step > START_STEP:
        # استئناف من checkpoint — لا graft (نحافظ على التقدم السابق)
        log_step(f"📍 state['current_step']={prev_step} > START_STEP={START_STEP} — استئناف بدلاً من graft")
        # بناء optimizer مؤقت لتحميل optimizer_state (سيتُجاوز لاحقاً بـ build_fresh_optimizer)
        # لكن نُرجع model فقط هنا — optimizer_state يُحمَّل في main بعد بناء optimizer الحقيقي.
        # ملاحظة: optimizer_state يُحمَّل من latest.pt في main بعد build_fresh_optimizer.
        model = _load_model_state_only(model)
        state['grafting_applied'] = False
        state['resumed_from'] = prev_step
    else:
        # graft من base_150k.pt — أول جلسة أو جلسة قبل START_STEP
        log_step(f"📍 state['current_step']={prev_step} ≤ START_STEP={START_STEP} — graft من base_150k.pt")
        model = apply_surgical_grafting(model)
        state['grafting_applied'] = True

    # ملاحظة: الحقل routing_shares قد يكون مُضافاً بالفعل إلى forward() dict في نسختك
    # من hepl_architecture.py. إن لم يكن، نُسجّل تحذيراً هنا (يجب إضافته يدوياً).
    try:
        with torch.no_grad():
            dummy = torch.randint(0, VOCAB_SIZE, (1, 16), device=device)
            out = model(dummy, step=0)
        if isinstance(out, dict) and "routing_shares" in out:
            log_step("✅ forward() dict يحتوي على routing_shares — التجميع عبر DataParallel جاهز")
        else:
            log_step("⚠️ forward() dict لا يحتوي على routing_shares — راجع المرجع §3.2 لإضافته")
    except Exception as e:
        log_step(f"⚠️ فشل التحقق من forward() (غير حرج): {e}")

    return model


def _load_model_state_only(model):
    """تحميل model_state فقط من checkpoints/latest.pt (بدون optimizer_state)."""
    try:
        latest_path = hf_hub_download(
            repo_id=HF_REPO, filename="checkpoints/latest.pt",
            repo_type="model", token=HF_TOKEN,
        )
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
    except Exception as e:
        log_step(f"⚠️ فشل تحميل latest.pt ({e}) — العودة لـ grafting")
        return apply_surgical_grafting(model)

    raw_model = get_raw_model(model)
    state_dict = ckpt.get('model_state', {})
    clean_sd = {k.replace("module.", "").replace("_orig_mod.", ""): v
                for k, v in state_dict.items()}
    missing, unexpected = raw_model.load_state_dict(clean_sd, strict=False)
    log_step(f"   📥 model_state مُحمّل (missing={len(missing)}, unexpected={len(unexpected)})")
    log_step(f"   ✅ استئناف من step {ckpt.get('step', '?')} — التقدم السابق محفوظ")
    # نُخزّن latest.pt path مؤقتاً في model لاستخدامه في build_fresh_optimizer
    raw_model._latest_ckpt_for_opt = latest_path
    return model


def _load_optimizer_state(optimizer, raw_model):
    """تحميل optimizer_state من latest.pt (مهم لاستئناف زخوم AdamW — الفجوة 3)."""
    if not hasattr(raw_model, '_latest_ckpt_for_opt'):
        log_step("   ℹ️ لا يوجد _latest_ckpt_for_opt — AdamW سيبدأ بزخوم صفرية")
        return
    try:
        ckpt_path = raw_model._latest_ckpt_for_opt
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
            log_step("   ✅ optimizer_state مُحمّل (الزخوم m/v مُستعادة من latest.pt)")
        else:
            log_step("   ⚠️ latest.pt لا يحوي optimizer_state — AdamW يبدأ بزخوم صفرية")
    except Exception as e:
        log_step(f"   ⚠️ فشل تحميل optimizer_state ({e}) — AdamW يبدأ بزخوم صفرية")
    finally:
        # تنظيف
        if hasattr(raw_model, '_latest_ckpt_for_opt'):
            delattr(raw_model, '_latest_ckpt_for_opt')


# ════════════════════════════════════════════════════════════════════════════
# تحميل البيانات: Distilled Stream + ChatML Stream
# ════════════════════════════════════════════════════════════════════════════

# ملاحظة بعد المقارنة بالنسخة القديمة:
# ─────────────────────────────────────
# منطق تحميل البيانات في النسخة القديمة يعتمد على:
#   1) load_distilled_records() — يجمع سجلات التقطير من smart-force/xandra-sft-corpus
#      (distilled/*.parquet) ويصنّفها حسب target_manifold (0=real, 1=intu, 2=imag, 3=rat)
#   2) refill_text_buffer() — يسحب دفعات من تيار ChatML العام (combined_stream)
#   3) get_batch() — يجمع بين المصدرين بنسبة 50/50 في مستوى batch selection
#      (وليس في مستوى loss weights كما افترضنا خطأً في v10 الأولي)
#
# في v10، نُورّث هذا المنطق من النسخة القديمة (Strategy A/B)، لأنه جوهري
# لتفعيل L_gate و L_KL على الفضاءات الأربعة بشكل متوازن.

# متغيرات عالمية (مطابقة للنسخة القديمة)
DISTILLED_BUFFER = []
MANIFOLD_INDEX = {0: [], 1: [], 2: [], 3: []}  # per-manifold index for balanced sampling
DISTILLED_BAD_COUNT = 0
DISTILLED_SAMPLING_WEIGHTS = None  # set only if min manifold < 30 records
TEXT_BUFFER = []
BUFFER_SIZE = 2500
STREAM_ITER = None
STREAMING_READY = False
STREAM_BAD_COUNT = 0
all_texts = []           # local texts fallback (from local sources if any)
combined_stream = None   # interleave of streaming datasets

# عتبات فحص جودة البيانات (مطابقة للنسخة القديمة)
_BAD_PREFIXES = ("[DETERMINISTIC]", "[INTUITIVE]", "[IMAGINATIVE]", "[RATIONAL]",
                 "[REAL]", "[INTU]", "[IMAG]", "[RAT]",
                 "DETERMINISTIC:", "INTUITIVE:", "IMAGINATIVE:", "RATIONAL:")
_NO_QUESTION_MARKERS = ("لم تقدم سؤال", "لم تطرح سؤال", "بالطبع، ولكن يبدو أنك لم",
                        "يبدو أنك لم تقدم", "لم تقدم أي سؤال", "لم تذكر أي سؤال")
# SAFE-RETRAIN v9.2: OCR-glued Arabic words filter (مطابق للنسخة القديمة)
OCR_GLUE_MAX_WORD_LEN = 12      # words longer than this are "glued"
OCR_GLUE_MAX_RATIO = 0.15      # if glued words > 15% of all words, reject the row
_AR_PUNCT = ".,،؛:!?()[]{}«»\"'\u060c\u061b\u061f\u066b\u066a "


def setup_combined_stream():
    """
    تهيئة تيار ChatML العام — مطابق للنسخة القديمة (الأسطر 801-875).
    يجمع عدة مصادر streaming عبر interleave_datasets مع أوزان.
    في Stage2: يحمّل من smart-force/xandra-sft-corpus/data/*.parquet
    """
    global combined_stream, STREAMING_READY
    if TRAINING_STAGE != "stage2_hepl_sft":
        STREAMING_READY = False
        return
    sources_config = {
        "sources": [
            {"name": "Xandra-SFT-Corpus", "type": "stream",
             "path": "smart-force/xandra-sft-corpus",
             "data_files": "data/*.parquet",
             "split": "train", "field": "text", "weight": 1.0}
        ]
    }
    try:
        from datasets import load_dataset, interleave_datasets
    except ImportError as e:
        log_step(f"⚠️ مكتبة datasets غير متاحة: {e} — STREAMING_READY=False")
        STREAMING_READY = False
        return
    stream_datasets = []
    stream_weights = []
    for source in sources_config["sources"]:
        try:
            log_step(f"🌐 تحميل بيانات Streaming: {source['name']}")
            if "config" in source:
                ds = load_dataset(source["path"], source["config"],
                                  data_files=source.get("data_files"),
                                  split=source["split"], streaming=True)
            else:
                ds = load_dataset(source["path"],
                                  data_files=source.get("data_files"),
                                  split=source["split"], streaming=True)
            stream_datasets.append(ds)
            stream_weights.append(source["weight"])
        except Exception as e:
            log_step(f"⚠️ فشل تحميل {source['name']}: {e}")
    if len(stream_datasets) > 0:
        total_w = sum(stream_weights)
        stream_weights = [w / total_w for w in stream_weights]
        combined_stream = interleave_datasets(stream_datasets,
                                              probabilities=stream_weights, seed=42)
        STREAMING_READY = True
        log_step(f"✅ تيار Streaming جاهز: {len(stream_datasets)} مصدر")
    else:
        STREAMING_READY = False
        log_step("⚠️ لا توجد مصادر Streaming متاحة")


def refill_text_buffer():
    """
    يسحب دفعة جديدة من تيار ChatML ويملأ TEXT_BUFFER.
    مطابق للنسخة القديمة (الأسطر 1063-1117).

    SAFE-RETRAIN: يحتوي على schema fallback (prompt+cot_response) لنفس سبب
    load_distilled_records — data/*.parquet من distill_sft.py بدون عمود 'text'.
    """
    global TEXT_BUFFER, STREAM_ITER, STREAMING_READY
    if not STREAMING_READY:
        return
    try:
        if STREAM_ITER is None:
            shuffled = combined_stream.shuffle(seed=int(time.time()), buffer_size=5000)
            STREAM_ITER = iter(shuffled)

        TEXT_BUFFER.clear()
        count = 0
        _attempts = 0
        _max_attempts = BUFFER_SIZE * 20  # safety valve
        while count < BUFFER_SIZE and _attempts < _max_attempts:
            _attempts += 1
            try:
                item = next(STREAM_ITER)
                # schema fallback: prompt+cot_response عند غياب 'text'
                text = item.get('text') or item.get('content') or item.get('nass')
                if not text:
                    _p = str(item.get('prompt', '') or item.get('question', '') or '').strip()
                    _c = str(item.get('cot_response', '') or item.get('response', '')
                              or item.get('answer', '') or '').strip()
                    if _p or _c:
                        text = (_p + "\n" + _c).strip()
                if text and len(str(text)) > 50:
                    if _is_bad_stream_row(text):
                        continue
                    TEXT_BUFFER.append(str(text))
                    count += 1
            except StopIteration:
                shuffled = combined_stream.shuffle(seed=int(time.time()), buffer_size=5000)
                STREAM_ITER = iter(shuffled)
        if count > 0:
            extra = f" | 🚫 تخطّي {STREAM_BAD_COUNT} سجل OCR ملتصق" if STREAM_BAD_COUNT > 0 else ""
            log_step(f"📦 تم تعبئة buffer بـ {count} نص جديد من الـ stream{extra}")
        else:
            log_step(f"⚠️ تعذّر تعبئة buffer بعد {_attempts} محاولة")
            try:
                _probe = next(iter(combined_stream.shuffle(seed=42, buffer_size=10)))
                log_step(f"   🔍 أعمدة data/*.parquet الفعلية: {list(_probe.keys())}")
            except Exception as _pe:
                log_step(f"   🔍 تعذّر استكشاف schema: {_pe}")
            STREAMING_READY = False
    except Exception as e:
        log_step(f"⚠️ فشل تعبئة buffer: {e}")
        STREAM_ITER = None
        STREAMING_READY = False


def _is_bad_distilled_row(text):
    """Returns True if the row is malformed. مطابق للنسخة القديمة."""
    if text is None:
        return True
    try:
        if isinstance(text, float) and math.isnan(text):
            return True
    except TypeError:
        pass
    s = str(text).strip()
    if len(s) < 60:
        return True
    for p in _BAD_PREFIXES:
        if s.startswith(p):
            return True
    head = s[:300]
    for m in _NO_QUESTION_MARKERS:
        if m in head:
            return True
    return False


def _is_bad_stream_row(text):
    """تصفية صفوف الـ stream الملتصقة (OCR glue). مطابق للنسخة القديمة."""
    global STREAM_BAD_COUNT
    if text is None:
        return True
    try:
        if isinstance(text, float) and math.isnan(text):
            return True
    except TypeError:
        pass
    s = str(text).strip()
    if len(s) < 60:
        return True
    if s.startswith("<|im_start|>") and "assistant" not in s[:30].lower():
        return True
    words = [w for w in s.translate(str.maketrans(_AR_PUNCT, " " * len(_AR_PUNCT))).split() if w]
    if len(words) < 4:
        return True
    glued = sum(1 for w in words if len(w) > OCR_GLUE_MAX_WORD_LEN)
    if glued / len(words) > OCR_GLUE_MAX_RATIO:
        STREAM_BAD_COUNT += 1
        return True
    return False


def load_distilled_records():
    """
    يسحب سجلات التقطير المعرفي من مجلد distilled/ بشكل مستقل.
    مطابق للنسخة القديمة — يصنّف السجلات حسب target_manifold (0..3) ويبني
    MANIFOLD_INDEX للاستعيانة المتوازنة (Strategy A) أو inverse-frequency (Strategy B).
    """
    global DISTILLED_BUFFER, MANIFOLD_INDEX, DISTILLED_BAD_COUNT, DISTILLED_SAMPLING_WEIGHTS
    if DISTILLED_BUFFER or TRAINING_STAGE != "stage2_hepl_sft":
        return
    try:
        import pandas as pd
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        files = api.list_repo_files(repo_id="smart-force/xandra-sft-corpus", repo_type="dataset")
        parquet_files = sorted([f for f in files if f.startswith("distilled/") and f.endswith(".parquet")])
        log_step(f"   📥 سيتم تحميل {len(parquet_files)} ملف parquet من distilled/")
        for pf in parquet_files:
            local_pq = hf_hub_download(
                repo_id="smart-force/xandra-sft-corpus",
                filename=pf, repo_type="dataset", token=HF_TOKEN,
            )
            try:
                df_pq = pd.read_parquet(local_pq)
            except Exception as _pq_err:
                log_step(f"   ⚠️ تخطّي ملف تالف {pf}: {type(_pq_err).__name__}")
                continue
            for _, row in df_pq.iterrows():
                # schema fallback: distill_sft.py يكتب prompt+cot_response بدون 'text'
                txt = str(row.get("text", "")).strip()
                if not txt:
                    _p = str(row.get("prompt", "")).strip()
                    _c = str(row.get("cot_response", "")).strip()
                    if _p or _c:
                        txt = (_p + "\n" + _c).strip()
                txt = txt.strip()
                if len(txt) < 30:
                    continue
                if _is_bad_distilled_row(txt):
                    DISTILLED_BAD_COUNT += 1
                    continue
                try:
                    t_ids = json.loads(row["teacher_top32_ids"]) if isinstance(row.get("teacher_top32_ids"), str) else None
                    t_lps = json.loads(row["teacher_top32_logprobs"]) if isinstance(row.get("teacher_top32_logprobs"), str) else None
                    manifold = int(row.get("target_manifold", -1))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
                if t_ids is None or t_lps is None or manifold not in (0, 1, 2, 3):
                    continue
                idx = len(DISTILLED_BUFFER)
                DISTILLED_BUFFER.append({
                    "text": txt,
                    "target_manifold": manifold,
                    "teacher_top32_ids": t_ids,
                    "teacher_top32_logprobs": t_lps,
                })
                MANIFOLD_INDEX[manifold].append(idx)
        total = len(DISTILLED_BUFFER)
        log_step(f"   ✅ تم تحميل {total} سجل تقطير (تخطّى {DISTILLED_BAD_COUNT} سجل تالف)")
        if total > 0:
            raw_dist = {m: len(MANIFOLD_INDEX[m]) for m in (0, 1, 2, 3)}
            min_m = min(raw_dist.values())
            if min_m < 30:
                weights = []
                for rec in DISTILLED_BUFFER:
                    m = rec["target_manifold"]
                    if m in (0, 1, 2, 3) and raw_dist[m] > 0:
                        weights.append(1.0 / raw_dist[m])
                    else:
                        weights.append(0.0)
                w_sum = sum(weights)
                if w_sum > 0:
                    DISTILLED_SAMPLING_WEIGHTS = [w / w_sum for w in weights]
                    log_step("   ✅ تم بناء أوزان استعيانة عكسية التكرار (Strategy B)")
            else:
                log_step("   ✅ جميع الفضاءات ≥ 30 سجل — استعيانة متوازنة 25%/25%/25%/25% (Strategy A)")
            for m in (0, 1, 2, 3):
                labels = {0: "real", 1: "intu", 2: "imag", 3: "rat"}
                cnt = len(MANIFOLD_INDEX[m])
                pct = (cnt / total * 100) if total else 0
                log_step(f"     • {labels[m]:5s}: {cnt:4d} ({pct:5.1f}%)")
    except Exception as e:
        log_step(f"⚠️ تعذر تحميل سجلات التقطير: {e} — التدريب يستمر بالبيانات الأساسية")


def get_batch(batch_size=BATCH_SIZE, max_len=MAX_LEN, tokenizer=None):
    """
    يُعيد قاموساً آمناً يحتوي على التوكنات وأعمدة التقطير عند توفرها.
    مطابق للنسخة القديمة — يطبّق التوزيع 50/50 في مستوى batch selection.

    50% من الدفعات تُسحب من DISTILLED_BUFFER (لتفعيل L_gate و L_KL)،
    50% تُسحب من TEXT_BUFFER (ChatML stream) أو all_texts.

    داخل DISTILLED_BUFFER، يستخدم:
      Strategy A: 25%/25%/25%/25% متوازن (إن كان كل فضاء ≥ 30 سجل)
      Strategy B: inverse-frequency (fallback عند ضعف فضاء)
    """
    if (TRAINING_STAGE == "stage2_hepl_sft"
            and len(DISTILLED_BUFFER) >= batch_size
            and tokenizer is not None
            and torch.rand(1).item() < 0.50):
        # Strategy B: inverse-frequency weighted sampling
        idxs = []
        if DISTILLED_SAMPLING_WEIGHTS is not None:
            idxs = random.choices(
                range(len(DISTILLED_BUFFER)),
                weights=DISTILLED_SAMPLING_WEIGHTS,
                k=batch_size
            )
        else:
            # Strategy A: pick manifold uniformly, then pick record within manifold
            for _ in range(batch_size):
                chosen_manifold = random.randint(0, 3)
                manifold_pool = MANIFOLD_INDEX[chosen_manifold]
                if len(manifold_pool) == 0:
                    valid_manifolds = [m for m in (0, 1, 2, 3) if len(MANIFOLD_INDEX[m]) > 0]
                    if not valid_manifolds:
                        break
                    chosen_manifold = random.choice(valid_manifolds)
                    manifold_pool = MANIFOLD_INDEX[chosen_manifold]
                idxs.append(random.choice(manifold_pool))
        if len(idxs) >= batch_size:
            samples = [DISTILLED_BUFFER[k] for k in idxs]
            input_list, man_list, ids_list, lps_list, mask_list = [], [], [], [], []
            has_distill = True
            for s in samples:
                toks = tokenizer(s["text"], return_tensors='pt',
                                 padding='max_length', truncation=True,
                                 max_length=max_len)['input_ids'][0].clamp(0, VOCAB_SIZE - 1)
                input_list.append(toks)
                if (s["target_manifold"] is not None
                        and s["teacher_top32_ids"] is not None
                        and s["teacher_top32_logprobs"] is not None):
                    man_list.append(s["target_manifold"])
                    raw_ids = s["teacher_top32_ids"][:max_len]
                    raw_lps = s["teacher_top32_logprobs"][:max_len]
                    valid_len = len(raw_ids)
                    pad_len = max_len - valid_len
                    if pad_len > 0:
                        raw_ids = raw_ids + [[0] * 32 for _ in range(pad_len)]
                        raw_lps = raw_lps + [[-100.0] * 32 for _ in range(pad_len)]
                    ids_list.append(raw_ids)
                    lps_list.append(raw_lps)
                    # الخطأ 3 (المقيم): kl_mask = "أين الـ padding؟" (لـ L_KL فقط)
                    #   m_row = [1.0]*valid_len + [0.0]*pad_len
                    # response_mask (لـ compute_response_only_loss) يُبنى دائماً
                    # عبر build_response_mask(batch) في الحلقة — لا يُمرّر هنا.
                    m_row = [1.0] * valid_len + [0.0] * pad_len
                    mask_list.append(m_row)
                else:
                    has_distill = False
            if has_distill:
                return {
                    "input_ids": torch.stack(input_list),
                    "target_manifold": torch.tensor(man_list, dtype=torch.long),
                    "teacher_top32_ids": torch.tensor(ids_list, dtype=torch.long),
                    "teacher_top32_logprobs": torch.tensor(lps_list, dtype=torch.float32),
                    "kl_mask": torch.tensor(mask_list, dtype=torch.float32)
                }

    # ─── Fallback: ChatML stream / all_texts (مطابق للنسخة القديمة الأسطر 1184-1214) ───
    texts = []
    # 70% من الدفعات غير المقطّرة تُسحب من TEXT_BUFFER (streaming)
    use_stream = STREAMING_READY and len(TEXT_BUFFER) > 0 and torch.rand(1).item() < 0.7
    if use_stream:
        indices = torch.randint(0, len(TEXT_BUFFER), (batch_size,))
        for idx in indices:
            texts.append(TEXT_BUFFER[idx])

    # الباقي يُسحب من all_texts (fallback محلي إن وُجد)
    if len(texts) < batch_size and len(all_texts) > 0:
        needed = batch_size - len(texts)
        indices = torch.randint(0, len(all_texts), (needed,))
        for idx in indices:
            texts.append(all_texts[idx])

    # إعادة تعبئة buffer عند انخفاضه
    if STREAMING_READY and len(TEXT_BUFFER) < 100:
        refill_text_buffer()

    # انتظار تعبئة البيانات إن كانت فارغة
    while len(texts) == 0:
        log_step("⏳ انتظار تعبئة البيانات من المستودعات السحابية...")
        refill_text_buffer()
        if len(TEXT_BUFFER) > 0:
            indices = torch.randint(0, len(TEXT_BUFFER), (batch_size,))
            for idx in indices:
                texts.append(TEXT_BUFFER[idx])
        else:
            time.sleep(10)

    batch = []
    for text in texts:
        tokens = tokenizer(text, return_tensors='pt',
                           padding='max_length', truncation=True,
                           max_length=max_len)['input_ids'][0].clamp(0, VOCAB_SIZE - 1)
        batch.append(tokens)
    return {"input_ids": torch.stack(batch)}


# ════════════════════════════════════════════════════════════════════════════
# الفصل 5 (المرجع §5 — القرار A3): Response-Only Loss Masking
# ════════════════════════════════════════════════════════════════════════════

# ملاحظة بعد تقييم المقيم:
# ─────────────────────────────────────────────
# المرجع الهندسي المعتمد [source: 0.1.56] القرار A3 يُلزم بـ Response-Only Loss
# Masking على L_sft — وهذا كان السبب الرئيسي في فساد الردود السابقة (التدريب
# على السؤال + الإجابة + الحشو يجعل النموذج يحفظ السؤال أيضاً).
#
# لذلك v10 يُطبّق Response-Only Loss Masking الحقيقي:
#   1) build_response_mask() يبني قناع "أين الإجابة؟" (بعد <|im_start|>assistant\n)
#   2) compute_response_only_loss() يُعيد حساب L_sft على الإجابة فقط (مع <|im_end|>)
#   3) l_sft من forward() يُستخدم فقط في المراقبة، لكن L_sft الفعلي في معادلة
#      الخسارة هو compute_response_only_loss()
#
# معادلة الخسارة المُجرّبة مع Response-Only:
#   loss = l_sft_resp + (alpha_w * l_bal) + (beta * l_gate) + (gamma * l_kl)
# حيث l_sft_resp = compute_response_only_loss(logits, labels, response_mask)

# نُخزّن نمط tokens الـ header "<|im_start|>assistant\n" بعد تحميل الـ tokenizer
_ASSISTANT_HEADER_IDS = None

def init_assistant_header_pattern(tokenizer):
    """تحضير نمط tokens الـ header "<|im_start|>assistant\\n" — يُستدعى مرة واحدة.
    مطابق للقرار A3 من المرجع."""
    global _ASSISTANT_HEADER_IDS
    _ASSISTANT_HEADER_IDS = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    log_step(f"   نمط assistant header = {_ASSISTANT_HEADER_IDS}")


def build_response_mask(input_ids, attention_mask=None):
    """
    بناء قناع الإجابة (المرجع §5.2 — القرار A3).

    input_ids:       [B, T]  — معرّفات التوكنات
    attention_mask:  [B, T]  — قناع الانتباه (1 = token صالح، 0 = padding)

    returns: response_mask [B, T] — bool فقط على:
      - محتوى الإجابة الفعلي (بعد "<|im_start|>assistant\\n")
      - رمز النهاية <|im_end|>   ← مهم! يُبقى ضمن القناع
    يستبعد:
      - <|im_start|>system...، <|im_start|>user...
      - <|im_start|>assistant\n  (الـ header نفسه)
      - PAD_ID
    """
    B, T = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    header_ids = _ASSISTANT_HEADER_IDS
    if header_ids is None:
        # fallback آمن: قناع "أين الـ padding؟" فقط
        if attention_mask is not None:
            return attention_mask.bool()
        return (input_ids != PAD_ID)
    header_len = len(header_ids)

    if attention_mask is None:
        attention_mask = (input_ids != PAD_ID)

    for b in range(B):
        ids = input_ids[b].tolist()
        in_response = False
        t = 0
        while t < T:
            if (not in_response) and ids[t] == IM_START_ID:
                window = ids[t : t + header_len]
                if window == header_ids:
                    # دخلنا في قسم الإجابة — ابدأ القناع بعد الـ header
                    in_response = True
                    t = t + header_len
                    continue
            if in_response:
                mask[b, t] = True
                if ids[t] == IM_END_ID:
                    # <|im_end|> مُضمَّن (مهم!)، ثم نوقف القناع
                    in_response = False
            t += 1

    # استبعاد PAD_ID نهائياً (حتى لو ظهر ضمن نافذة الإجابة)
    final_mask = mask & (input_ids != PAD_ID) & attention_mask.bool()
    return final_mask


def compute_response_only_loss(logits, labels, response_mask):
    """
    حساب CE loss على الإجابة فقط (المرجع §5.3 — القرار A3).
    يطبّق shift القياسي (logits[:-1] ↔ labels[1:]) ثم يُقنّع.

    logits:         [B, T, V]
    labels:         [B, T]   — نفس input_ids (التدريب التلقائي الناتج)
    response_mask:  [B, T]   — bool (من build_response_mask)

    returns: scalar loss (mean over response tokens)
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask   = response_mask[..., 1:].contiguous()

    # log-softmax للأرقام مستقرة (fp32 للاستقرار العددي)
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    gathered = log_probs.gather(-1, shift_labels.unsqueeze(-1).long()).squeeze(-1)

    masked = gathered * shift_mask.float()
    n_tokens = shift_mask.float().sum().clamp(min=1.0)
    return -masked.sum() / n_tokens


# ════════════════════════════════════════════════════════════════════════════
# الفصل 7 (المرجع): المراقبة الثلاثية (Entropy / Gini / Drift)
# ════════════════════════════════════════════════════════════════════════════

def compute_gate_metrics(routing_shares):
    """
    حساب المؤشرات الثلاثية على routing_shares المُجمَّع (المرجع §7).

    routing_shares: tensor [4] (بعد gather + mean)
    الفهرس: 0=M_real, 1=M_intu, 2=M_imag, 3=M_rat

    returns: dict {entropy, gini, drift}
    """
    s = routing_shares.detach().cpu().tolist()

    # 1. الإنتروبيا — كشف الانهيار أو العشوائية
    entropy = -sum(p * math.log(p + 1e-8) for p in s)
    # مثالي: 1.386 (= ln 4 — توزيع متوازن)

    # 2. معامل Gini — كشف تركّز التوجيه
    # الصيغة القياسية على n قيم مرتّبة تصاعدياً:
    #   Gini = (2·Σ((i+1)·x_i) - (n+1)·Σx_i) / (n·Σx_i)
    # مثالي: 0.0 (متوازن) — تحذير عند > 0.4 (تركّز خطير) — أقصى قيمة: (n-1)/n ≈ 0.75
    sorted_s = sorted(s)   # تصاعدياً (الأصغر أولاً) — يعطي gini موجب للتركّز
    n = len(sorted_s)
    total = sum(sorted_s)
    if total < 1e-8:
        gini = 0.0
    else:
        weighted_sum = sum((i + 1) * v for i, v in enumerate(sorted_s))
        gini = (2 * weighted_sum - (n + 1) * total) / (n * total)
    # مثالي: 0.0 (متوازن) — تحذير عند > 0.4 (تركّز خطير)

    # 3. الانحراف عن المرجع المتوازن
    target = torch.tensor([0.25, 0.25, 0.25, 0.25])
    drift = float((routing_shares.detach().cpu() - target).abs().mean())
    # مثالي: 0.0 — تحذير عند > 0.15

    return {
        "entropy": round(entropy, 4),
        "gini":    round(gini, 4),
        "drift":   round(drift, 4),
        "shares":  {k: round(v, 4) for k, v in zip(
            ["M_real", "M_intu", "M_imag", "M_rat"], s
        )},
    }


def check_safety_violation(metrics, step):
    """
    التحقق من عتبات المراقبة (المرجع §7.1).
    يُعيد (violated, message). إن انتُهكت أي عتبة، يجب التوقف الفوري.
    """
    violations = []
    if not (ENTROPY_MIN <= metrics["entropy"] <= ENTROPY_MAX):
        violations.append(
            f"entropy={metrics['entropy']:.4f} خارج [{ENTROPY_MIN}, {ENTROPY_MAX}]"
        )
    if metrics["gini"] >= GINI_MAX:
        violations.append(f"gini={metrics['gini']:.4f} ≥ {GINI_MAX} (تركّز خطير)")
    if metrics["drift"] >= DRIFT_MAX:
        violations.append(f"drift={metrics['drift']:.4f} ≥ {DRIFT_MAX} (انحراف خطير)")
    if violations:
        return True, f"step {step}: " + " | ".join(violations)
    return False, ""


# ════════════════════════════════════════════════════════════════════════════
# الفصل 8 (المرجع): بوابة السلامة عند step 152,000
# ════════════════════════════════════════════════════════════════════════════

def safety_gate_152k(model, optimizer, telemetry_log, loss_log, m_real_baseline, current_m_real=None):
    """
    بوابة القرار الإلزامية عند step 152,000 (المرجع §8).
    تتحقق من ستة شروط قبل السماح بالاستمرار إلى 180K.

    الخطأ 2 (المقيم): لا نستدعي model.fluency_probe() — فهي تتطلب idx
    (model.fluency_probe(batch)) ولا تعمل بلا معاملات، وستنهار عند DataParallel.
    بدلاً منه نستخدم current_m_real المحسوبة بالفعل من out_dict['m_real_norm']
    في حلقة التدريب ونمرّرها هنا كمعامل.
    """
    # fallback: لو لم يُمرَّر current_m_real، نأخذ آخر قيمة من telemetry_log إن وُجدت
    if current_m_real is None:
        if telemetry_log and "m_real_norm" in telemetry_log[-1]:
            current_m_real = telemetry_log[-1]["m_real_norm"]
        else:
            current_m_real = 0.0
            log_step("⚠️ current_m_real غير متاح — استخدام 0.0 (قد يُفشل البوابة)")

    checks = {
        "loss_below_5":           loss_log[-1].get("loss", 999) < 5.0,
        "entropy_in_range":       ENTROPY_MIN <= telemetry_log[-1]["entropy"] <= ENTROPY_MAX,
        "gini_below_0.4":         telemetry_log[-1]["gini"] < GINI_MAX,
        "drift_below_0.15":       telemetry_log[-1]["drift"] < DRIFT_MAX,
        # الضبط 2 (المقيم): عند 152K، L_KL في أول خطوة لها (~35–38) — نخفّض العتبة إلى 42.
        # الفحص الصارم (< 15.0) يُؤجّل إلى 154K (انظر safety_gate_154k أسفل).
        "kl_below_ceiling":      loss_log[-1].get("l_kl", 999) < KL_CEILING_AT_152K,
        "m_real_norm_healthy":   current_m_real > M_REAL_NORM_MIN_RATIO * max(m_real_baseline, 1e-8),
    }
    all_pass = all(checks.values())

    log_step("=" * 60)
    log_step(f"🛡️  SAFETY GATE @ step 152,000 — {'✅ PASS' if all_pass else '❌ FAIL'}")
    for k, v in checks.items():
        log_step(f"   [{'✓' if v else '✗'}] {k}")
    log_step(f"   (current_m_real={current_m_real:.6f}, baseline={m_real_baseline:.6f}, "
             f"threshold={M_REAL_NORM_MIN_RATIO * max(m_real_baseline, 1e-8):.6f})")
    log_step("=" * 60)

    if all_pass:
        # checkpoint خاص قبل الاستمرار
        try:
            raw_model = model.module if hasattr(model, 'module') else model
            raw_model = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model
            torch.save({
                "model_state": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": SAFETY_GATE_STEP,
                "tag": "safety_gate_pass",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, "safety_gate_152k.pt")
            upload_file(path_or_fileobj="safety_gate_152k.pt",
                        path_in_repo="checkpoints/safety_gate_152k.pt",
                        repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
            log_step("✅ تم رفع checkpoints/safety_gate_152k.pt")
            # التعديل 2 (المقيم): حذف safety_gate_152k.pt محلياً بعد الرفع
            if os.path.exists("safety_gate_152k.pt"):
                try:
                    os.remove("safety_gate_152k.pt")
                except OSError:
                    pass
        except Exception as e:
            log_step(f"⚠️ فشل رفع safety_gate checkpoint (غير حرج): {e}")
    return all_pass


def safety_gate_154k(model, optimizer, telemetry_log, loss_log, m_real_baseline,
                     current_m_real=None, current_m_real_norm=None):
    """
    البوابة الصارمة الثانية عند step 154,000 (المرجع §8 + الضبط 2 من المقيم).
    تتحقق من هبوط L_KL إلى < 15.0 بعد ~2,000 خطوة من تفعيل gamma (step 152,000).

    لماذا 154K؟ لأن gamma = GAMMA_L_KL if stage2_step >= 2000 else 0.0
    يُفعّل L_KL لأول مرة عند step 152,000، وقيمتها الأولى ~35–38،
    وتصل إلى < 15.0 بعد ~1,500 خطوة (step 153,500). فحص 154K يضمن هبوطها.
    """
    # fallback لـ current_m_real إن لم يُمرّر
    if current_m_real is None:
        if telemetry_log and "m_real_norm" in telemetry_log[-1]:
            current_m_real = telemetry_log[-1]["m_real_norm"]
        else:
            current_m_real = 0.0

    # m_real_norm عبر DataParallel (current_m_real_norm المُجمّع) إن أُمرّر، وإلا fallback
    m_real_for_check = current_m_real_norm if current_m_real_norm is not None else current_m_real

    checks = {
        "loss_below_5":           loss_log[-1].get("loss", 999) < 5.0,
        "entropy_in_range":       ENTROPY_MIN <= telemetry_log[-1]["entropy"] <= ENTROPY_MAX,
        "gini_below_0.4":         telemetry_log[-1]["gini"] < GINI_MAX,
        "drift_below_0.15":       telemetry_log[-1]["drift"] < DRIFT_MAX,
        # الفحص الصارم: L_KL يجب أن يكون قد هبط إلى < 15.0 الآن
        "kl_below_15":            loss_log[-1].get("l_kl", 999) < KL_TARGET_BY_154K,
        "m_real_norm_healthy":    m_real_for_check > M_REAL_NORM_MIN_RATIO * max(m_real_baseline, 1e-8),
    }
    all_pass = all(checks.values())

    log_step("=" * 60)
    log_step(f"🛡️  SAFETY GATE @ step 154,000 (L_KL strict) — {'✅ PASS' if all_pass else '❌ FAIL'}")
    for k, v in checks.items():
        log_step(f"   [{'✓' if v else '✗'}] {k}")
    log_step(f"   (l_kl={loss_log[-1].get('l_kl', 999):.4f}, target<{KL_TARGET_BY_154K}; "
             f"m_real={m_real_for_check:.6f}, baseline={m_real_baseline:.6f})")
    log_step("=" * 60)

    if all_pass:
        try:
            raw_model = model.module if hasattr(model, 'module') else model
            raw_model = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model
            torch.save({
                "model_state": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": 154_000,
                "tag": "safety_gate_154k_pass",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, "safety_gate_154k.pt")
            upload_file(path_or_fileobj="safety_gate_154k.pt",
                        path_in_repo="checkpoints/safety_gate_154k.pt",
                        repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
            log_step("✅ تم رفع checkpoints/safety_gate_154k.pt")
            # التعديل 2 (المقيم): حذف safety_gate_154k.pt محلياً بعد الرفع
            if os.path.exists("safety_gate_154k.pt"):
                try:
                    os.remove("safety_gate_154k.pt")
                except OSError:
                    pass
        except Exception as e:
            log_step(f"⚠️ فشل رفع safety_gate_154k checkpoint (غير حرج): {e}")
    return all_pass


# ════════════════════════════════════════════════════════════════════════════
# تحميل نموذج XandraHEPL + patch إضافة routing_shares إلى forward() dict
# ════════════════════════════════════════════════════════════════════════════
# (build_hepl_model مُعرّفة في الأعلى — السطر 441 — مع منطق graft/resume)


def wrap_data_parallel(model):
    """تغليف النموذج بـ nn.DataParallel إن توفرت أكثر من بطاقة GPU.

    فجوة 5: عند تفعيل DataParallel على N_GPUS كروت، نضرب BATCH_SIZE في N_GPUS
    لِنحافظ على حجم الدفعة الفعلي لكل كرت (مطابق للنسخة القديمة السطر 787).
    """
    global BATCH_SIZE
    if N_GPUS > 1:
        log_step(f"🚀 تفعيل DataParallel على {N_GPUS} GPUs")
        model = nn.DataParallel(model)
        # فجوة 5: BATCH_SIZE *= N_GPUS (مطابق للنسخة القديمة السطر 787)
        BATCH_SIZE = BATCH_SIZE * N_GPUS
        log_step(f"   ✅ DataParallel مُفعّل — BATCH_SIZE ضُرب في N_GPUS → {BATCH_SIZE}")
        log_step("   routing_shares سيُجمع تلقائياً عبر dim=0")
    else:
        log_step("   بطاقة GPU واحدة — لا حاجة لـ DataParallel")
    return model


def get_raw_model(model):
    """استخراج النموذج الأصلي من وراء DataParallel / torch.compile."""
    raw = model.module if hasattr(model, 'module') else model
    raw = raw._orig_mod if hasattr(raw, '_orig_mod') else raw
    return raw


# ════════════════════════════════════════════════════════════════════════════
# حلقة التدريب الرئيسية + checkpoint + منطق الأمان
# ════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, step, loss, state, tag="latest"):
    """
    حفظ checkpoint كل 5K خطوة (المرجع §2: A10).
    يرفع الملف إلى HuggingFace باسم checkpoints/<tag>.pt
    """
    try:
        raw_model = get_raw_model(model)
        checkpoint = {
            'model_state': raw_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'step': step,
            'loss': float(loss),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'config': {
                'batch_size': BATCH_SIZE, 'max_len': MAX_LEN,
                'mixed_precision': USE_MIXED_PRECISION,
                'model': 'XandraHEPL_v10',
                'params': sum(p.numel() for p in raw_model.parameters()),
                'dim': MODEL_DIM, 'n_layers': N_LAYERS,
                'n_heads': N_HEADS, 'n_kv_heads': N_KV_HEADS,
                'training_stage': TRAINING_STAGE,
                'grafting': 'surgical_from_base_150k_sigma_0.001',
                'loss_equation': '0.50*L_distilled + 0.50*L_chatml + 0.05*L_gate + 0.50*L_KL',
            }
        }
        local = f"{tag}.pt"
        torch.save(checkpoint, local)
        upload_file(path_or_fileobj=local, path_in_repo=f"checkpoints/{tag}.pt",
                    repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
        log_step(f"💾 تم رفع checkpoints/{tag}.pt @ step {step} (loss={loss:.4f})")
        # التعديل 2 (المقيم): حذف ملف tag.pt المحلي بعد الرفع لمنع امتلاء القرص.
        # كل ملف ~1.99 GB؛ 5 حفظات = 10 GB على قرص كاجل المؤقت (~20 GB).
        # نحتفظ بـ latest.pt محلياً فقط (للاستئناف السريع)، ونحذف step_XXX.pt بعد الرفع.
        if tag != "latest" and os.path.exists(local):
            try:
                os.remove(local)
                log_step(f"🗑️ تم حذف {local} المحلي بعد الرفع (تحرير القرص)")
            except OSError:
                pass

        # حفظ خاص لـ latest دائماً
        if tag != "latest":
            torch.save(checkpoint, "latest.pt")
            upload_file(path_or_fileobj="latest.pt", path_in_repo="checkpoints/latest.pt",
                        repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
            # latest.pt يُحتفظ به محلياً (للاستئناف السريع)، لا يُحذف.
    except Exception as e:
        log_step(f"⚠️ فشل حفظ checkpoint ({tag}): {e}")


def emergency_checkpoint(model, optimizer, step, reason):
    """checkpoint طارئ قبل التوقف بسبب خلل (المرجع §9.2)."""
    try:
        raw_model = get_raw_model(model)
        local = f"emergency_step_{step}.pt"
        torch.save({
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, local)
        upload_file(path_or_fileobj=local,
                    path_in_repo=f"checkpoints/emergency_step_{step}.pt",
                    repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
        log_step(f"🆘 تم رفع checkpoints/emergency_step_{step}.pt — السبب: {reason}")
        # التعديل 2 (المقيم): حذف emergency_step_*.pt محلياً بعد الرفع (تحرير القرص)
        if os.path.exists(local):
            try:
                os.remove(local)
            except OSError:
                pass
    except Exception as e:
        log_step(f"🆘 فشل رفع emergency checkpoint: {e}")


def training_loop(model, optimizer, scheduler_fn, tokenizer, state):
    """
    حلقة التدريب الرئيسية (المرجع §2, §6, §7, §8).
    تنفّذ: grafting → 30K خطوة نظيفة → checkpoint كل 5K → بوابة سلامة عند 152K.
    """
    log_step("\n" + "═" * 70)
    log_step("🚀 بدء حلقة التدريب v10 — Stage 2 HEPL Clean Restart")
    log_step(f"   من step {START_STEP:,} → {TARGET_STEP:,} ({TARGET_STEP - START_STEP:,} خطوة)")
    log_step(f"   BATCH={BATCH_SIZE} × GRAD_ACCUM={GRAD_ACCUM_STEPS} = eff.batch {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    log_step(f"   LR warmup: {LR_WARMUP_START} → {LR_PEAK} خلال {WARMUP_STEPS} خطوة")
    log_step(f"   Loss = 0.50·L_distilled + 0.50·L_chatml + 0.05·L_gate + 0.50·L_KL")
    log_step("═" * 70 + "\n")

    # إعداد loss function للـ KL (لو متوفر في hepl_architecture)
    kl_loss_fn = None
    try:
        from hepl_architecture import SparseTopKDistillationLoss
        kl_loss_fn = SparseTopKDistillationLoss(temperature=2.0).to(device)
        log_step("✅ SparseTopKDistillationLoss جاهز (temperature=2.0)")
    except Exception as e:
        log_step(f"⚠️ SparseTopKDistillationLoss غير متاح — سنُحسب KL يدوياً: {e}")

    # تتبّع البيانات
    losses = []
    telemetry_log = []
    loss_log = []
    m_real_norm_baseline = None
    current_m_real = 0.0   # الخطأ 2: تهيئة لاستخدامها في safety_gate_152k

    # الخطأ 1 (المقيم): عدّاد الاستئناف — نقرأ state['current_step'] ولا نفرض START_STEP.
    # دون هذا الإصلاح، الجلسة الثانية تبدأ من 150K رغم أن main() حدّث state.
    step = max(START_STEP, int(state.get('current_step', START_STEP)))
    log_step(f"📍 training_loop تبدأ من step={step} (state['current_step']={state.get('current_step')})")
    start_time = time.time()
    crash_reason = None

    # ─── BatchPrefetcher (مطابق للنسخة القديمة) ───
    import threading
    import queue as queue_mod

    class BatchPrefetcher:
        def __init__(self, batch_size, max_len, prefetch_count=2):
            self.batch_size = batch_size
            self.max_len = max_len
            self.prefetch_count = prefetch_count
            self.queue = queue_mod.Queue(maxsize=prefetch_count)
            self.stop_flag = False
            self.worker_thread = None

        def _background_loader(self):
            while not self.stop_flag:
                try:
                    batch = get_batch(self.batch_size, self.max_len, tokenizer)
                    self.queue.put(batch, timeout=30)
                except Exception as e:
                    log_step(f"⚠️ خطأ في الـ prefetcher: {e}")
                    time.sleep(1)

        def start(self):
            self.worker_thread = threading.Thread(target=self._background_loader, daemon=True)
            self.worker_thread.start()
            log_step(f"🚀 Prefetcher started (prefetch_count={self.prefetch_count})")

        def get(self):
            return self.queue.get(timeout=60)

        def stop(self):
            self.stop_flag = True

    log_step("🔥 تشغيل BatchPrefetcher...")
    prefetcher = BatchPrefetcher(BATCH_SIZE, MAX_LEN, PREFETCH_BATCHES)
    prefetcher.start()
    time.sleep(2)  # إحماء prefetcher

    # إعداد GradScaler (مطابق للنسخة القديمة)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_MIXED_PRECISION)

    optimizer.zero_grad()
    accum_loss = 0.0

    try:
        # الخطأ 1 (المقيم): عدد التكرارات = TARGET_STEP - step (لا TARGET_STEP - START_STEP).
        # max(0, ...) يحمي من حالة شاذة: step >= TARGET_STEP.
        for i in range(max(0, TARGET_STEP - step)):
            elapsed = (time.time() - start_time) / 60
            if elapsed >= MAX_TIME_MINUTES:
                log_step(f"⏰ بلغنا الحد الزمني ({MAX_TIME_MINUTES} دقيقة) — إنهاء نظيف")
                crash_reason = "time_limit"
                break

            # ضبط LR حسب جدولة الإحماء + cosine
            current_lr = scheduler_fn(step)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

            # ─── جلب batch ───
            try:
                raw_batch = prefetcher.get()
                batch = raw_batch["input_ids"].to(device, non_blocking=True)
                t_manifold = raw_batch.get("target_manifold")
                if t_manifold is not None:
                    t_manifold = t_manifold.to(device, non_blocking=True)
                t_top_ids = raw_batch.get("teacher_top32_ids")
                if t_top_ids is not None:
                    t_top_ids = t_top_ids.to(device, non_blocking=True)
                t_top_lps = raw_batch.get("teacher_top32_logprobs")
                if t_top_lps is not None:
                    t_top_lps = t_top_lps.to(device, non_blocking=True)
                # الخطأ 3 (المقيم): kl_mask من get_batch = "أين الـ padding؟" (لـ L_KL)
                kl_mask = raw_batch.get("kl_mask")
                if kl_mask is not None:
                    kl_mask = kl_mask.to(device, non_blocking=True)
            except Exception as e:
                log_step(f"⚠️ تخطّي batch بسبب خطأ: {e}")
                continue

            # ─── بناء قناع "أين الإجابة؟" الحقيقي (المرجع §5.2 — القرار A3) ───
            # الخطأ 3 (المقيم): resp_mask يُبنى دائماً عبر build_response_mask(batch)
            # (لا يُؤخذ من get_batch الذي يُمرّر kl_mask فقط). هذا يضمن أن
            # compute_response_only_loss يرى قناع الإجابة الحقيقي حتى في دفعات التقطير.
            resp_mask = build_response_mask(batch)

            # kl_mask fallback: إن لم يُمرّر من get_batch (دفعات ChatML)، نستخدم resp_mask
            # (قناع الإجابة) لـ L_KL أيضاً — هذا أفضل من قناع padding فقط.
            if kl_mask is None:
                kl_mask = resp_mask.float()

            # ─── forward pass ───
            stage2_step = max(0, step - START_STEP)  # 0 → 30K
            with torch.amp.autocast('cuda', enabled=USE_MIXED_PRECISION, dtype=torch.float16):
                out_dict = model(
                    batch, targets=batch, step=stage2_step,
                    target_manifold=t_manifold
                )

                # ─── قراءة الخسائر الفرعية من forward() dict ───
                # l_sft من forward() = CE على كل التسلسل (يُستخدم للمراقبة فقط)
                l_sft_fwd = out_dict["l_sft"]
                if l_sft_fwd is not None and l_sft_fwd.ndim > 0:
                    l_sft_fwd = l_sft_fwd.mean()
                elif l_sft_fwd is None:
                    l_sft_fwd = torch.tensor(0.0, device=device)

                l_bal = out_dict["l_bal"]
                if l_bal.ndim > 0:
                    l_bal = l_bal.mean()

                l_gate = out_dict["l_gate"]
                if l_gate.ndim > 0:
                    l_gate = l_gate.mean()

                # ─── L_sft الفعلي: Response-Only Loss Masking (القرار A3) ───
                # نُعيد حساب CE على الإجابة فقط (مع <|im_end|>، مستبعد PAD_ID).
                # هذا يحل مشكلة فساد الردود السابقة (التدريب على السؤال + الإجابة).
                if out_dict.get("logits") is not None:
                    l_sft = compute_response_only_loss(out_dict["logits"], batch, resp_mask)
                else:
                    l_sft = l_sft_fwd  # fallback نادر: لا logits متاحة

                l_kl = torch.tensor(0.0, device=device)
                # KL: إن وُجدت teacher_top_ids و logits (مطابق للنسخة القديمة)
                # الخطأ 3 (المقيم): L_KL يستخدم kl_mask (قناع padding) وليس resp_mask.
                if kl_loss_fn is not None and t_top_ids is not None and out_dict.get("logits") is not None:
                    l_kl = kl_loss_fn(out_dict["logits"], t_top_ids, t_top_lps, kl_mask)

                # ─── تفعيل beta و gamma تدريجياً (مطابق للنسخة القديمة v9.1) ───
                beta  = BETA_L_GATE  if stage2_step >= 1000 else 0.0
                gamma = GAMMA_L_KL  if stage2_step >= 2000 else 0.0
                alpha_w = ALPHA_L_BAL   # ثابت 0.01

                # ─── معادلة الخسارة (المرجع §6 + القرار A3) ───
                # loss = l_sft_resp + (alpha_w * l_bal) + (beta * l_gate) + (gamma * l_kl)
                # حيث l_sft_resp = compute_response_only_loss(logits, labels, response_mask)
                # لا توجد أوزان 0.5/0.5 صريحة — التوزيع 50/50 يحدث في مستوى batch selection.
                loss = l_sft + (alpha_w * l_bal) + (beta * l_gate) + (gamma * l_kl)
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            # ─── NaN/Inf detection (المرجع §9.1: F13) ───
            if not torch.isfinite(loss):
                log_step(f"[FATAL] non-finite loss at step {step}: {loss.item()}")
                emergency_checkpoint(model, optimizer, step, "non_finite_loss")
                crash_reason = "non_finite_loss"
                break

            # ─── مراقبة L_KL والخسائر الفرعية كل 100 خطوة (مطابق للنسخة القديمة) ───
            if step % 100 == 0:
                _mr = "n/a"
                if "m_real_norm" in out_dict and out_dict["m_real_norm"] is not None:
                    current_m_real_norm = float(out_dict["m_real_norm"].mean().item())
                    _mr = f"{current_m_real_norm:.4f}"
                # l_sft = Response-Only loss (الفعلي)؛ l_sft_fwd = CE على كل التسلسل (مراقبة)
                log_step(f"[Stage2] step {step} | L_SFT_resp {l_sft.item():.4f} | "
                         f"L_SFT_full {l_sft_fwd.item():.4f} | "
                         f"L_bal {l_bal.item():.4f} | L_gate {l_gate.item():.4f} | "
                         f"L_KL {l_kl.item():.4f} | beta {beta} gamma {gamma} | "
                         f"M_real_norm {_mr}")

            # ─── مراقبة نسبة L_distilled/L_chatml في [150K, 152K] (المرجع §6.1) ───
            # ملاحظة: في النسخة القديمة لا توجد L_distilled/L_chatml منفصلتين —
            # التوزيع 50/50 يحدث في مستوى batch. لذلك نراقب L_SFT بدلاً منه.
            if step < SAFETY_GATE_STEP and step % SAFETY_MONITOR_EVERY == 0 and step > START_STEP:
                log_step(f"[SAFETY MONITOR step={step}] "
                         f"L_SFT={l_sft.item():.4f} "
                         f"L_KL={l_kl.item():.4f} (target: <{KL_TARGET_BY_152K} by 152K)")
                # شرط التوقف: L_KL لا يهبوط من القيمة الابتدائية (~38.6)
                # هذا شرط slack — نتحقق من L_KL < 25 بحلول 152K (عتبة وسطى)
                if stage2_step >= 1500 and l_kl.item() > 25.0:
                    log_step(f"⚠️ L_KL لم يهبوط كفاية ({l_kl.item():.2f} > 25.0) — مراقبة مشدّدة")

            # ─── تحديث optimizer كل GRAD_ACCUM_STEPS ───
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # ─── M_real norm kill-switch (المرجع §7.1, §9.1: F10) ───
            if "m_real_norm" in out_dict and out_dict["m_real_norm"] is not None:
                current_m_real = float(out_dict["m_real_norm"].mean().item())
                if step % 200 == 0:
                    if m_real_norm_baseline is None:
                        m_real_norm_baseline = current_m_real
                        log_step(f"[KillSwitch] baseline M_real_norm = {m_real_norm_baseline:.4f}")
                    else:
                        _ratio = current_m_real / max(m_real_norm_baseline, 1e-8)
                        if step % 500 == 0:
                            log_step(f"[KillSwitch] step {step} M_real_norm = {current_m_real:.4f} (ratio {_ratio:.2f})")
                        if _ratio < M_REAL_NORM_MIN_RATIO:
                            log_step(f"[FATAL] M_real_norm collapsed to {_ratio:.2%} — catastrophic forgetting")
                            emergency_checkpoint(model, optimizer, step, "m_real_norm_collapse")
                            crash_reason = "m_real_norm_collapse"
                            break

            # ─── المراقبة الثلاثية كل 500 خطوة (المرجع §7) ───
            if step % TELEMETRY_EVERY == 0 and step > START_STEP:
                # تجميع routing_shares عبر DataParallel (إن وُجد الحقل)
                if "routing_shares" in out_dict:
                    shares_tensor = out_dict["routing_shares"]
                    # shape [num_gpus, 4] بعد gather → mean على dim=0
                    if shares_tensor.ndim == 2:
                        shares_final = shares_tensor.mean(dim=0)
                    elif shares_tensor.ndim == 1:
                        shares_final = shares_tensor
                    else:
                        shares_final = shares_tensor.flatten()[:4]

                    metrics = compute_gate_metrics(shares_final)
                    telemetry_log.append(metrics)
                    # التعديل 2 (المقيم، جولة 4): حفظ manifold_telemetry في state.json
                    # لِيقرأها شريط تقدم الفضاءات في لوحة التحكم الرئيسية.
                    # دون هذا، تُطبع القيم في السجل فقط ولا تراها لوحة التحكم.
                    state["manifold_telemetry"] = metrics["shares"]
                    state["manifold_telemetry_step"] = step
                    state["manifold_telemetry_at"] = datetime.now(timezone.utc).isoformat()
                    log_step(f"[TELEMETRY step={step}] "
                             f"entropy={metrics['entropy']:.4f} "
                             f"gini={metrics['gini']:.4f} "
                             f"drift={metrics['drift']:.4f} "
                             f"shares={metrics['shares']}")
                    # رفع state.json فوراً (لا ننتظر heartbeat كل 100 خطوة) لِترى
                    # لوحة التحكم شريط الفضاءات محدّثاً في اللحظة (كل 500 خطوة).
                    try:
                        heartbeat(state, step)
                    except Exception as _hb_err:
                        log_step(f"⚠️ heartbeat بعد telemetry فشل (غير حرج): {_hb_err}")

                    # تحقق من العتبات
                    violated, msg = check_safety_violation(metrics, step)
                    if violated:
                        log_step(f"⚠️ SAFETY GATE — {msg}")
                        emergency_checkpoint(model, optimizer, step, "telemetry_violation")
                        crash_reason = "telemetry_violation"
                        break

            step += 1
            losses.append(loss.item() * GRAD_ACCUM_STEPS)
            loss_log.append({
                "step": step,
                "loss": loss.item() * GRAD_ACCUM_STEPS,
                # l_sft = Response-Only loss (الفعلي في المعادلة، القرار A3)
                # l_sft_full = CE على كل التسلسل (مراقبة فقط)
                "l_sft":       float(l_sft.detach().item()) if isinstance(l_sft, torch.Tensor) else 0,
                "l_sft_full":  float(l_sft_fwd.detach().item()) if isinstance(l_sft_fwd, torch.Tensor) else 0,
                "l_bal":  float(l_bal.detach().item()) if isinstance(l_bal, torch.Tensor) else 0,
                "l_gate": float(l_gate.detach().item()) if isinstance(l_gate, torch.Tensor) else 0,
                "l_kl":   float(l_kl.detach().item()) if isinstance(l_kl, torch.Tensor) else 0,
            })

            # ─── بوابة السلامة عند step 152,000 (المرجع §8) ───
            # الخطأ 2 (المقيم): نمرّر current_m_real (المحدّثة في كل خطوة من out_dict)
            # بدلاً من استدعاء model.fluency_probe() التي تنهار عند DataParallel.
            if step == SAFETY_GATE_STEP:
                passed = safety_gate_152k(
                    get_raw_model(model), optimizer, telemetry_log, loss_log,
                    m_real_norm_baseline, current_m_real=current_m_real
                )
                if not passed:
                    log_step("⛔ بوابة السلامة فشلت — توقف قبل الاستمرار إلى 180K")
                    crash_reason = "safety_gate_failed"
                    break
                log_step("✅ بوابة السلامة اجتازت — الاستمرار إلى 180K")

            # ─── بوابة السلامة الصارمة الثانية عند step 154,000 (الضبط 2 من المقيم) ───
            # gamma يُفعّل L_KL عند 152K؛ قيمتها الأولى ~35–38، وتصل إلى < 15 بعد ~1,500 خطوة.
            # فحص 154K يضمن هبوط L_KL فعلاً (لا مجرد عدم انفجارها).
            if step == 154_000:
                passed_154 = safety_gate_154k(
                    get_raw_model(model), optimizer, telemetry_log, loss_log,
                    m_real_norm_baseline, current_m_real=current_m_real
                )
                if not passed_154:
                    log_step("⛔ بوابة السلامة 154K فشلت (L_KL لم يهبط إلى < 15) — توقف")
                    crash_reason = "safety_gate_154k_failed"
                    break
                log_step("✅ بوابة السلامة 154K اجتازت — L_KL هبط إلى < 15.0")

            # ─── checkpoint كل 5K خطوة (المرجع §2: A10) ───
            if step > START_STEP and (step - START_STEP) % SAVE_EVERY == 0:
                save_checkpoint(model, optimizer, step,
                                loss.item() * GRAD_ACCUM_STEPS, state,
                                tag=f"step_{step}")
                # تحديث state.json
                state['current_step'] = step
                state['current_loss'] = loss.item() * GRAD_ACCUM_STEPS
                state['updated_at'] = datetime.now(timezone.utc).isoformat()
                if 'loss_history' not in state:
                    state['loss_history'] = []
                state['loss_history'].append({
                    'step': step, 'loss': loss.item() * GRAD_ACCUM_STEPS,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                })
                if len(state['loss_history']) > 100:
                    state['loss_history'] = state['loss_history'][-100:]
                heartbeat(state, step)

            # ─── نبضة قلب كل 100 خطوة ───
            if i > 0 and i % 100 == 0:
                state['current_step'] = step
                state['current_loss'] = loss.item() * GRAD_ACCUM_STEPS
                heartbeat(state, step)

            # ─── سجل تقدّم كل 10 خطوات ───
            if i % 10 == 0:
                avg_loss = sum(losses[-10:]) / min(10, len(losses))
                elapsed_min = (time.time() - start_time) / 60
                spm = len(losses) / max(elapsed_min, 0.01)
                log_step(f"[{datetime.now().strftime('%H:%M:%S')}] "
                         f"Step {step:,} | loss: {loss.item()*GRAD_ACCUM_STEPS:.4f} | "
                         f"avg: {avg_loss:.4f} | {spm:.1f} steps/min | lr={current_lr:.2e}")

            # ─── الهدف النهائي: 180K ───
            if step >= TARGET_STEP:
                log_step(f"🎯 اكتملت الجولة النظيفة عند step {step:,} — إنهاء نظيف!")
                break

    except KeyboardInterrupt:
        crash_reason = "manual_stop"
        log_step("⛔ تم الإيقاف اليدوي — تحرير القفل...")
    except Exception as e:
        crash_reason = f"crash: {str(e)[:100]}"
        log_step(f"💥 انهيار غير متوقع: {e}")
        traceback.print_exc()
        with open('crash_log.txt', 'w') as f:
            f.write(traceback.format_exc())

    # ─── حفظ checkpoint نهائي ───
    if losses:
        try:
            final_loss = losses[-1]
            save_checkpoint(model, optimizer, step, final_loss, state, tag="step_final")
            state['current_step'] = step
            state['current_loss'] = final_loss
            state['best_loss'] = min(state.get('best_loss', float('inf')), final_loss)
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            gpu_hours = (time.time() - SESSION_START_TIME) / 3600
            state['gpu_hours_used'] = state.get('gpu_hours_used', 0) + gpu_hours
            state.setdefault('history', []).append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'started_at': SESSION_STARTED_AT,
                'final_step': step, 'final_loss': final_loss,
                'steps_done': len(losses), 'gpu_hours': gpu_hours,
                'worker_id': WORKER_ID,
                'exit_reason': crash_reason or "completed",
                'version': 'v10',
            })
            if len(state['history']) > 50:
                state['history'] = state['history'][-50:]
        except Exception as e:
            log_step(f"⚠️ فشل حفظ checkpoint النهائي: {e}")

    log_step(f"\n✅ انتهت الجلسة. الخطوة النهائية: {step:,}")
    log_step(f"⏱️  زمن الجلسة: {(time.time() - SESSION_START_TIME)/3600:.2f} ساعة")
    log_step(f"📊 متوسط الخطوة/دقيقة: {len(losses) / max((time.time() - SESSION_START_TIME)/60, 0.01):.1f}")
    return crash_reason


# ════════════════════════════════════════════════════════════════════════════
# نقطة الدخول الرئيسية
# ════════════════════════════════════════════════════════════════════════════

def main():
    """نقطة الدخول — تجريبي يتحقق من سلامة البنية، لا يدرب فعلياً حتى يُحقن مُحمّل البيانات."""
    log_step("=" * 70)
    log_step("🚀 xandra_worker.py v10 — Stage 2 HEPL Clean Restart")
    log_step("🩺 المرجع: Xandra_Master_Engineering_Reference.docx")
    log_step("=" * 70)

    # 1. طلب القفل
    if not acquire_lock():
        log_step("⛔ لا يمكن التدريب — جلسة أخرى قيد التشغيل")
        sys.exit(0)

    # الضبط 1 (المقيم): إشارة الإقلاع 'booted' فور حجز القفل — قبل أي تنزيل ثقيل.
    # دون هذا، يظن المايسترو أن الحاوية عالقة أثناء تنزيل base_150k.pt (1.14 GB)
    # وملفات التقطير، وقد يُعيد التشغيل خطأً.
    try:
        _boot_state = load_state()
        _boot_state['notebook_status'] = {
            'phase': 'booted',
            'started_at': SESSION_STARTED_AT,
        }
        _boot_state['v10_initialized'] = True
        heartbeat(_boot_state, int(_boot_state.get('current_step', START_STEP)))
        log_step(f"📡 تم إرسال إشارة 'booted' للمايسترو (started_at={SESSION_STARTED_AT})")
    except Exception as _boot_err:
        log_step(f"⚠️ فشل إرسال إشارة booted (غير حرج): {_boot_err}")

    try:
        # 2. تحميل hepl_architecture.py و tokenizer
        load_hepl_architecture()
        tokenizer = load_tokenizer()

        # 2.5 تحضير نمط assistant header (مطلوب لـ Response-Only Loss Masking — القرار A3)
        init_assistant_header_pattern(tokenizer)

        # 3. تحميل بيانات التقطير (Strategy A/B للاستعيانة المتوازنة بين الفضاءات)
        load_distilled_records()

        # 3.5 تحميل تيار ChatML العام (combined_stream)
        setup_combined_stream()
        if STREAMING_READY:
            refill_text_buffer()

        # 4. قراءة state مبكراً — قبل بناء النموذج (الفجوة 3: Resuming Bug)
        #    نحتاج state['current_step'] لِنقرر: graft من base_150k.pt أم استئناف من latest.pt
        state = load_state()
        prev_step = int(state.get('current_step', 0))

        # 5. بناء النموذج + Surgical Grafting (أو استئناف من checkpoint — الفجوة 3)
        #    build_hepl_model يقرأ state['current_step'] داخلياً لِيقرر graft أم resume
        model = build_hepl_model(tokenizer, state=state)
        model = wrap_data_parallel(model)

        # 6. مُحسّن AdamW نظيف
        optimizer = build_fresh_optimizer(get_raw_model(model))

        # 6.5 استئناف optimizer_state إن كنا في وضع الاستئناف (الفجوة 3)
        if prev_step > START_STEP and hasattr(get_raw_model(model), '_latest_ckpt_for_opt'):
            _load_optimizer_state(optimizer, get_raw_model(model))

        # 7. لا نفرض START_STEP إن كان التدريب توقّف بعده (الفجوة 3)
        if state.get('current_step', 0) < START_STEP:
            state['current_step'] = START_STEP
        state['v10_initialized'] = True
        heartbeat(state, state['current_step'])

        # 8. حلقة التدريب
        crash_reason = training_loop(
            model, optimizer, warmup_lr_schedule, tokenizer, state
        )

        # 9. تحرير القفل
        release_lock(state, reason=crash_reason or "completed")
        log_step("🔒 تم تحرير القفل")

    except Exception as e:
        log_step(f"🚨 خطأ كارثي: {e}")
        traceback.print_exc()
        try:
            state = load_state()
            release_lock(state, reason=f"fatal: {str(e)[:50]}")
        except Exception:
            log_step("💀 فشل تحرير القفل — تدخّل يدوي مطلوب")
    finally:
        # فجوة 5: os._exit(0) لِمنع خيوط BatchPrefetcher المعلّقة (Zombie state)
        # مطابق للنسخة القديمة السطر 1258. بدون هذا، تبقى خيوط daemon معلّقة
        # وتدخل الجلسة في وضعية زومبي بعد انتهاء التدريب.
        log_step("🔚 إغلاق إجباري عبر os._exit(0) — منع خيوط prefetcher المعلّقة")
        # flush stdout/stderr قبل الإغلاق
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(0)


if __name__ == "__main__":
    main()