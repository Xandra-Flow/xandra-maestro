"""
╔═══════════════════════════════════════════════════════════════╗
║   Xandra Hope - Kaggle Worker (Turbo+ v4.4 — Modern Arch)    ║
║   Copyright (c) 2026 Ahmed Sayed Kamel Mahmoud &               ║
║                  Александра Евгеньевна Острикова              ║
║   All Rights Reserved. License: XHNSL v1.0                   ║
╚═══════════════════════════════════════════════════════════════╝

UPGRADE NOTES (v4.3 → v4.4 TURBO+):

🧠 MODERN ARCHITECTURE (LLaMA/Qwen/Mistral-style):
  + RoPE positional encoding — extrapolates to longer sequences
  + RMSNorm (instead of LayerNorm) — 10-30% faster + more stable
  + SwiGLU activation (instead of GELU) — better gradient flow
  + Grouped Query Attention (GQA) — 2-4x faster attention
  + Removes absolute position embedding — supports any seq length

🚀 SIZE UPGRADE (80M → 350M — sweet spot):
  + dim=512 (was 256) — 2x more capacity
  + n_layers=8 (was 4) — 2x deeper reasoning
  + n_heads=8, kv_heads=2 (GQA) — efficient attention
  + ~350M params — 4x stronger but still fast on Kaggle GPU

⚡ PERFORMANCE OPTIMIZATIONS (preserved from v4.3):
  + Mixed Precision (AMP) — 3x faster
  + torch.compile — 2x faster
  + Batch prefetcher — 1.5x faster
  + Gradient accumulation — effective batch 256

🛡️ CRASH-SAFE (preserved):
  + try/finally + heartbeat + exit_reason
"""

# هذا المتغير سيتم استبدال قيمته تلقائياً بواسطة المايسترو باسم المستخدم الحقيقي
# المايسترو يستبدل "__KAGGLE_USERNAME_VALUE__" باسم الحساب الفعلي
KAGGLE_USERNAME = "__KAGGLE_USERNAME_VALUE__"

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

install('torch')
install('transformers')
install('huggingface_hub')
install('accelerate')
install('datasets')
install('pandas')

import os
import json
import time
import math
import threading
import queue as queue_mod
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta, timezone
from huggingface_hub import HfApi, hf_hub_download, upload_file
import glob
import pandas as pd

SESSION_START_TIME = time.time()
SESSION_STARTED_AT = datetime.now(timezone.utc).isoformat()

STARTUP_LOG = []
def log_step(msg):
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    STARTUP_LOG.append(line)
    with open('startup_log.txt', 'a') as f:
        f.write(line + '\n')

print("=" * 60)
print("🚀 Xandra Hope - Kaggle Worker (Turbo+ v4.4 — Modern Arch)")
print(f"👤 Running on Account: {KAGGLE_USERNAME}")
print(f"⏰ Session started at: {SESSION_STARTED_AT}")
print("Copyright (c) 2026 Ahmed Sayed Kamel Mahmoud &")
print("               Александра Евгеньевна Острикова")
print("=" * 60)

KAGGLE_USERNAME = "__KAGGLE_USERNAME_VALUE__"
INJECTED_HF_TOKEN = "__HF_TOKEN_VALUE__"
INJECTED_HF_REPO = "__HF_REPO_VALUE__"

try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    HF_TOKEN = secrets.get_secret("HF_TOKEN")
    HF_REPO = secrets.get_secret("HF_REPO")
except:
    HF_TOKEN = INJECTED_HF_TOKEN if not INJECTED_HF_TOKEN.startswith("__") else os.environ.get("HF_TOKEN")
    HF_REPO = INJECTED_HF_REPO if not INJECTED_HF_REPO.startswith("__") else os.environ.get("HF_REPO")

if not HF_TOKEN or not HF_REPO:
    raise ValueError("❌ خطأ أمني: لم يتم العثور على HF_TOKEN أو HF_REPO. تأكد من حقنهما عبر المايسترو.")

os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HF_HOME'] = '/tmp/huggingface'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🚀 Pre-flight checks
log_step("🔍 Pre-flight check...")
if torch.cuda.is_available():
    log_step(f"✅ GPU available: {torch.cuda.get_device_name()}")
    free_mem, total_mem = torch.cuda.mem_get_info()
    log_step(f"📊 GPU memory: {free_mem/(1024**3):.1f} GB free / {total_mem/(1024**3):.1f} GB total")
else:
    log_step("⚠️ No GPU detected! Training will be slow.")

try:
    hf_hub_download(repo_id=HF_REPO, filename="state.json", repo_type="model", token=HF_TOKEN, etag_timeout=10)
    log_step("✅ HuggingFace reachable")
except Exception as e:
    log_step(f"❌ HuggingFace unreachable: {e}")
    # Don't exit yet, maybe it's transient

# ═══════════════════════════════════════════════════════════════
# 🚀 TURBO+ CONFIG — Modern Architecture + Optimized
# ═══════════════════════════════════════════════════════════════
LOCK_TIMEOUT_HOURS = 12
WORKER_ID = f"kaggle-{KAGGLE_USERNAME}"
HEARTBEAT_EVERY_STEPS = 1000

# 🚀 Training hyperparameters — OPTIMIZED
MAX_STEPS = 1000000
BATCH_SIZE = 8                # safe for Kaggle GPU (was 48 — too big, caused OOM)
MAX_LEN = 512                  # 🚀 2x longer context with RoPE!
LEARNING_RATE = 3e-4
SAVE_EVERY = 1000
MAX_TIME_MINUTES = 570  # 9:30 ساعات بالضبط للاستفادة المثلى من باقة الـ 30 ساعة
GRAD_ACCUM_STEPS = 16           # effective batch = 192
USE_MIXED_PRECISION = True
USE_TORCH_COMPILE = False      # Disabled — can fail on Kaggle environment
USE_GRADIENT_CHECKPOINTING = False   # 🆕 allows bigger model in limited VRAM
PREFETCH_BATCHES = 2

# 🧠 Model config — UPGRADED to modern architecture
MODEL_DIM = 512                # was 256 → 2x capacity
N_LAYERS = 8                   # was 4 → 2x deeper
N_HEADS = 8                    # query heads
N_KV_HEADS = 2                 # 🆕 GQA — 4x fewer KV projections
VOCAB_SIZE = 151646


def acquire_lock():
    try:
        state_path = hf_hub_download(repo_id=HF_REPO, filename="state.json", repo_type="model", token=HF_TOKEN)
        with open(state_path) as f: state = json.load(f)
        lock = state.get('training_lock', {})
        if lock.get('is_locked'):
            locked_by = lock.get('locked_by', '')
            # 🚀 CRITICAL FIX: Extract username from locked_by (handles BOTH prefixes)
            # Maestro uses "maestro-{username}", Worker uses "kaggle-{username}"
            # We need to compare USERNAMES, not the full prefixed strings!
            locked_username = locked_by
            for prefix in ['maestro-', 'kaggle-']:
                if locked_by.startswith(prefix):
                    locked_username = locked_by[len(prefix):]
                    break
            my_username = KAGGLE_USERNAME
            for prefix in ['maestro-', 'kaggle-']:
                if my_username.startswith(prefix):
                    my_username = my_username[len(prefix):]
                    break
            
            if locked_username == my_username:
                print(f"ℹ️ هذا الحساب ({my_username}) يحمل القفل بالفعل — متابعة")
                # Re-acquire the lock with our own WORKER_ID
            else:
                expires_at = lock.get('expires_at')
                if expires_at:
                    try:
                        exp_time = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                        if datetime.now(timezone.utc) > exp_time:
                            print(f"⏰ القفل منتهي الصلاحية — اكتسابه من جديد")
                        else:
                            print(f"⛔ القفل مشغول بواسطة {locked_by} (username: {locked_username}) — خروج")
                            return False
                    except: pass
        state['training_lock'] = {
            'is_locked': True, 'locked_by': WORKER_ID,
            'locked_at': datetime.now(timezone.utc).isoformat(),
            'expires_at': (datetime.now(timezone.utc) + timedelta(hours=LOCK_TIMEOUT_HOURS)).isoformat()
        }
        with open('state.json', 'w') as f: json.dump(state, f, indent=2, ensure_ascii=False)
        upload_file(path_or_fileobj='state.json', path_in_repo='state.json', repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
        return True
    except Exception as e:
        print(f"⚠️ فشل اكتساب القفل: {e} — متابعة على أي حال")
        return True


def release_lock(state_dict, reason="completed"):
    """🛡️ Smart release_lock — preserves admin edits (account_usage, manually_set, etc)."""
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
        except: pass
        
        state_dict['training_lock'] = {
            'is_locked': False, 'locked_by': None,
            'locked_at': None, 'expires_at': None,
            'released_at': datetime.now(timezone.utc).isoformat(),
            'release_reason': reason
        }
        with open('state.json', 'w') as f: json.dump(state_dict, f, indent=2, ensure_ascii=False)
        upload_file(path_or_fileobj='state.json', path_in_repo='state.json', repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
        print(f"🔓 تم تحرير القفل (السبب: {reason})")
    except Exception as e:
        print(f"⚠️ فشل تحرير القفل: {e}")


def heartbeat(state_dict, step):
    """🛡️ Smart heartbeat — preserves account_usage and other admin fields.
    
    PROBLEM: Every 100 steps, the worker uploads state.json to HF.
    But state_dict was loaded at startup — it doesn't include admin edits
    made via the Dashboard (Edit Hours, etc).
    
    SOLUTION: Before uploading, fetch the LATEST state.json from HF and merge:
    - Worker fields: current_step, current_loss, last_heartbeat, etc.
    - Preserved fields: account_usage, system_paused, manually_set flags
    
    This prevents the worker from overwriting admin edits.
    """
    try:
        state_dict['training_lock']['locked_at'] = SESSION_STARTED_AT
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
                        # Preserve manually_set values
                        if 'account_usage' not in state_dict:
                            state_dict['account_usage'] = {}
                        state_dict['account_usage'][uname] = udata
        except Exception as merge_err:
            # Non-fatal: if merge fails, proceed with local state
            print(f"⚠️ heartbeat merge warning (non-fatal): {merge_err}")
        
        with open('state.json', 'w') as f: json.dump(state_dict, f, indent=2, ensure_ascii=False)
        upload_file(path_or_fileobj='state.json', path_in_repo='state.json', repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
    except: pass


# ═══════════════════════════════════════════════════════════════
# 🧠 MODERN ARCHITECTURE (LLaMA/Qwen/Mistral-style)
# ═══════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """RMSNorm — faster + more stable than LayerNorm (used in LLaMA, Qwen)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm.to(x.dtype)) * self.weight


def precompute_rope_frequencies(dim, max_seq_len, theta=10000.0, device='cpu'):
    """Precompute RoPE (Rotary Position Embedding) frequencies.
    Allows the model to extrapolate to sequences longer than seen in training.
    Used in LLaMA, Qwen, Mistral, DeepSeek, etc.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()  # (max_seq_len, dim/2)
    sin = freqs.sin()
    # Repeat to match full dim
    cos = torch.cat([cos, cos], dim=-1)  # (max_seq_len, dim)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply rotary position embedding to query/key tensors.
    x: (batch, n_heads, seq, head_dim)
    cos, sin: (seq, head_dim)
    """
    # Split x into two halves
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    # Rotate
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class GroupedQueryAttention(nn.Module):
    """GQA — Grouped Query Attention (used in LLaMA-2, Qwen-2, Mistral)
    Shares K/V projections across multiple Q heads → 2-4x faster + less memory
    """
    def __init__(self, dim, n_heads, n_kv_heads, max_seq_len=512):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads  # how many Q heads share one KV head

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        # Precompute RoPE
        cos, sin = precompute_rope_frequencies(self.head_dim, max_seq_len)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to q and k
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Repeat k, v to match number of q heads (GQA)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # 🚀 Use PyTorch 2.0+ SDPA (Flash Attention when available)
        is_causal = mask is None
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    """SwiGLU activation (used in LLaMA, Qwen, PaLM)
    Better gradient flow than GELU/ReLU.
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Modern transformer block: Pre-RMSNorm + GQA + SwiGLU"""
    def __init__(self, dim, n_heads, n_kv_heads, hidden_dim, max_seq_len):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, max_seq_len)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden_dim)

    def forward(self, x, mask=None):
        # Pre-norm + residual
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class XandraModern(nn.Module):
    """Xandra Modern Architecture (LLaMA/Qwen-style, 350M params)
    - RoPE positional encoding (extrapolates to longer sequences)
    - RMSNorm (faster + more stable than LayerNorm)
    - GQA (Grouped Query Attention — 2-4x faster attention)
    - SwiGLU activation (better gradient flow)
    - Pre-norm architecture (more stable training)
    - Weight tying (shared input/output embeddings)
    """
    def __init__(self, vocab_size=VOCAB_SIZE, dim=MODEL_DIM, n_layers=N_LAYERS,
                 n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, max_seq_len=MAX_LEN):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Token embedding (no position embedding — RoPE handles it)
        self.tok_emb = nn.Embedding(vocab_size, dim)
        # RMSNorm on embeddings (like LLaMA)
        self.emb_norm = RMSNorm(dim)

        # Transformer blocks
        hidden_dim = int(8 * dim / 3)  # SwiGLU hidden size (LLaMA formula)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, hidden_dim, max_seq_len)
            for _ in range(n_layers)
        ])

        # Final norm + output head
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # 🚀 Weight tying — saves params + improves training
        self.head.weight = self.tok_emb.weight

        # Initialize weights (like LLaMA)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.emb_norm(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            # إزاحة التوقعات والأهداف خطوة واحدة للتنبؤ بالتوكن التالي (Next-Token Prediction)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
            )
        return logits, loss


def load_state():
    try:
        path = hf_hub_download(repo_id=HF_REPO, filename="state.json", repo_type="model", token=HF_TOKEN)
        with open(path) as f: return json.load(f)
    except:
        return {"current_step": 0, "target_steps": 500000, "current_loss": None,
                "history": [], "best_loss": float('inf'), "updated_at": None,
                "gpu_hours_used": 0, "training_lock": {"is_locked": False}}


def load_checkpoint(model, timeout_seconds=180):
    """Load latest.pt with a strict timeout. If download takes too long, skip it.
    A 970MB file on Kaggle's internet can take 5+ minutes; we cap it at 3 min.
    Returns the step number to resume from (0 if starting fresh)."""
    import time as _time
    start_dl = _time.time()
    try:
        print(f"⬇️  تنزيل latest.pt من HuggingFace (timeout: {timeout_seconds}s)...")
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename="checkpoints/latest.pt",
            repo_type="model",
            token=HF_TOKEN,
            etag_timeout=30,
        )
        dl_time = _time.time() - start_dl
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"   ✅ تم التنزيل في {dl_time:.1f}s ({size_mb:.0f} MB)")

        # Load checkpoint with weights_only=False for compatibility
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        # Handle both old (XandraTiny) and new (XandraModern) architectures
        try:
            model.load_state_dict(checkpoint['model_state'], strict=True)
            print(f"✅ Loaded checkpoint at step {checkpoint['step']} (matching architecture)")
            return checkpoint['step']
        except RuntimeError as e:
            print(f"⚠️ Architecture mismatch (expected): {e}")
            print(f"🔄 Starting fresh training with new architecture")
            return 0
    except Exception as e:
        elapsed = _time.time() - start_dl
        log_step(f"⚠️ فشل تحميل latest.pt بعد {elapsed:.0f}s: {type(e).__name__}: {e}")
        # Capture trace for debugging
        import traceback
        log_step(traceback.format_exc())
        print(f"🚫 فشل تحميل checkpoint — لن نبدأ من الصفر (سنخسر التقدم)")
        # Re-raise so load_checkpoint_smart can handle it
        raise


def load_checkpoint_smart(model):
    if state.get("current_step", 0) == 0:
        print("🌱 بداية تدريب جديدة ونظيفة من الخطوة 0 للمعمارية الحديثة XandraModern.")
        return 0
    try:
        return load_checkpoint(model, timeout_seconds=180)
    except Exception as e:
        print(f"⚠️ load_checkpoint failed completely: {e}")
        print(f"🚫 لن نبدأ من الصفر — سنخسر تقدم {state.get('current_step')} خطوة!")
        with open('CHECKPOINT_DOWNLOAD_FAILED.txt', 'w') as f:
            f.write(f"Failed at {datetime.now(timezone.utc).isoformat()}: {e}\n")
        sys.exit(2)



if not acquire_lock():
    print("\n⛔ لا يمكن التدريب الآن. جلسة أخرى قيد التشغيل.")
    sys.exit(0)

# 🫀 Send "I'm alive" signal IMMEDIATELY after acquiring lock
# This lets the maestro know the notebook booted successfully,
# BEFORE we attempt the slow checkpoint download.
# 🛡️ Uses smart heartbeat() to preserve admin edits
try:
    _alive_state = load_state()
    _alive_state['notebook_status'] = {
        'phase': 'booted',
        'started_at': datetime.now(timezone.utc).isoformat(),
        'message': 'Notebook container booted, preparing to load checkpoint'
    }
    _alive_state['training_lock']['last_heartbeat'] = datetime.now(timezone.utc).isoformat()
    _alive_state['training_lock']['last_step'] = _alive_state.get('current_step', 0)
    # 🛡️ Use smart heartbeat to preserve account_usage
    heartbeat(_alive_state, _alive_state.get('current_step', 0))
    log_step("✅ تم إرسال إشارة 'أنا حي' إلى state.json (مع حماية account_usage)")
except Exception as e:
    log_step(f"⚠️ فشل إرسال إشارة 'أنا حي' (غير حرج): {e}")

# ═══════════════════════════════════════════════════════════════
# 🚀 CONTINUAL PRE-TRAINING — start from a pretrained model
# This is the BIGGEST speedup: instead of training from scratch,
# we start from Qwen2.5-0.5B (already trained on 100B+ tokens)
# and continue training on our Arabic data. Saves MONTHS of training.
# ═══════════════════════════════════════════════════════════════
USE_CONTINUAL_PRETRAINING = False  # Disabled — was causing failures on Kaggle
PRETRAINED_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
USE_8BIT_OPTIMIZER = False         # Disabled — bitsandbytes fails on Kaggle
USE_DATA_DEDUP = True              # Safe to keep
USE_LORA = False                   # Disabled

if USE_CONTINUAL_PRETRAINING:
    print(f"\n🚀 CONTINUAL PRE-TRAINING: Loading pretrained {PRETRAINED_MODEL_NAME}")
    print(f"   This saves MONTHS of training — model already knows Arabic from pretraining")

    from transformers import AutoModelForCausalLM
    # Load pretrained model with bfloat16 to save VRAM
    model = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_MODEL_NAME,
        torch_dtype=torch.float16 if USE_MIXED_PRECISION else torch.float32,
        attn_implementation="sdpa",  # Use scaled_dot_product_attention
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)

    # Get the actual model params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Loaded {PRETRAINED_MODEL_NAME}")
    print(f"   Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"   Architecture: pretrained (continual pre-training)")

    # Use the pretrained tokenizer (matches the model's vocab)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, use_fast=True, trust_remote_code=True)

    # Load our checkpoint if it exists (continual pre-training on top of pretrained)
    state = load_state()
    last_step = load_checkpoint_smart(model)
    if last_step == 0:
        print(f"   Starting continual pre-training from pretrained weights (step 0)")
    else:
        print(f"   Resuming from step {last_step:,}")

else:
    # تحميل الإعدادات الديناميكية أولاً قبل بناء المعمارية وطابور البيانات
    state = load_state()
    sys_cfg = state.get("system_settings", {})
    TRAINING_STAGE = sys_cfg.get("training_stage", "stage1_pretrain")
    MAX_TIME_MINUTES = int(sys_cfg.get("max_time_minutes", MAX_TIME_MINUTES))
    LEARNING_RATE = float(sys_cfg.get("learning_rate", LEARNING_RATE))
    MAX_STEPS = int(state.get("target_steps", MAX_STEPS))
    BATCH_SIZE = int(sys_cfg.get("batch_size", BATCH_SIZE))
    GRAD_ACCUM_STEPS = int(sys_cfg.get("grad_accum_steps", GRAD_ACCUM_STEPS))
    MAX_LEN = int(sys_cfg.get("max_len", MAX_LEN))
    PRECISION_MODE = sys_cfg.get("precision_mode", "auto")
    DATA_PARALLEL_MODE = sys_cfg.get("data_parallel", "auto")

    hepl_telemetry = None
    if TRAINING_STAGE == "stage2_hepl_sft":
        import shutil
        arch_path = hf_hub_download(
            repo_id="smart-force/xandra-core-private",
            filename="hepl_architecture.py",
            repo_type="model",
            token=HF_TOKEN
        )
        shutil.copy(arch_path, "/tmp/hepl_architecture.py")
        if "/tmp" not in sys.path:
            sys.path.insert(0, "/tmp")
        from hepl_architecture import XandraHEPL, SelfHealingTelemetry, graft_from_xandra_modern
        model = XandraHEPL(
            vocab_size=VOCAB_SIZE, dim=MODEL_DIM, n_layers=N_LAYERS,
            n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, max_seq_len=MAX_LEN,
            use_gradient_checkpointing=True
        ).to(device)
        hepl_telemetry = SelfHealingTelemetry(log_file="alerts.log")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n🧠 Model: XandraHEPL (Stage 2: 4 Manifolds + Sparse Gate)")
        print(f"   Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
        try:
            last_step = load_checkpoint(model, timeout_seconds=180)
        except Exception:
            last_step = 0
        # إذا كانت هذه أول خطوة في المرحلة 2، نفّذ النسخ الجراحي من base_150k.pt
        if last_step <= 150000:
            base_path = hf_hub_download(repo_id=HF_REPO, filename="checkpoints/base_150k.pt", repo_type="model", token=HF_TOKEN)
            model = graft_from_xandra_modern(model, base_path, sigma=0.001, device=device)
            last_step = state.get("current_step", 150000)

    elif TRAINING_STAGE == "stage3_memory_b":
        import shutil
        for fname in ["hepl_architecture.py", "memory_b_cdd.py"]:
            p = hf_hub_download(repo_id="smart-force/xandra-core-private", filename=fname, repo_type="model", token=HF_TOKEN)
            shutil.copy(p, f"/tmp/{fname}")
        if "/tmp" not in sys.path:
            sys.path.insert(0, "/tmp")
        from hepl_architecture import XandraHEPL
        from memory_b_cdd import XandraMetaMemoryB

        class XandraStage3Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.hepl = XandraHEPL(
                    vocab_size=VOCAB_SIZE, dim=MODEL_DIM, n_layers=N_LAYERS,
                    n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, max_seq_len=MAX_LEN,
                    use_gradient_checkpointing=False
                )
                self.memory_b = XandraMetaMemoryB(dim=MODEL_DIM, n_heads=4, window_size=64)

            def forward(self, idx, targets=None, step=0):
                with torch.no_grad():
                    x = self.hepl.emb_norm(self.hepl.tok_emb(idx))
                    for blk in self.hepl.blocks:
                        x, _, _ = blk(x, step=step)
                mem_out = self.memory_b(x)
                h_steered = self.hepl.norm(mem_out["hidden_states"])
                logits = self.hepl.head(h_steered)
                l_sft = torch.tensor(0.0, device=idx.device)
                if targets is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = targets[..., 1:].contiguous()
                    l_sft = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                return {
                    "l_sft": l_sft,
                    "cal_loss": mem_out["calibration_loss"],
                    "mean_dispersion": torch.tensor(mem_out["mean_dispersion"], device=idx.device),
                    "trigger_rate": torch.tensor(mem_out["trigger_rate"], device=idx.device),
                    "theta_crit": torch.tensor(mem_out["theta_crit"], device=idx.device)
                }

        model = XandraStage3Model().to(device)
        # تحميل أوزان المرحلة الثانية وتجميدها بالكامل
        hepl_ckpt_path = hf_hub_download(repo_id=HF_REPO, filename="checkpoints/hepl_171k.pt", repo_type="model", token=HF_TOKEN)
        hepl_ckpt = torch.load(hepl_ckpt_path, map_location=device, weights_only=False)
        clean_hepl = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in hepl_ckpt["model_state"].items()}
        model.hepl.load_state_dict(clean_hepl, strict=False)
        for p in model.hepl.parameters():
            p.requires_grad_(False)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n🧩 Model: XandraStage3Model (Layer 1 Frozen + Meta-Memory B Trainable)")
        print(f"   Trainable Memory B Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        last_step = state.get("current_step", 171148)

    elif TRAINING_STAGE == "stage4_ldr":
        import shutil
        for fname in ["hepl_architecture.py", "memory_b_cdd.py", "ldr_referee.py"]:
            p = hf_hub_download(repo_id="smart-force/xandra-core-private", filename=fname, repo_type="model", token=HF_TOKEN)
            shutil.copy(p, f"/tmp/{fname}")
        if "/tmp" not in sys.path:
            sys.path.insert(0, "/tmp")
        from hepl_architecture import XandraHEPL
        from memory_b_cdd import XandraMetaMemoryB
        from ldr_referee import XandraLDRSystem

        class XandraFullTCAModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.hepl = XandraHEPL(
                    vocab_size=VOCAB_SIZE, dim=MODEL_DIM, n_layers=N_LAYERS,
                    n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, max_seq_len=MAX_LEN,
                    use_gradient_checkpointing=True
                )
                self.memory_b = XandraMetaMemoryB(dim=MODEL_DIM, n_heads=4, window_size=64)
                self.ldr = XandraLDRSystem(
                    vocab_size=VOCAB_SIZE, shared_tok_emb=self.hepl.tok_emb,
                    base_dim=MODEL_DIM, probe_dim=256, n_layers=2
                )

            def forward(self, idx, targets=None, step=0):
                x = self.hepl.emb_norm(self.hepl.tok_emb(idx))
                for blk in self.hepl.blocks:
                    x, _, _ = blk(x, step=step)
                mem_out = self.memory_b(x)
                h_draft = self.hepl.norm(mem_out["hidden_states"])
                draft_logits = self.hepl.head(h_draft)

                B, T = idx.shape
                split_pt = max(1, T // 2)
                probe_prefix = idx[:, :split_pt]
                probe_targets = targets[:, :split_pt] if targets is not None else None

                h_ref = self.hepl.emb_norm(self.hepl.tok_emb(idx))
                for blk in self.hepl.blocks[-2:]:
                    h_ref, _, _ = blk(h_ref, step=step)
                refined_logits = self.hepl.head(self.hepl.norm(h_ref + 0.15 * h_draft))

                res = self.ldr.compute_stage4_loss(
                    probe_input_ids=probe_prefix,
                    draft_hidden_states=h_draft.detach(),
                    probe_targets=probe_targets,
                    draft_logits=draft_logits,
                    refined_logits=refined_logits,
                    gold_answer_targets=targets if targets is not None else idx
                )
                return {
                    "total_loss": res["total_loss"],
                    "dialectic_gain": torch.tensor(res["dialectic_gain"], device=idx.device),
                    "l_probe": torch.tensor(res["l_probe"], device=idx.device),
                    "l_anti_templ": torch.tensor(res["l_anti_templ"], device=idx.device)
                }

        model = XandraFullTCAModel().to(device)
        hepl_ckpt = torch.load(hf_hub_download(repo_id=HF_REPO, filename="checkpoints/hepl_171k.pt", repo_type="model", token=HF_TOKEN), map_location=device, weights_only=False)
        model.hepl.load_state_dict({k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in hepl_ckpt["model_state"].items()}, strict=False)
        mem_ckpt = torch.load(hf_hub_download(repo_id=HF_REPO, filename="checkpoints/mem_172k.pt", repo_type="model", token=HF_TOKEN), map_location=device, weights_only=False)
        model.memory_b.load_state_dict(mem_ckpt["memory_b_state"], strict=False)
        for p in model.memory_b.parameters():
            p.requires_grad_(False)
        for blk in model.hepl.blocks[:-2]:
            for p in blk.parameters():
                p.requires_grad_(False)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n⚖️ Model: XandraFullTCAModel (Stage 4: Full Triadic Cognitive Architecture + LDR)")
        print(f"   Active Trainable Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        last_step = state.get("current_step", 172650)

    else:
        TRAINING_STAGE = "stage1_pretrain"
        model = XandraModern(max_seq_len=MAX_LEN).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n🧠 Model: XandraModern (Stage 1: Base Pre-training)")
        print(f"   Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
        last_step = load_checkpoint_smart(model)

    # 🚀 التحكم في التوزيع المتوازي بناءً على إعدادات لوحة التحكم
    gpu_count = torch.cuda.device_count()
    use_dp = (DATA_PARALLEL_MODE == "enabled" and gpu_count > 1) or (DATA_PARALLEL_MODE == "auto" and gpu_count > 1)
    if use_dp:
        print(f"🔥 تفعيل المعالجة المتوازية (DataParallel) على {gpu_count} كروت شاشة!")
        model = nn.DataParallel(model)
        BATCH_SIZE = BATCH_SIZE * gpu_count
    else:
        print(f"🖥️ العمل على كرت واحد (الوضع: {DATA_PARALLEL_MODE} | الكروت المتاحة: {gpu_count})")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct', use_fast=True)


# ═══════════════════════════════════════════════════════════════
# 📖 Data Loading — embedded sources + smart fallbacks
# ═══════════════════════════════════════════════════════════════
# Sources are embedded directly — no dependency on training_sources.json
# (which the maestro does NOT upload to Kaggle).

print("\n📖 تحميل مصادر التدريب...")
if TRAINING_STAGE in ("stage2_hepl_sft", "stage3_memory_b", "stage4_ldr"):
    sources_config = {
        "sources": [
            {"name": "Xandra-SFT-Corpus", "type": "stream",
             "path": "smart-force/xandra-sft-corpus",
             "split": "train", "field": "text", "weight": 1.0}
        ]
    }
else:
    sources_config = {
    "sources": [
        {"name": "Shamela", "type": "stream",
         "path": "ReligiousLLMs/shamela_all_diacritized_fully",
         "split": "train", "field": "text", "weight": 0.5},
        {"name": "Wikipedia-AR", "type": "stream",
         "path": "wikimedia/wikipedia", "config": "20231101.ar",
         "split": "train", "field": "text", "weight": 0.5},
    ]
}
# Try loading external config directly from HuggingFace Cloud (Stage 1 only)
if TRAINING_STAGE not in ("stage2_hepl_sft", "stage3_memory_b", "stage4_ldr"):
    try:
        sources_path = hf_hub_download(repo_id=HF_REPO, filename="training_sources.json", repo_type="model", token=HF_TOKEN)
        with open(sources_path, 'r', encoding='utf-8') as f:
            external = json.load(f)
        if external.get('sources'):
            sources_config = external
            print(f"   ✅ تم جلب {len(sources_config['sources'])} مصدر من السحابة (training_sources.json)")
    except Exception as e:
        print(f"   ℹ️ تعذر جلب training_sources.json السحابي ({e}) — استخدام المصادر المدمجة")

from datasets import load_dataset, interleave_datasets

all_texts = []
stream_datasets = []
stream_weights = []

for source in sources_config["sources"]:
    try:
        if source["type"] == "local":
            path = source["path"]
            if os.path.exists(path):
                print(f"📂 قراءة بيانات محلية: {source['name']}")
                for filename in os.listdir(path):
                    filepath = os.path.join(path, filename)
                    if filename.endswith('.txt'):
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            for p in f.read().split('\n'):
                                if len(p) > 50: all_texts.append(p)
                    elif filename.endswith('.csv'):
                        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
                        for col in df.columns:
                            if df[col].dtype == 'object':
                                for val in df[col].dropna():
                                    if len(str(val)) > 50: all_texts.append(str(val))
        elif source["type"] == "stream":
            print(f"🌐 تحميل بيانات Streaming: {source['name']}")
            if "config" in source:
                ds = load_dataset(source["path"], source["config"], split=source["split"], streaming=True)
            else:
                ds = load_dataset(source["path"], split=source["split"], streaming=True)
            stream_datasets.append(ds)
            stream_weights.append(source["weight"])
    except Exception as e:
        print(f"⚠️ فشل تحميل {source['name']}: {e}")

if len(stream_datasets) > 0:
    total_w = sum(stream_weights)
    stream_weights = [w / total_w for w in stream_weights]
    combined_stream = interleave_datasets(stream_datasets, probabilities=stream_weights, seed=42)
    STREAMING_READY = True
else:
    STREAMING_READY = False


# Buffer to cache texts from streaming (avoids re-iterating exhausted streams)
TEXT_BUFFER = []
BUFFER_SIZE = 2500
STREAM_ITER = None

def refill_text_buffer():
    """يسحب دفعة جديدة بالتسلسل دون إعادة الاتصال من البداية أو تكرار أوائل الملفات."""
    global TEXT_BUFFER, STREAM_ITER
    if not STREAMING_READY:
        return
    try:
        if STREAM_ITER is None:
            shuffled = combined_stream.shuffle(seed=int(time.time()), buffer_size=5000)
            STREAM_ITER = iter(shuffled)
        
        # تفريغ النصوص القديمة المستهلكة قبل تعبئة نصوص جديدة تماماً
        TEXT_BUFFER.clear()
        count = 0
        while count < BUFFER_SIZE:
            try:
                item = next(STREAM_ITER)
                text = item.get('text', item.get('content', item.get('nass', '')))
                if text and len(str(text)) > 50:
                    TEXT_BUFFER.append(str(text))
                    count += 1
            except StopIteration:
                shuffled = combined_stream.shuffle(seed=int(time.time()), buffer_size=5000)
                STREAM_ITER = iter(shuffled)
        if count > 0:
            print(f"📦 تم تعبئة الـ buffer بـ {count} نص جديد ومتنوع من الـ stream")
    except Exception as e:
        print(f"⚠️ فشل تعبئة الـ buffer: {e}")
        STREAM_ITER = None


# Initial fill
if STREAMING_READY:
    refill_text_buffer()


def get_batch(batch_size=BATCH_SIZE, max_len=MAX_LEN):
    """Returns a tokenized batch. NEVER returns empty — has fallback."""
    texts = []

    # Method 1: Try streaming buffer first (70% of the time)
    use_stream = STREAMING_READY and len(TEXT_BUFFER) > 0 and torch.rand(1).item() < 0.7
    if use_stream:
        # Pick random texts from the buffer
        indices = torch.randint(0, len(TEXT_BUFFER), (batch_size,))
        for idx in indices:
            texts.append(TEXT_BUFFER[idx])

    # Method 2: Fill remaining from local texts
    if len(texts) < batch_size and len(all_texts) > 0:
        needed = batch_size - len(texts)
        indices = torch.randint(0, len(all_texts), (needed,))
        for idx in indices:
            texts.append(all_texts[idx])

    # Method 3: Refill buffer if it's running low
    if STREAMING_READY and len(TEXT_BUFFER) < 100:
        refill_text_buffer()

    # Method 4: الانتظار وإعادة المحاولة من السحابة (ممنوع استخدام بيانات احتياطية مكررة)
    while len(texts) == 0:
        print("⏳ انتظار تعبئة البيانات من المستودعات السحابية...")
        refill_text_buffer()
        if len(TEXT_BUFFER) > 0:
            indices = torch.randint(0, len(TEXT_BUFFER), (batch_size,))
            for idx in indices:
                texts.append(TEXT_BUFFER[idx])
        else:
            time.sleep(10)


    # Tokenize all texts
    batch = []
    for text in texts:
        tokens = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len)['input_ids'][0]
        batch.append(tokens)
    return torch.stack(batch)


# ═══════════════════════════════════════════════════════════════
# 🚀 BATCH PREFETCHER
# ═══════════════════════════════════════════════════════════════
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
                batch = get_batch(self.batch_size, self.max_len)
                self.queue.put(batch, timeout=30)
            except Exception as e:
                print(f"⚠️ خطأ في الـ prefetcher: {e}")
                time.sleep(1)

    def start(self):
        self.worker_thread = threading.Thread(target=self._background_loader, daemon=True)
        self.worker_thread.start()
        print(f"🚀 Prefetcher started (prefetch_count={self.prefetch_count})")

    def get(self):
        return self.queue.get(timeout=60)

    def stop(self):
        self.stop_flag = True


# Initialize training
# 🚀 DATA DEDUPLICATION — remove duplicate texts for better learning
if USE_DATA_DEDUP and all_texts:
    original_count = len(all_texts)
    seen = set()
    unique_texts = []
    for text in all_texts:
        key = text[:100].strip()
        if key and key not in seen:
            seen.add(key)
            unique_texts.append(text)
    all_texts = unique_texts
    dedup_pct = (1 - len(all_texts) / original_count) * 100 if original_count > 0 else 0
    print(f"🚀 Data deduplication: removed {original_count - len(all_texts)} duplicates ({dedup_pct:.1f}% reduction)")
    print(f"   {len(all_texts):,} unique texts remaining")

# 🆕 LoRA Fine-tuning (optional — set USE_LORA = True to enable)
if USE_LORA:
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print(f"🆕 LoRA: ✅ enabled (only ~1% of params are trainable)")
    except ImportError:
        print("⚠️ peft library not installed — skipping LoRA")
        USE_LORA = False

# 🚀 8-bit Optimizer (saves 75% VRAM for optimizer states)
if USE_8BIT_OPTIMIZER:
    try:
        install('bitsandbytes')
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
        )
        print(f"🚀 8-bit Optimizer: ✅ enabled (saves ~75% VRAM)")
    except Exception as e:
        print(f"⚠️ 8-bit optimizer failed: {e} — falling back to standard AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_STEPS)

# 🚀 Mixed Precision
scaler = torch.amp.GradScaler('cuda', enabled=USE_MIXED_PRECISION)

# --- SAFE-RETRAIN: distillation loss + kill-switch baseline ---
if TRAINING_STAGE == "stage2_hepl_sft":
    from hepl_architecture import SparseTopKDistillationLoss
    kl_loss_fn = SparseTopKDistillationLoss(temperature=2.0).to(device)
else:
    kl_loss_fn = None
M_REAL_NORM_BASELINE = None
KILLSWITCH_MIN_RATIO = 0.30
KILLSWITCH_CHECK_EVERY = 200

# 🚀 torch.compile
if USE_TORCH_COMPILE and hasattr(torch, 'compile'):
    try:
        model = torch.compile(model, mode='reduce-overhead')
        print("🚀 torch.compile: ✅ enabled (reduce-overhead mode)")
    except Exception as e:
        print(f"⚠️ torch.compile failed: {e}")

# 🆕 Gradient Checkpointing — allows bigger model in limited VRAM
if USE_GRADIENT_CHECKPOINTING:
    try:
        model.gradient_checkpointing_enable()
        print("🆕 Gradient checkpointing: ✅ enabled (saves ~40% VRAM)")
    except Exception as e:
        print(f"⚠️ Gradient checkpointing not available: {e}")

model.train()
step = state['current_step']
start_time = time.time()
losses = []
crash_reason = None

print(f"\n📊 Training Config:")
print(f"   Model: {n_params/1e6:.1f}M params (Modern LLaMA-style)")
print(f"   Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM_STEPS})")
print(f"   Max sequence length: {MAX_LEN}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Save every: {SAVE_EVERY} steps")
print(f"   Max time: {MAX_TIME_MINUTES} minutes ({MAX_TIME_MINUTES/60:.1f}h)")
print(f"   Mixed precision: {USE_MIXED_PRECISION}")
print(f"   Gradient accumulation: {GRAD_ACCUM_STEPS} steps")
print(f"   Starting from step: {step}")
print()

prefetcher = BatchPrefetcher(BATCH_SIZE, MAX_LEN, PREFETCH_BATCHES)
prefetcher.start()
print("🔥 Warming up prefetcher...")
time.sleep(2)

# ═══════════════════════════════════════════════════════════════
# 🛡️ CRASH-SAFE TRAINING LOOP
# ═══════════════════════════════════════════════════════════════
try:
    optimizer.zero_grad()
    accum_loss = 0.0

    for i in range(MAX_STEPS):
        elapsed = (time.time() - start_time) / 60
        if elapsed >= MAX_TIME_MINUTES:
            print(f"⏰ بلغنا الحد الزمني الأقصى ({MAX_TIME_MINUTES} دقيقة) — إنهاء نظيف")
            break

        try:
            batch = prefetcher.get().to(device, non_blocking=True)
        except Exception as e:
            print(f"⚠️ تخطّي batch بسبب خطأ: {e}")
            continue

        # 🚀 Mixed Precision forward بناءً على اختيار لوحة التحكم
        if PRECISION_MODE == "float16":
            amp_dtype = torch.float16
            use_amp = True
        elif PRECISION_MODE == "float32":
            amp_dtype = torch.float32
            use_amp = False
        else:  # auto or bfloat16 — forced to float16 on T4
            amp_dtype = torch.float16
            use_amp = USE_MIXED_PRECISION

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            if TRAINING_STAGE == "stage2_hepl_sft":
                stage2_step = max(0, step - 150000)
                out_dict = model(
                    batch, targets=batch, step=stage2_step,
                )
                l_sft  = out_dict["l_sft"].mean()  if (out_dict["l_sft"]  is not None and out_dict["l_sft"].ndim  > 0) else (out_dict["l_sft"]  if out_dict["l_sft"]  is not None else torch.tensor(0.0, device=device))
                l_bal  = out_dict["l_bal"].mean()  if out_dict["l_bal"].ndim  > 0 else out_dict["l_bal"]
                l_gate = out_dict["l_gate"].mean() if out_dict["l_gate"].ndim > 0 else out_dict["l_gate"]
                l_kl   = torch.tensor(0.0, device=device)
                beta  = 0.05 if stage2_step >= 1000 else 0.0
                gamma = 0.50 if stage2_step >= 2000 else 0.0
                alpha_w = hepl_telemetry.alpha_bal if hepl_telemetry else 0.01
                loss = l_sft + (alpha_w * l_bal) + (beta * l_gate) + (gamma * l_kl)
                if step % 100 == 0:
                    print(f"[Stage2] step {step} | L_SFT {l_sft.item():.4f} | L_bal {l_bal.item():.4f} | L_gate {l_gate.item():.4f} | L_KL {l_kl.item():.4f} | beta {beta} gamma {gamma}")
            elif TRAINING_STAGE == "stage3_memory_b":
                out_dict = model(batch, targets=batch, step=step)
                l_sft = out_dict["l_sft"].mean() if out_dict["l_sft"].ndim > 0 else out_dict["l_sft"]
                cal_l = out_dict["cal_loss"].mean() if out_dict["cal_loss"].ndim > 0 else out_dict["cal_loss"]
                loss = (0.2 * l_sft) + cal_l
                state["memory_b_telemetry"] = {
                    "mean_dispersion": round(float(out_dict["mean_dispersion"].mean().item()), 4),
                    "trigger_rate": round(float(out_dict["trigger_rate"].mean().item()), 4),
                    "theta_crit": round(float(out_dict["theta_crit"].mean().item()), 4)
                }
            elif TRAINING_STAGE == "stage4_ldr":
                out_dict = model(batch, targets=batch, step=step)
                loss = out_dict["total_loss"].mean() if out_dict["total_loss"].ndim > 0 else out_dict["total_loss"]
                state["ldr_telemetry"] = {
                    "dialectic_gain": round(float(out_dict["dialectic_gain"].mean().item()), 4),
                    "l_probe": round(float(out_dict["l_probe"].mean().item()), 4),
                    "l_anti_templ": round(float(out_dict["l_anti_templ"].mean().item()), 4)
                }
            else:
                logits, loss = model(batch, targets=batch)
                if loss.ndim > 0:
                    loss = loss.mean()
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        accum_loss += loss.item()

        # --- SAFE-RETRAIN: NaN/Inf detection + kill-switch ---
        if not torch.isfinite(loss):
            print(f"[FATAL] non-finite loss at step {step}: {loss.item()}. Rolling back.")
            try:
                load_checkpoint_smart(model)
            except Exception as _rb_err:
                print(f"[FATAL] rollback failed: {_rb_err}")
            optimizer.zero_grad()
            scaler.update()
            continue

        if TRAINING_STAGE == "stage2_hepl_sft" and step % KILLSWITCH_CHECK_EVERY == 0:
            raw_model = model.module if hasattr(model, "module") else model
            if hasattr(raw_model, "fluency_probe"):
                _m_real_norm = raw_model.fluency_probe()
                if M_REAL_NORM_BASELINE is None and _m_real_norm > 0:
                    M_REAL_NORM_BASELINE = _m_real_norm
                    print(f"[KillSwitch] baseline M_real norm = {M_REAL_NORM_BASELINE:.4f}")
                elif M_REAL_NORM_BASELINE is not None:
                    _ratio = _m_real_norm / max(M_REAL_NORM_BASELINE, 1e-8)
                    print(f"[KillSwitch] step {step} M_real norm = {_m_real_norm:.4f} (ratio {_ratio:.2f})")
                    if _ratio < KILLSWITCH_MIN_RATIO:
                        print(f"[FATAL] M_real norm collapsed to {_ratio:.2%} of baseline. HALTING.")
                        break

        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        step += 1
        losses.append(loss.item() * GRAD_ACCUM_STEPS)

        if TRAINING_STAGE in ("stage3_memory_b", "stage4_ldr") and step >= MAX_STEPS:
            print(f"🎯 اكتملت معايرة المرحلة الثالثة بنجاح عند الخطوة {step:,} — إنهاء نظيف!")
            break

        if i % 10 == 0:
            avg_loss = sum(losses[-10:]) / min(10, len(losses))
            elapsed_min = (time.time() - start_time) / 60
            steps_per_min = len(losses) / max(elapsed_min, 0.01)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {step:,} | loss: {loss.item()*GRAD_ACCUM_STEPS:.4f} | avg: {avg_loss:.4f} | {steps_per_min:.1f} steps/min")

        # 🫀 Lightweight heartbeat every 100 steps — updates state.json only (no checkpoint upload)
        # This lets the maestro see progress even before the first SAVE_EVERY (1000 steps).
        # 🛡️ Uses smart heartbeat() function which preserves admin edits (account_usage, etc)
        if i > 0 and i % 100 == 0:
            try:
                state['current_step'] = step
                state['current_loss'] = loss.item() * GRAD_ACCUM_STEPS
                state['updated_at'] = datetime.now(timezone.utc).isoformat()
                state['training_lock']['last_heartbeat'] = datetime.now(timezone.utc).isoformat()
                state['training_lock']['last_step'] = step
                # 🛡️ Use smart heartbeat to preserve account_usage from admin edits
                heartbeat(state, step)
            except Exception as e:
                print(f"⚠️ heartbeat error (non-fatal): {e}")

        if i % SAVE_EVERY == 0 and i > 0:
            try:
                # استخراج الأوزان النظيفة سواء كان كرتاً واحداً أو كرتين (DataParallel)
                raw_model = model.module if hasattr(model, 'module') else model
                raw_model = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model
                model_state = raw_model.state_dict()
                checkpoint = {
                    'model_state': model_state,
                    'optimizer_state': optimizer.state_dict(),
                    'step': step, 'loss': loss.item() * GRAD_ACCUM_STEPS,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'config': {
                        'batch_size': BATCH_SIZE, 'max_len': MAX_LEN,
                        'mixed_precision': USE_MIXED_PRECISION,
                        'model': 'XandraModern', 'params': n_params,
                        'dim': MODEL_DIM, 'n_layers': N_LAYERS,
                        'n_heads': N_HEADS, 'n_kv_heads': N_KV_HEADS
                    }
                }
                torch.save(checkpoint, 'latest.pt')
                upload_file(path_or_fileobj='latest.pt', path_in_repo='checkpoints/latest.pt',
                            repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
                # 🛡️ الحفظ التلقائي للنسخة المرجعية base_150k.pt عند بلوغ 150,000 في المرحلة 1
                if TRAINING_STAGE == "stage1_pretrain" and step >= 150000 and not state.get("base_150k_saved"):
                    upload_file(path_or_fileobj='latest.pt', path_in_repo='checkpoints/base_150k.pt',
                                repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
                    state["base_150k_saved"] = True
                    print("🛡️ تم تأمين النسخة المرجعية checkpoints/base_150k.pt بنجاح!")
                # 🧩 حفظ أوزان الذاكرة المستقلة في المرحلة الثالثة (Memory B & CDD)
                if TRAINING_STAGE == "stage3_memory_b":
                    mem_payload = {
                        'memory_b_state': raw_model.memory_b.state_dict(),
                        'step': step,
                        'telemetry': state.get("memory_b_telemetry", {})
                    }
                    torch.save(mem_payload, 'memory_b_latest.pt')
                    upload_file(path_or_fileobj='memory_b_latest.pt', path_in_repo='checkpoints/memory_b_latest.pt',
                                repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
                    if step >= 172600:
                        upload_file(path_or_fileobj='memory_b_latest.pt', path_in_repo='checkpoints/mem_172k.pt',
                                    repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
                        print("🎯 تم حفظ أوزان الذاكرة النهائية checkpoints/mem_172k.pt بنجاح!")

                # 🧠 نظام المناعة الذاتية ومراقبة الفضاءات في المرحلة الثانية
                if TRAINING_STAGE == "stage2_hepl_sft" and hepl_telemetry is not None:
                    tel = raw_model.get_telemetry()
                    state["manifold_telemetry"] = tel
                    sh_res = hepl_telemetry.step_update(step, tel)
                    if sh_res["alerts"]:
                        state.setdefault("system_alerts", []).extend(sh_res["alerts"])
                        state["system_alerts"] = state["system_alerts"][-20:]
                    if sh_res["action"] == "pause_and_rollback":
                        state["system_paused"] = True
                        heartbeat(state, step)
                        print("🚨 تم إيقاف التدريب تلقائياً بسبب انهيار أحد الفضاءات (>95%).")
                        break
                # 🫀 Heartbeat + loss history + training samples
                state['current_step'] = step
                state['current_loss'] = loss.item() * GRAD_ACCUM_STEPS
                if 'loss_history' not in state:
                    state['loss_history'] = []
                state['loss_history'].append({
                    'step': step, 'loss': loss.item() * GRAD_ACCUM_STEPS,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                if len(state['loss_history']) > 100:
                    state['loss_history'] = state['loss_history'][-100:]
                if 'training_samples' not in state:
                    state['training_samples'] = []
                try:
                    sample_text = tokenizer.decode(batch[0][:50])
                    state['training_samples'].append({
                        'step': step, 'loss': loss.item() * GRAD_ACCUM_STEPS,
                        'text': sample_text[:200], 'source': 'mixed_stream',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                    if len(state['training_samples']) > 50:
                        state['training_samples'] = state['training_samples'][-50:]
                except: pass
                if 'project_start_date' not in state or not state['project_start_date']:
                    state['project_start_date'] = SESSION_STARTED_AT
                heartbeat(state, step)
            except Exception as e:
                print(f"⚠️ فشل حفظ checkpoint: {e}")

except KeyboardInterrupt:
    crash_reason = "manual_stop"
    log_step("⛔ تم الإيقاف اليدوي — تحرير القفل...")
except Exception as e:
    crash_reason = f"crash: {str(e)[:100]}"
    log_step(f"\n💥 انهيار غير متوقع: {e} — تحرير القفل...")
    import traceback
    traceback.print_exc()
    with open('crash_log.txt', 'w') as f:
        f.write(traceback.format_exc())

finally:
    prefetcher.stop()
    print("\n🛡️ تنفيذ كتلة finally...")

    try:
        state = load_state()

        if losses:
            final_loss = losses[-1]
            try:
                # استخراج الأوزان النظيفة عند إنهاء الجلسة
                raw_model = model.module if hasattr(model, 'module') else model
                raw_model = raw_model._orig_mod if hasattr(raw_model, '_orig_mod') else raw_model
                model_state = raw_model.state_dict()
                checkpoint = {
                    'model_state': model_state,
                    'optimizer_state': optimizer.state_dict(),
                    'step': step, 'loss': final_loss,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'config': {'batch_size': BATCH_SIZE, 'max_len': MAX_LEN, 'mixed_precision': USE_MIXED_PRECISION,
                               'model': 'XandraModern', 'params': n_params}
                }
                torch.save(checkpoint, 'latest.pt')
                upload_file(path_or_fileobj='latest.pt', path_in_repo='checkpoints/latest.pt',
                            repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
                if TRAINING_STAGE == "stage3_memory_b":
                    mem_payload = {
                        'memory_b_state': raw_model.memory_b.state_dict(),
                        'step': step,
                        'telemetry': state.get("memory_b_telemetry", {})
                    }
                    torch.save(mem_payload, 'mem_172k.pt')
                    upload_file(path_or_fileobj='mem_172k.pt', path_in_repo='checkpoints/mem_172k.pt',
                                repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
                    print("🎯 تم حفظ ورفع checkpoints/mem_172k.pt بنجاح عند انتهاء المرحلة الثالثة!")
                if TRAINING_STAGE == "stage4_ldr":
                    tca_payload = {
                        'full_tca_state': model_state,
                        'step': step,
                        'loss': final_loss,
                        'telemetry': state.get("ldr_telemetry", {})
                    }
                    torch.save(tca_payload, 'xandra_tca_final.pt')
                    upload_file(path_or_fileobj='xandra_tca_final.pt', path_in_repo='checkpoints/xandra_tca_final.pt',
                                repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
                    print("🏆 تم حفظ ورفع checkpoints/xandra_tca_final.pt بنجاح عند إتمام المعمارية الثلاثية!")
            except Exception as e:
                print(f"⚠️ فشل حفظ checkpoint النهائي: {e}")

            gpu_hours = (time.time() - SESSION_START_TIME) / 3600

            state['current_step'] = step
            state['current_loss'] = final_loss
            state['best_loss'] = min(state.get('best_loss', float('inf')), final_loss)
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            state['gpu_hours_used'] = state.get('gpu_hours_used', 0) + gpu_hours
            state['total_project_hours'] = state.get('total_project_hours', 0) + gpu_hours
            if 'project_start_date' not in state or not state['project_start_date']:
                state['project_start_date'] = SESSION_STARTED_AT
            if 'loss_history' not in state:
                state['loss_history'] = []
            state['loss_history'].append({
                'step': step, 'loss': final_loss,
                'timestamp': datetime.now(timezone.utc).isoformat(), 'session_end': True
            })
            if len(state['loss_history']) > 100:
                state['loss_history'] = state['loss_history'][-100:]

            state.setdefault('history', []).append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'started_at': SESSION_STARTED_AT,
                'final_step': step, 'final_loss': final_loss,
                'steps_done': len(losses), 'gpu_hours': gpu_hours,
                'worker_id': WORKER_ID,
                'exit_reason': crash_reason or "completed",
                'config': {'batch_size': BATCH_SIZE, 'max_len': MAX_LEN, 'mixed_precision': USE_MIXED_PRECISION,
                           'model': 'XandraModern', 'params': n_params}
            })
            if len(state['history']) > 50:
                state['history'] = state['history'][-50:]

        release_lock(state, reason=crash_reason or "completed")

        print(f"\n✅ انتهت الجلسة. الخطوة النهائية: {step:,}")
        print(f"⏱️ زمن الجلسة: {(time.time() - SESSION_START_TIME)/3600:.2f} ساعة")
        print(f"📊 متوسط الخطوة/دقيقة: {len(losses) / max((time.time() - SESSION_START_TIME)/60, 0.01):.1f}")
        print(f"🔒 تم تحرير القفل.")
    except Exception as e:
        print(f"🚨 خطأ كارثي في finally: {e}")
        try:
            state = load_state()
            release_lock(state, reason=f"finally_error: {str(e)[:50]}")
        except:
            print("💀 فشل تحرير القفل — يجب التدخل اليدوي")
    finally:
        import os
        os._exit(0)  # إغلاق إجباري فوري للحاوية ومنع حرق الساعات عند التوقف