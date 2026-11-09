"""
╔═══════════════════════════════════════════════════════════════╗
║   Xandra Hope - Kaggle Worker (Crash-Safe Edition v4.2)       ║
║   Copyright (c) 2026 Ahmed Sayed Kamel Mahmoud &               ║
║                  Александра Евгеньевна Острикова              ║
║   All Rights Reserved. License: XHNSL v1.0                   ║
╚═══════════════════════════════════════════════════════════════╝

UPGRADE NOTES (v4.1 → v4.2):
- Wrapped the entire training loop in try/finally so the lock is ALWAYS released
  (previously a crash left the lock stuck for 8 hours)
- Added heartbeat updates: every 500 steps the worker refreshes `locked_at`
  in state.json so the Maestro can confirm the session is alive
- Added `started_at` to history entries so the Maestro can compute real gaps
- Reduced LOCK_TIMEOUT_HOURS to 12 (Maestro now does real status checks)
- Added KeyboardInterrupt handling for clean shutdown on manual stop
"""

# هذا المتغير سيتم استبداله تلقائياً بواسطة المايسترو باسم المستخدم الحقيقي
__KAGGLE_USERNAME__ = "manual_run"

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta, timezone
from huggingface_hub import HfApi, hf_hub_download, upload_file
import glob
import pandas as pd

SESSION_START_TIME = time.time()
SESSION_STARTED_AT = datetime.now(timezone.utc).isoformat()

print("=" * 60)
print("🎯 Xandra Hope - Kaggle Worker (Crash-Safe v4.2)")
print(f"👤 Running on Account: {__KAGGLE_USERNAME__}")
print(f"⏰ Session started at: {SESSION_STARTED_AT}")
print("Copyright (c) 2026 Ahmed Sayed Kamel Mahmoud &")
print("               Александра Евгеньевна Острикова")
print("=" * 60)

try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    HF_TOKEN = secrets.get_secret("HF_TOKEN")
    HF_REPO = secrets.get_secret("HF_REPO")
except:
    HF_TOKEN = os.environ.get("HF_TOKEN", "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")
    HF_REPO = os.environ.get("HF_REPO", "smart-force/xandra-training-state")

os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HF_HOME'] = '/tmp/huggingface'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# القفل الذكي — مُخفّض إلى 12 ساعة (المايسترو سيتحقق فعلياً)
LOCK_TIMEOUT_HOURS = 12
WORKER_ID = f"kaggle-{__KAGGLE_USERNAME__}"
HEARTBEAT_EVERY_STEPS = 500  # تحديث locked_at كل 500 خطوة


def acquire_lock():
    try:
        state_path = hf_hub_download(repo_id=HF_REPO, filename="state.json", repo_type="model", token=HF_TOKEN)
        with open(state_path) as f: state = json.load(f)
        lock = state.get('training_lock', {})
        if lock.get('is_locked'):
            locked_by = lock.get('locked_by', '')
            # إذا كان نفس الحساب يحمل القفل، نسمح بإعادة الاكتساب (resume)
            if WORKER_ID in locked_by:
                print(f"ℹ️ هذا الحساب يحمل القفل بالفعل — متابعة")
            else:
                expires_at = lock.get('expires_at')
                if expires_at:
                    try:
                        exp_time = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                        if datetime.now(timezone.utc) > exp_time:
                            print(f"⏰ القفل منتهي الصلاحية — اكتسابه من جديد")
                        else:
                            print(f"⛔ القفل مشغول بواسطة {locked_by} — خروج")
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
    try:
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
    """Refreshes locked_at so Maestro knows we're alive."""
    try:
        state_dict['training_lock']['locked_at'] = datetime.now(timezone.utc).isoformat()
        state_dict['training_lock']['last_step'] = step
        with open('state.json', 'w') as f: json.dump(state_dict, f, indent=2, ensure_ascii=False)
        upload_file(path_or_fileobj='state.json', path_in_repo='state.json', repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
    except: pass


class XandraTiny(nn.Module):
    def __init__(self, vocab_size=151646, dim=256, n_layers=4, n_heads=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(128, dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=1024, batch_first=True, activation='gelu')
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        for block in self.blocks: x = block(x, src_mask=mask)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None: loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def load_state():
    try:
        path = hf_hub_download(repo_id=HF_REPO, filename="state.json", repo_type="model", token=HF_TOKEN)
        with open(path) as f: return json.load(f)
    except:
        return {"current_step": 0, "target_steps": 500000, "current_loss": None,
                "history": [], "best_loss": float('inf'), "updated_at": None,
                "gpu_hours_used": 0, "training_lock": {"is_locked": False}}


def load_checkpoint(model):
    try:
        path = hf_hub_download(repo_id=HF_REPO, filename="checkpoints/latest.pt", repo_type="model", token=HF_TOKEN)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        return checkpoint['step']
    except: return 0


if not acquire_lock():
    print("\n⛔ لا يمكن التدريب الآن. جلسة أخرى قيد التشغيل.")
    sys.exit(0)

model = XandraTiny().to(device)
state = load_state()
last_step = load_checkpoint(model)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct', use_fast=True)

# ===================================================
# نظام تحميل مصادر التدريب من ملف JSON خارجي
# ===================================================
print("\n📖 قراءة ملف مصادر التدريب (training_sources.json)...")
try:
    with open('training_sources.json', 'r') as f:
        sources_config = json.load(f)
    print(f"✅ تم تحميل {len(sources_config['sources'])} مصدر تدريب.")
except:
    print("⚠️ لم يتم العثور على training_sources.json، استخدام الإعدادات الافتراضية.")
    sources_config = {
        "sources": [
            {"name": "Shamela", "type": "stream", "path": "ReligiousLLMs/shamela_all_diacritized_fully", "split": "train", "field": "text", "weight": 0.4},
            {"name": "Wikipedia", "type": "stream", "path": "wikimedia/wikipedia", "config": "20231101.ar", "split": "train", "field": "text", "weight": 0.3},
            {"name": "Tashkeela", "type": "local", "path": "/kaggle/input/tashkeela", "weight": 0.3}
        ]
    }

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
    combined_stream = interleave_datasets(stream_datasets, probabilities=stream_weights, seed=42)
    stream_iter = iter(combined_stream)
    STREAMING_READY = True
else:
    STREAMING_READY = False


def get_batch(batch_size=32, max_len=128):
    texts = []
    use_stream = STREAMING_READY and (len(all_texts) == 0 or torch.rand(1).item() < 0.3)
    if use_stream:
        for item in stream_iter:
            if len(texts) >= batch_size: break
            text = item.get('text', item.get('content', item.get('nass', '')))
            if text and len(str(text)) > 50: texts.append(str(text))
    if len(texts) < batch_size and len(all_texts) > 0:
        indices = torch.randint(0, len(all_texts), (batch_size - len(texts),))
        for idx in texts: texts.append(all_texts[idx])

    batch = []
    for text in texts:
        tokens = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len)['input_ids'][0]
        batch.append(tokens)
    return torch.stack(batch)


MAX_STEPS = 1000000
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
SAVE_EVERY = 500
MAX_TIME_MINUTES = 650

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_STEPS)

model.train()
step = state['current_step']
start_time = time.time()
losses = []
crash_reason = None

# ═══════════════════════════════════════════════════════════════
# 🛡️ CRASH-SAFE TRAINING LOOP
# The try/finally guarantees the lock is ALWAYS released, even if:
#  - OOM error
#  - Network failure to Hugging Face
#  - CUDA error
#  - Manual stop (KeyboardInterrupt)
#  - Any unhandled exception
# ═══════════════════════════════════════════════════════════════
try:
    for i in range(MAX_STEPS):
        elapsed = (time.time() - start_time) / 60
        if elapsed >= MAX_TIME_MINUTES:
            print(f"⏰ بلغنا الحد الزمني الأقصى ({MAX_TIME_MINUTES} دقيقة) — إنهاء نظيف")
            break

        try:
            batch = get_batch(BATCH_SIZE).to(device)
        except Exception as e:
            print(f"⚠️ تخطّي batch بسبب خطأ: {e}")
            continue

        logits, loss = model(batch, targets=batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # حماية من الانفجار
        optimizer.step()
        scheduler.step()

        step += 1
        losses.append(loss.item())

        if i % 10 == 0:
            avg_loss = sum(losses[-10:]) / min(10, len(losses))
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {step} | loss: {loss.item():.4f} | avg: {avg_loss:.4f}")

        # Save checkpoint + heartbeat every SAVE_EVERY steps
        if i % SAVE_EVERY == 0 and i > 0:
            try:
                checkpoint = {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'step': step, 'loss': loss.item(),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                torch.save(checkpoint, 'latest.pt')
                upload_file(path_or_fileobj='latest.pt', path_in_repo='checkpoints/latest.pt',
                            repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
                # 🫀 Heartbeat — أبلغ المايسترو أننا ما زلنا أحياء
                state['current_step'] = step
                state['current_loss'] = loss.item()
                heartbeat(state, step)
            except Exception as e:
                print(f"⚠️ فشل حفظ checkpoint: {e}")

except KeyboardInterrupt:
    crash_reason = "manual_stop"
    print("\n⛔ تم الإيقاف اليدوي (KeyboardInterrupt) — تحرير القفل...")
except Exception as e:
    crash_reason = "crash"
    print(f"\n💥 انهيار غير متوقع: {e} — تحرير القفل قبل الخروج...")
    import traceback
    traceback.print_exc()

finally:
    # 🔒 هذا الكتلة تُنفّذ دائماً — نجاح أو فشل أو إيقاف يدوي
    print("\n🛡️ تنفيذ كتلة finally لتحديث state.json وتحرير القفل...")

    try:
        # إعادة قراءة أحدث state (لأن المايسترو قد يكون حدّثه أثناء جلساتنا)
        state = load_state()

        if losses:
            final_loss = losses[-1]
            try:
                checkpoint = {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'step': step, 'loss': final_loss,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                torch.save(checkpoint, 'latest.pt')
                upload_file(path_or_fileobj='latest.pt', path_in_repo='checkpoints/latest.pt',
                            repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
            except Exception as e:
                print(f"⚠️ فشل حفظ checkpoint النهائي: {e}")

            gpu_hours = (time.time() - SESSION_START_TIME) / 3600

            state['current_step'] = step
            state['current_loss'] = final_loss
            state['best_loss'] = min(state.get('best_loss', float('inf')), final_loss)
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            state['gpu_hours_used'] = state.get('gpu_hours_used', 0) + gpu_hours

            # ⭐ نضيف started_at حتى يتمكن المايسترو من حساب الفجوات الزمنية
            state.setdefault('history', []).append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'started_at': SESSION_STARTED_AT,
                'final_step': step, 'final_loss': final_loss,
                'steps_done': len(losses), 'gpu_hours': gpu_hours,
                'worker_id': WORKER_ID,
                'exit_reason': crash_reason or "completed"
            })
            if len(state['history']) > 50:
                state['history'] = state['history'][-50:]

        # تحرير القفل دائماً
        release_lock(state, reason=crash_reason or "completed")

        print(f"\n✅ انتهت الجلسة. الخطوة النهائية: {step}")
        print(f"⏱️ زمن الجلسة: {(time.time() - SESSION_START_TIME)/3600:.2f} ساعة")
        print(f"🔒 تم تحرير القفل. المايسترو يمكنه إطلاق الجلسة التالية فوراً.")
    except Exception as e:
        print(f"🚨 خطأ كارثي في finally: {e}")
        # محاولة أخيرة لتحرير القفل
        try:
            state = load_state()
            release_lock(state, reason=f"finally_error: {str(e)[:50]}")
        except:
            print("💀 فشل تحرير القفل حتى في المحاولة الأخيرة — يجب التدخل اليدوي")
