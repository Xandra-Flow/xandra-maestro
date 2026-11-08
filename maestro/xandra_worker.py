"""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   Xandra Hope - Kaggle Worker (Real Data Edition)             ║
║   Copyright (c) 2026 Ahmed Sayed Kamel Mahmoud &               ║
║                  Александра Евгеньевна Острикова              ║
║   All Rights Reserved.                                         ║
║                                                               ║
║   License: XHNSL v1.0                                         ║
║   Contact: ahmed.kamel@xandraflow.com                         ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""

# ===================================================
# 1. تثبيت المكتبات
# ===================================================
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

install('torch')
install('transformers')
install('huggingface_hub')
install('accelerate')
install('datasets')

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
from huggingface_hub import HfApi, hf_hub_download, upload_file

print("=" * 60)
print("🎯 Xandra Hope - Kaggle Worker (Real Data Edition)")
print("Copyright (c) 2026 Ahmed Sayed Kamel Mahmoud &")
print("               Александра Евгеньевна Острикова")
print("=" * 60)

# ===================================================
# 2. قراءة الـ Tokens من Kaggle Secrets
# ===================================================
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

print(f"📁 HF Repo: {HF_REPO}")
print(f"🔑 Token: {HF_TOKEN[:10]}...{HF_TOKEN[-4:]}")

# ===================================================
# 3. فحص GPU
# ===================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("❌ لا يوجد GPU!")

# ===================================================
# 4. آلية القفل الذكي (Smart Lock)
# ===================================================
LOCK_TIMEOUT_HOURS = 8
WORKER_ID = f"kaggle-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

def acquire_lock():
    try:
        state_path = hf_hub_download(repo_id=HF_REPO, filename="state.json", repo_type="model", token=HF_TOKEN)
        with open(state_path) as f:
            state = json.load(f)
        
        lock = state.get('training_lock', {})
        if lock.get('is_locked'):
            expires_at = lock.get('expires_at')
            if expires_at:
                expires_time = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                if datetime.now(expires_time.tzinfo) > expires_time:
                    print("⚠️ القفل منتهي الصلاحية، سيتم تجاوزه")
                else:
                    print(f"❌ النظام مشغول بواسطة: {lock.get('locked_by')}")
                    return False
        
        state['training_lock'] = {
            'is_locked': True, 'locked_by': WORKER_ID,
            'locked_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=LOCK_TIMEOUT_HOURS)).isoformat()
        }
        
        with open('state.json', 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        upload_file(path_or_fileobj='state.json', path_in_repo='state.json', repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
        print(f"✅ تم الحصول على القفل (ID: {WORKER_ID})")
        return True
    except Exception as e:
        print(f"⚠️ لا توجد حالة سابقة، بدء جديد: {e}")
        return True

def release_lock(state_dict):
    try:
        state_dict['training_lock'] = {'is_locked': False, 'locked_by': None, 'locked_at': None, 'expires_at': None}
        with open('state.json', 'w') as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)
        upload_file(path_or_fileobj='state.json', path_in_repo='state.json', repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
        print("🔓 تم تحرير القفل")
    except Exception as e:
        print(f"⚠️ فشل تحرير القفل: {e}")

# ===================================================
# 5. معمارية Xandra-Tiny
# ===================================================
class XandraTiny(nn.Module):
    def __init__(self, vocab_size=151646, dim=256, n_layers=4, n_heads=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(128, dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=n_heads, dim_feedforward=1024,
                batch_first=True, activation='gelu'
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        for block in self.blocks:
            x = block(x, src_mask=mask)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ===================================================
# 6. تحميل آخر checkpoint
# ===================================================
def load_state():
    try:
        path = hf_hub_download(repo_id=HF_REPO, filename="state.json", repo_type="model", token=HF_TOKEN)
        with open(path) as f:
            return json.load(f)
    except:
        return {
            "current_step": 0, "target_steps": 500000,
            "current_loss": None, "history": [],
            "best_loss": float('inf'), "updated_at": None,
            "gpu_hours_used": 0, "training_lock": {"is_locked": False}
        }

def load_checkpoint(model):
    try:
        path = hf_hub_download(repo_id=HF_REPO, filename="checkpoints/latest.pt", repo_type="model", token=HF_TOKEN)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        print(f"✅ تحميل checkpoint من الخطوة {checkpoint['step']}")
        return checkpoint['step']
    except:
        print("🆕 بدء تدريب من الصفر")
        return 0

# ===================================================
# 7. الحصول على القفل
# ===================================================
if not acquire_lock():
    print("\n⛔ لا يمكن التدريب الآن. جلسة أخرى قيد التشغيل.")
    sys.exit(0)

# ===================================================
# 8. بناء النموذج وتحميل البيانات الحقيقية
# ===================================================
print("\n🏗️ بناء Xandra-Tiny...")
model = XandraTiny().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"📊 الباراميترات: {n_params:,} ({n_params/1e6:.2f}M)")

state = load_state()
last_step = load_checkpoint(model)
print(f"📊 الخطوة الحالية: {state['current_step']}")

print("\n📚 تحميل Tokenizer...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct', use_fast=True)

# تحميل البيانات العربية الحقيقية من Hugging Face
print("\n🌐 تحميل البيانات العربية الحقيقية من Hugging Face...")
from datasets import load_dataset

try:
    # محاولة 1: wikimedia/wikipedia (النسخة الجديدة)
    print("⏳ تحميل ويكيبيديا العربية (wikimedia/wikipedia)...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train[:5000]")
    texts = [item['text'] for item in dataset]
    print(f"✅ تم تحميل {len(texts)} مقال عربي")
except Exception as e1:
    print(f"⚠️ فشل wikimedia/wikipedia: {e1}")
    try:
        # محاولة 2: oscar (نصوص عربية من الإنترنت)
        print("⏳ تحميل OSCAR العربية...")
        dataset = load_dataset("oscar-corpus/OSCAR-2301", "ar", split="train[:5000]")
        texts = [item['text'] for item in dataset]
        print(f"✅ تم تحميل {len(texts)} نص عربي")
    except Exception as e2:
        print(f"⚠️ فشل OSCAR: {e2}")
        try:
            # محاولة 3: FineWeb Arabic
            print("⏳ تحميل FineWeb العربية...")
            dataset = load_dataset("HuggingFaceFW/fineweb", "arb_Arab", split="train[:5000]")
            texts = [item['text'] for item in dataset]
            print(f"✅ تم تحميل {len(texts)} نص عربي")
        except Exception as e3:
            print(f"⚠️ فشل FineWeb: {e3}")
            print("⏳ تحميل بيانات احتياطية...")
            texts = [
                'العمارة الإسلامية فن راقٍ يمتد عبر القرون',
                'البرمجة العكسية تحليل البرمجيات لفهم آلية عملها',
                'الهندسة العكسية تستخرج التصميم من المنتج النهائي',
                'اللغة العربية لغة غنية بالمفردات والتراكيب',
                'العلوم الرياضية أساس التطور التكنولوجي',
                'الحضارة المصرية القديمة من أعرق الحضارات',
                'الطب عند العرب كان متقدماً في العصور الوسطى',
                'الفلسفة العربية أثرت في الفكر العالمي',
                'الكيمياء نشأت عند العرب والمسلمين',
                'الفلك علم درسه العرب بعمق',
            ] * 100
            print(f"✅ تم تحميل {len(texts)} نص احتياطي")

# تجهيز البيانات للتدريب
def encode_texts(texts, tokenizer, max_len=128):
    """تحويل النصوص إلى tokens"""
    all_tokens = []
    for text in texts:
        # نأخذ أول 128 توكن من كل مقال
        tokens = tokenizer(text, return_tensors='pt', padding='max_length',
                          truncation=True, max_length=max_len)['input_ids'][0]
        all_tokens.append(tokens)
    return torch.stack(all_tokens)

print("\n⚙️ معالجة البيانات...")
data = encode_texts(texts, tokenizer).to(device)
print(f"✅ شكل البيانات النهائي: {data.shape}")

# ===================================================
# 9. حلقة التدريب
# ===================================================
MAX_STEPS = 500
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
SAVE_EVERY = 50
MAX_TIME_MINUTES = 700

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_STEPS)

model.train()
step = state['current_step']
start_time = time.time()
losses = []

print(f"\n🚀 بدء التدريب من الخطوة {step}")
print("=" * 60)

for i in range(MAX_STEPS):
    elapsed = (time.time() - start_time) / 60
    if elapsed >= MAX_TIME_MINUTES:
        print(f"⏰ انتهى الوقت ({elapsed:.1f} دقيقة)")
        break
    
    indices = torch.randint(0, len(data), (BATCH_SIZE,))
    batch = data[indices]
    
    logits, loss = model(batch, targets=batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    step += 1
    losses.append(loss.item())
    
    if i % 10 == 0:
        avg_loss = sum(losses[-10:]) / min(10, len(losses))
        progress = (i / MAX_STEPS) * 100
        print(f"[{datetime.now().strftime('%H:%M:%S')}] خطوة {step} | "
              f"loss: {loss.item():.4f} | avg: {avg_loss:.4f} | {progress:.1f}%")
    
    if i % SAVE_EVERY == 0 and i > 0:
        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'step': step,
            'loss': loss.item(),
            'timestamp': datetime.now().isoformat(),
            'authors': 'Ahmed Sayed Kamel Mahmoud & Александра Евгеньевна Острикова',
            'copyright': '© 2026 Xandra Flow'
        }
        torch.save(checkpoint, 'latest.pt')
        upload_file(
            path_or_fileobj='latest.pt',
            path_in_repo='checkpoints/latest.pt',
            repo_id=HF_REPO,
            repo_type="model",
            token=HF_TOKEN
        )
        print(f"💾 تم حفظ checkpoint عند الخطوة {step}")

print("=" * 60)
print(f"✅ انتهى التدريب عند الخطوة {step}")

# ===================================================
# 10. حفظ النتائج النهائية وتحرير القفل
# ===================================================
if losses:
    final_loss = losses[-1]
    
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'step': step,
        'loss': final_loss,
        'timestamp': datetime.now().isoformat(),
        'authors': 'Ahmed Sayed Kamel Mahmoud & Александра Евгеньевна Острикова',
        'copyright': '© 2026 Xandra Flow'
    }
    torch.save(checkpoint, 'latest.pt')
    upload_file(
        path_or_fileobj='latest.pt',
        path_in_repo='checkpoints/latest.pt',
        repo_id=HF_REPO,
        repo_type="model",
        token=HF_TOKEN
    )
    print(f"💾 تم حفظ checkpoint نهائي عند الخطوة {step}")
    
    gpu_hours = (time.time() - start_time) / 3600
    
    state['current_step'] = step
    state['current_loss'] = final_loss
    state['best_loss'] = min(state['best_loss'], final_loss)
    state['updated_at'] = datetime.now().isoformat()
    state['gpu_hours_used'] = state.get('gpu_hours_used', 0) + gpu_hours
    state['history'].append({
        'timestamp': datetime.now().isoformat(),
        'final_step': step,
        'final_loss': final_loss,
        'steps_done': len(losses),
        'gpu_hours': gpu_hours,
        'worker_id': WORKER_ID
    })
    
    if len(state['history']) > 50:
        state['history'] = state['history'][-50:]
    
    state['training_lock'] = {
        'is_locked': False, 'locked_by': None,
        'locked_at': None, 'expires_at': None
    }
    
    with open('state.json', 'w') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    
    upload_file(
        path_or_fileobj='state.json',
        path_in_repo='state.json',
        repo_id=HF_REPO,
        repo_type="model",
        token=HF_TOKEN
    )
    
    print(f"\n📊 تم تحديث state.json")
    print(f"📈 الخطوة النهائية: {step}")
    print(f"📉 الخسارة النهائية: {final_loss:.4f}")
    print(f"🏆 أفضل خسارة: {state['best_loss']:.4f}")
    print(f"⏱️ ساعات GPU المستهلكة: {gpu_hours:.2f}")
    print("🔓 تم تحرير القفل")
else:
    print("⚠️ لم يتم التدريب")
    state = load_state()
    release_lock(state)

print("\n" + "=" * 60)
print("© 2026 Xandra Flow - All Rights Reserved")
print("✅ انتهى! اغلق المتصفح بأمان.")
