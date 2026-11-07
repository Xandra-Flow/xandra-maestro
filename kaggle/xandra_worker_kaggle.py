"""
Xandra Hope - Kaggle Training Notebook
=======================================
هذا الـ Notebook يعمل على Kaggle (مجاني 30 ساعة أسبوعياً)

طريقة الاستخدام:
1. اذهب إلى: https://www.kaggle.com/code
2. New Notebook
3. الإعدادات (يمين): Accelerator → GPU T4 x2 | Internet → On
4. Add-ons → Secrets:
   - HF_TOKEN: (ضع الـ Token الخاص بك)
   - HF_REPO: smart-force/xandra-training-state
5. انسخ هذا الكود في خلية واحدة
6. Run All (Ctrl+F9)
"""

# ====================================================
# تثبيت المكتبات والاستيراد
# ====================================================
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

install('torch')
install('transformers')
install('huggingface_hub')
install('accelerate')

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download, upload_file

print("=" * 60)
print("🎯 Xandra Hope Training Notebook (Kaggle Edition)")
print("=" * 60)

# ====================================================
# قراءة الـ Tokens من Kaggle Secrets
# ====================================================
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()

HF_TOKEN = secrets.get_secret("HF_TOKEN")
HF_REPO = secrets.get_secret("HF_REPO")

os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HF_HOME'] = '/tmp/huggingface'

print(f"📁 HF Repo: {HF_REPO}")
print(f"🔑 Token: {HF_TOKEN[:10]}...{HF_TOKEN[-4:]}")

# فحص GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🎮 GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    device = 'cuda'
else:
    print("❌ لا يوجد GPU! فعّل GPU T4 من الإعدادات")
    device = 'cpu'

# ====================================================
# تعريف معمارية Xandra-Tiny
# ====================================================
class XandraTinyConfig:
    vocab_size = 32000
    dim = 256
    n_layers = 4
    n_heads = 4
    ff_dim = 1024
    max_seq_len = 128

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiheadAttention(config.dim, config.n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.dim)
        )
        self.norm1 = nn.LayerNorm(config.dim)
        self.norm2 = nn.LayerNorm(config.dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

class XandraTiny(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None: config = XandraTinyConfig()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.dim)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks: x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

model = XandraTiny().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"✅ تم بناء نموذج Xandra-Tiny")
print(f"📊 الباراميترات: {n_params:,} ({n_params/1e6:.2f}M)")

# ====================================================
# تحميل آخر checkpoint
# ====================================================
def load_state():
    try:
        path = hf_hub_download(repo_id=HF_REPO, filename="state.json", repo_type="model", token=HF_TOKEN)
        with open(path) as f: return json.load(f)
    except Exception as e:
        print(f"🆕 بدء تدريب جديد: {e}")
        return {"current_step": 0, "target_steps": 500000, "current_loss": None, "history": [], "best_loss": float('inf'), "updated_at": None}

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

state = load_state()
last_step = load_checkpoint(model)
print(f"📊 الخطوة الحالية: {state['current_step']}")

# ====================================================
# بيانات التدريب (تجريبية)
# ====================================================
from transformers import AutoTokenizer

print("📚 تحميل Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct', use_fast=True)

training_texts = [
    'العمارة الإسلامية فن راقٍ يمتد عبر القرون',
    'الزراعة في مصر القديمة كانت متقدمة جداً',
    'الصناعة تحتاج إلى معرفة وخبرة وتدريب',
    'اللغة العربية لغة غنية بالمفردات والتراكيب',
    'العلوم الرياضية أساس التطور التكنولوجي',
    'الحضارة المصرية القديمة من أعرق الحضارات',
    'الطب عند العرب كان متقدماً في العصور الوسطى',
    'الفلسفة العربية أثرت في الفكر العالمي',
    'الكيمياء نشأت عند العرب والمسلمين',
    'الفلك علم درسه العرب بعمق',
] * 100

def encode_batch(texts, max_len=128):
    batch = []
    for text in texts:
        tokens = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len)['input_ids'][0]
        batch.append(tokens)
    return torch.stack(batch)

data = encode_batch(training_texts).to(device)
print(f"✅ شكل البيانات: {data.shape}")

# ====================================================
# حلقة التدريب
# ====================================================
import time

MAX_STEPS = 500
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
SAVE_EVERY = 50
MAX_TIME_MINUTES = 700  # ~12 ساعة - احتياط

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
model.train()

step = state['current_step']
start_time = time.time()
losses = []

print(f"🚀 بدء التدريب من الخطوة {step}")
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
    
    step += 1
    losses.append(loss.item())
    
    if i % 10 == 0:
        avg_loss = sum(losses[-10:]) / min(10, len(losses))
        progress = (i / MAX_STEPS) * 100
        print(f"[{datetime.now().strftime('%H:%M:%S')}] خطوة {step} | loss: {loss.item():.4f} | avg: {avg_loss:.4f} | {progress:.1f}%")
    
    if i % SAVE_EVERY == 0 and i > 0:
        save_checkpoint(model, optimizer, step, loss.item())

print("=" * 60)
print(f"✅ انتهى التدريب عند الخطوة {step}")

# ====================================================
# حفظ النتائج النهائية
# ====================================================
def save_checkpoint(model, optimizer, step, loss):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'step': step,
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, 'latest.pt')
    upload_file(
        path_or_fileobj='latest.pt',
        path_in_repo='checkpoints/latest.pt',
        repo_id=HF_REPO,
        repo_type='model',
        token=HF_TOKEN
    )
    print(f"💾 تم حفظ checkpoint عند الخطوة {step}")

if losses:
    final_loss = losses[-1]
    save_checkpoint(model, optimizer, step, final_loss)
    
    state['current_step'] = step
    state['current_loss'] = final_loss
    state['best_loss'] = min(state['best_loss'], final_loss)
    state['updated_at'] = datetime.now().isoformat()
    state['history'].append({
        'timestamp': datetime.now().isoformat(),
        'final_step': step,
        'final_loss': final_loss,
        'steps_done': len(losses)
    })
    
    if len(state['history']) > 50:
        state['history'] = state['history'][-50:]
    
    with open('state.json', 'w') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    
    upload_file(
        path_or_fileobj='state.json',
        path_in_repo='state.json',
        repo_id=HF_REPO,
        repo_type='model',
        token=HF_TOKEN
    )
    
    print(f"📊 تم تحديث state.json")
    print(f"📈 الخطوة النهائية: {step}")
    print(f"📉 الخسارة النهائية: {final_loss:.4f}")
    print(f"🏆 أفضل خسارة: {state['best_loss']:.4f}")
else:
    print("⚠️ لم يتم التدريب")

print("=" * 60)
print("✅ انتهى! اغلق المتصفح بأمان.")
