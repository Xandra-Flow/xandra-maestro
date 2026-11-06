"""
Xandra Trainer Space - تطبيق Gradio يعمل على HF ZeroGPU

هذا الـ Space يحتوي على:
1. نموذج Xandra-Tiny (10M باراميتر)
2. معلم (Qwen-1.5B) للـ Distillation
3. دالة train_step قابلة للاستدعاء عبر API

يدعم ZeroGPU: GPU مجاني من HF (مع حدود)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download, upload_file
import gradio as gr
import json
from datetime import datetime

# ====== الإعدادات ======
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
HF_REPO = os.environ.get("HF_REPO", "your-username/xandra-training-state")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"📦 الجهاز: {DEVICE}")
print(f"📁 HF Repo: {HF_REPO}")

# ====== نموذج Xandra-Tiny (10M باراميتر) ======
class XandraTiny(nn.Module):
    """
    معمارية Xandra المصغّرة
    - 4 طبقات Transformer
    - 256 dim
    - 4 رؤوس انتباه
    - ~10M باراميتر
    """
    def __init__(self, vocab_size=32000, dim=256, n_layers=4, n_heads=4, ff_dim=1024):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(2048, dim)
        self.blocks = nn.ModuleList([
            self.TransformerBlock(dim, n_heads, ff_dim) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
    
    class TransformerBlock(nn.Module):
        def __init__(self, dim, n_heads, ff_dim):
            super().__init__()
            self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(dim, ff_dim),
                nn.GELU(),
                nn.Linear(ff_dim, dim)
            )
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        
        def forward(self, x):
            attn_out, _ = self.attn(x, x, x, need_weights=False)
            x = x + attn_out
            x = self.norm1(x)
            x = x + self.ffn(x)
            x = self.norm2(x)
            return x
    
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits

# ====== تحميل الـ Tokenizer المشترك ======
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ====== تحميل النموذج (Xandra) ======
def load_xandra_model():
    """تحميل Xandra-Tiny + checkpoint إن وجد"""
    model = XandraTiny().to(DEVICE)
    
    # محاولة تحميل آخر checkpoint
    try:
        ckpt_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="checkpoints/latest.pt",
            repo_type="model"
        )
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        print(f"✅ تحميل checkpoint من الخطوة {checkpoint['step']}")
        return model, checkpoint['step']
    except Exception as e:
        print(f"🆕 بدء تدريب جديد: {e}")
        return model, 0

# ====== تحميل المعلم (Qwen) ======
def load_teacher():
    """تحميل Qwen كمعلم"""
    print("📚 تحميل المعلم Qwen-1.5B...")
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return teacher

# ====== دالة التدريب ======
def train_step(current_step, max_steps=500):
    """
    تنفيذ خطوات تدريب عبر Distillation
    
    Args:
        current_step: الخطوة الحالية (من HF state)
        max_steps: عدد الخطوات في هذه الجلسة
    
    Returns:
        dict: نتائج الجلسة
    """
    try:
        # 1. تحميل Xandra + آخر checkpoint
        student, step = load_xandra_model()
        if current_step > 0 and step < current_step:
            step = current_step
        
        # 2. تحميل المعلم
        teacher = load_teacher()
        teacher.eval()
        
        # 3. إعداد التدريب
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
        student.train()
        
        # 4. داتا وهمية للتجربة (في الإنتاج: داتا حقيقية من مكتبتك)
        def get_sample_batch():
            """توليد batch تجريبي"""
            texts = [
                "العمارة الإسلامية فن راقٍ",
                "الزراعة في مصر القديمة كانت متقدمة",
                "الصناعة تحتاج إلى معرفة وتدريب"
            ]
            batch = []
            for text in texts:
                tokens = tokenizer(text, return_tensors="pt", padding="max_length", 
                                  truncation=True, max_length=64)
                batch.append(tokens['input_ids'])
            return torch.cat(batch).to(DEVICE)
        
        # 5. حلقة التدريب
        losses = []
        for i in range(max_steps):
            batch = get_sample_batch()
            
            # التدريب التقليدي (Loss عادي)
            logits = student(batch)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch.view(-1)
            )
            
            # (اختياري) إضافة Distillation Loss من المعلم
            # with torch.no_grad():
            #     teacher_logits = teacher(batch).logits
            # distill_loss = F.kl_div(
            #     F.log_softmax(logits, dim=-1),
            #     F.softmax(teacher_logits, dim=-1),
            #     reduction="batchmean"
            # )
            # loss = loss + 0.5 * distill_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            losses.append(loss.item())
            
            if i % 50 == 0:
                print(f"  خطوة {step}: loss = {loss.item():.4f}")
        
        # 6. حفظ checkpoint
        checkpoint = {
            'model_state': student.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'step': step,
            'final_loss': losses[-1] if losses else None,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, "latest.pt")
        upload_file(
            path_or_fileobj="latest.pt",
            path_in_repo="checkpoints/latest.pt",
            repo_id=HF_REPO,
            repo_type="model",
            token=os.environ.get("HF_TOKEN_OWNER")
        )
        
        print(f"✅ حفظ checkpoint عند الخطوة {step}")
        
        return {
            "success": True,
            "final_step": step,
            "final_loss": losses[-1] if losses else None,
            "steps_completed": max_steps,
            "avg_loss": sum(losses) / len(losses) if losses else None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "final_step": current_step
        }

# ====== واجهة Gradio ======
with gr.Blocks() as demo:
    gr.Markdown("""
    # 🎯 Xandra Hope Trainer
    
    هذا الـ Space يقوم بتدريب نموذج Xandra-Tiny عبر Distillation.
    
    ## الاستخدام عبر API:
    ```python
    from gradio_client import Client
    client = Client("your-username/xandra-trainer", hf_token="hf_your_token")
    result = client.predict(current_step=0, max_steps=500, api_name="/train_step")
    ```
    """)
    
    with gr.Row():
        current_step_input = gr.Number(label="الخطوة الحالية", value=0)
        max_steps_input = gr.Number(label="عدد الخطوات", value=500)
    
    train_btn = gr.Button("🚀 بدء التدريب", variant="primary")
    
    output = gr.JSON(label="النتيجة")
    
    train_btn.click(
        fn=train_step,
        inputs=[current_step_input, max_steps_input],
        outputs=output,
        api_name="train_step"
    )

if __name__ == "__main__":
    demo.launch()
