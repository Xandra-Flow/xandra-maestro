"""
╔═══════════════════════════════════════════════════════════════╗
║   Xandra Hope - Maestro (Kaggle API + Gmail + Smart Lock)     ║
║   Copyright (c) 2026 Ahmed Sayed Kamel Mahmoud &               ║
║                  Александра Евгеньевна Острикова              ║
║   All Rights Reserved. License: XHNSL v1.0                   ║
╚═══════════════════════════════════════════════════════════════╝

المايسترو الجديد:
1. يقرأ state من HF
2. يفحص القفل الذكي
3. يحسب ساعات GPU المتبقية
4. يطلق Kaggle API تلقائياً (kernels_push)
5. يرسل إيميلات عبر Gmail للتنبيهات المهمة فقط
"""

import os
import json
import smtplib
import requests
import subprocess
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download, upload_file

# ====== الإعدادات ======
HF_REPO = os.environ.get("HF_REPO")
HF_TOKEN = os.environ.get("HF_TOKEN_OWNER")
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY")

# إعدادات Gmail
GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_PASS")
EMAIL_TO = os.environ.get("EMAIL_TO", GMAIL_USER)

WEEKLY_GPU_LIMIT_HOURS = 30  # كاجل تعطي 30 ساعة أسبوعياً
LOCK_TIMEOUT_HOURS = 8
MIN_HOURS_BETWEEN_SESSIONS = 6  # ننتظر 6 ساعات بين الجلسات

# ====== الإيميل عبر Gmail ======
def send_email(subject, body):
    """إرسال إيميل عبر Gmail SMTP"""
    if not GMAIL_USER or not GMAIL_PASS:
        print(f"📧 (بدون إيميل) {subject}: {body}")
        return
    
    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = EMAIL_TO
        msg['Subject'] = f"🎵 Xandra Maestro: {subject}"
        
        # تحويل النص العادي إلى HTML بسيط
        html_body = body.replace('\n', '<br>')
        msg.attach(MIMEText(html_body, 'html', 'utf-8'))
        
        # الاتصال بـ Gmail
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
        
        # إرسال
        server.sendmail(GMAIL_USER, EMAIL_TO, msg.as_string())
        server.quit()
        
        print(f"📧 تم إرسال إيميل: {subject}")
    except Exception as e:
        print(f"❌ خطأ في الإيميل: {e}")

# ====== قراءة state من HF ======
def get_state():
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

def update_state(state):
    state["updated_at"] = datetime.now().isoformat()
    with open("state.json", "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    upload_file(
        path_or_fileobj="state.json",
        path_in_repo="state.json",
        repo_id=HF_REPO, repo_type="model", token=HF_TOKEN
    )

# ====== فحص القفل ======
def check_lock(state):
    """فحص هل النظام مشغول"""
    lock = state.get("training_lock", {})
    if not lock.get("is_locked"):
        return False, None
    
    expires_at = lock.get("expires_at")
    if expires_at:
        expires_time = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        if datetime.now(expires_time.tzinfo) > expires_time:
            return False, "منتهي الصلاحية"
    
    return True, lock.get("locked_by")

# ====== الحصول على القفل ======
def acquire_lock(state):
    state["training_lock"] = {
        "is_locked": True,
        "locked_by": "maestro-kaggle",
        "locked_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(hours=LOCK_TIMEOUT_HOURS)).isoformat()
    }
    return state

# ====== فحص ساعات GPU ======
def check_gpu_hours(state):
    """فحص هل لدينا ساعات GPU متبقية هذا الأسبوع"""
    now = datetime.now()
    week_start = now - timedelta(days=now.weekday())
    
    weekly_hours = 0
    for session in state.get("history", []):
        try:
            session_time = datetime.fromisoformat(session["timestamp"].replace("Z", "+00:00"))
            if session_time > week_start:
                weekly_hours += session.get("gpu_hours", 0)
        except:
            pass
    
    remaining = WEEKLY_GPU_LIMIT_HOURS - weekly_hours
    return remaining, weekly_hours

# ====== فحص آخر جلسة ======
def check_last_session(state):
    """فحص متى كانت آخر جلسة تدريب"""
    history = state.get("history", [])
    if not history:
        return True
    
    last = history[-1]
    try:
        last_time = datetime.fromisoformat(last["timestamp"].replace("Z", "+00:00"))
        hours_since = (datetime.now(last_time.tzinfo) - last_time).total_seconds() / 3600
        return hours_since >= MIN_HOURS_BETWEEN_SESSIONS
    except:
        return True

# ====== إطلاق Kaggle Notebook ======
def launch_kaggle_notebook():
    """إطلاق Notebook على Kaggle عبر API"""
    try:
        # إعداد kaggle credentials
        os.makedirs("/root/.kaggle", exist_ok=True)
        with open("/root/.kaggle/kaggle.json", "w") as f:
            json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, f)
        os.chmod("/root/.kaggle/kaggle.json", 0o600)
        
        # تثبيت kaggle CLI
        subprocess.run(["pip", "install", "-q", "kaggle"], check=True)
        
        # فحص حالة Notebook الحالي
        result = subprocess.run([
            "kaggle", "kernels", "status", f"{KAGGLE_USERNAME}/xandra_worker"
        ], capture_output=True, text=True)
        
        print(f"📊 حالة Kaggle: {result.stdout}")
        
        # لو Notebook متوقف أو لم يُشغل، نطلقه
        status_text = result.stdout.lower()
        if "complete" in status_text or "error" in status_text or "canceled" in status_text or "no kernels found" in status_text:
            print("🚀 إطلاق Notebook جديد...")
            
            # إنشاء metadata للـ Notebook
            metadata = {
                "id": f"{KAGGLE_USERNAME}/xandra_worker",
                "title": "Xandra Worker",
                "code_file": "xandra_worker.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True
            }
            
            with open("kernel-metadata.json", "w") as f:
                json.dump(metadata, f)
            
            # نسخ كود Worker (لو موجود في نفس المجلد)
            # لو ليس موجوداً، نحتاج رفعه كملف منفصل
            
            result = subprocess.run([
                "kaggle", "kernels", "push", "-p", "."
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ تم إطلاق Notebook على Kaggle")
                return True
            else:
                print(f"❌ فشل الإطلاق: {result.stderr}")
                return False
        else:
            print("⚠️ Notebook يعمل بالفعل")
            return True
        
    except Exception as e:
        print(f"❌ خطأ في Kaggle API: {e}")
        return False

# ====== فحص المراحل المهمة ======
def check_milestones(state):
    """فحص هل وصلنا لمرحلة مهمة"""
    step = state.get("current_step", 0)
    milestones = [1000, 5000, 10000, 50000, 100000, 250000, 500000]
    
    for m in milestones:
        if step == m:
            send_email(
                f"🎉 مرحلة مهمة: {m:,} خطوة!",
                f"وصلنا للخطوة {m:,}\nالخسارة الحالية: {state.get('current_loss', 'N/A')}\n\nشاهد التفاصيل: https://xandra-api.smart-brute-force.workers.dev/stats"
            )

# ====== فحص النضج ======
def check_convergence(state):
    """فحص هل Loss توقف عن التحسن"""
    history = state.get("history", [])
    if len(history) < 5:
        return
    
    recent = history[-5:]
    losses = [h.get("final_loss", 0) for h in recent if h.get("final_loss")]
    
    if len(losses) >= 5:
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        if abs(improvement) < 1:
            send_email(
                "⚠️ النموذج قد يكون نضج",
                f"الخسارة لم تتحسن خلال آخر 5 جلسات\n"
                f"متوسط الخسارة: {sum(losses)/len(losses):.4f}\n\n"
                f"قد تحتاج لمراجعة النموذج يدوياً."
            )

# ====== الحلقة الرئيسية ======
def main():
    print(f"🎵 Xandra Maestro - {datetime.now()}")
    
    # 1. قراءة state
    state = get_state()
    print(f"📊 الخطوة الحالية: {state.get('current_step', 0)}")
    
    # 2. فحص المراحل
    check_milestones(state)
    check_convergence(state)
    
    # 3. فحص القفل
    is_locked, locked_by = check_lock(state)
    if is_locked:
        print(f"⏳ النظام مشغول بواسطة: {locked_by}")
        return
    
    # 4. فحص ساعات GPU
    remaining_hours, used_hours = check_gpu_hours(state)
    print(f"⏱️ ساعات GPU: {used_hours:.1f} مستخدمة / {remaining_hours:.1f} متبقية")
    
    if remaining_hours < 2:
        print("⚠️ لا توجد ساعات GPU كافية هذا الأسبوع")
        if remaining_hours > 0 and not state.get("low_gpu_alerted"):
            send_email(
                "⚠️ ساعات GPU شبه منتهية",
                f"تبقى {remaining_hours:.1f} ساعة فقط هذا الأسبوع على Kaggle."
            )
            state["low_gpu_alerted"] = True
            update_state(state)
        return
    
    # 5. فحص آخر جلسة
    if not check_last_session(state):
        print("⏳ لم تمر 6 ساعات بعد على آخر جلسة")
        return
    
    # 6. إطلاق Notebook على Kaggle
    print("🚀 إطلاق جلسة تدريب جديدة...")
    state = acquire_lock(state)
    update_state(state)
    
    success = launch_kaggle_notebook()
    
    if success:
        print("✅ تم إطلاق الجلسة بنجاح")
        state["low_gpu_alerted"] = False
    else:
        print("❌ فشل الإطلاق")
        send_email(
            "🚨 فشل إطلاق جلسة Kaggle",
            "حدث خطأ عند محاولة إطلاق Notebook على Kaggle.\n"
            "تحقق من الإعدادات في GitHub Actions."
        )
        # تحرير القفل
        state["training_lock"] = {
            "is_locked": False, "locked_by": None,
            "locked_at": None, "expires_at": None
        }
        update_state(state)

if __name__ == "__main__":
    main()
