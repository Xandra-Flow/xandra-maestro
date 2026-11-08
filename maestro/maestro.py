"""
╔═══════════════════════════════════════════════════════════════╗
║   Xandra Hope - Maestro (Multi-Account + Gmail + Smart Lock)  ║
║   Copyright (c) 2026 Ahmed Sayed Kamel Mahmoud &              ║
║                  Александра Евгеньевна Острикова              ║
║   All Rights Reserved. License: XHNSL v1.0                    ║
╚═══════════════════════════════════════════════════════════════╝

المايسترو الجديد مع Multi-Account Rotation:
1. يقرأ state من HF
2. يفحص القفل الذكي
3. يدور بين حسابات كاجل المتعددة (Rotation)
4. يطلق Notebook على الحساب المتاح
5. يرسل إيميلات عبر Gmail للتنبيهات المهمة
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

# إعدادات Gmail
GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_PASS")
EMAIL_TO = os.environ.get("EMAIL_TO", GMAIL_USER)

# قائمة حسابات كاجل (حتى 6 حسابات)
KAGGLE_ACCOUNTS = []
for i in range(1, 7):
    username = os.environ.get(f"KAGGLE_USERNAME_{i}" if i > 1 else "KAGGLE_USERNAME")
    key = os.environ.get(f"KAGGLE_KEY_{i}" if i > 1 else "KAGGLE_KEY")
    if username and key:
        KAGGLE_ACCOUNTS.append({"username": username, "key": key, "index": i})

WEEKLY_GPU_LIMIT_PER_ACCOUNT = 30
LOCK_TIMEOUT_HOURS = 8
MIN_HOURS_BETWEEN_SESSIONS = 2  # ساعتان بين الجلسات (للاستفادة من الحسابات المتعددة)

# ====== الإيميل عبر Gmail ======
def send_email(subject, body):
    if not GMAIL_USER or not GMAIL_PASS:
        print(f"📧 (بدون إيميل) {subject}: {body}")
        return
    
    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = EMAIL_TO
        msg['Subject'] = f"🎵 Xandra Maestro: {subject}"
        
        html_body = body.replace('\n', '<br>')
        msg.attach(MIMEText(html_body, 'html', 'utf-8'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
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
            "gpu_hours_used": 0, "training_lock": {"is_locked": False},
            "current_account_index": 0
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
def acquire_lock(state, account_name):
    state["training_lock"] = {
        "is_locked": True,
        "locked_by": f"maestro-{account_name}",
        "locked_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(hours=LOCK_TIMEOUT_HOURS)).isoformat()
    }
    return state

# ====== اختيار حساب كاجل متاح (Rotation) ======
def select_available_account(state):
    if not KAGGLE_ACCOUNTS:
        print("❌ لا توجد حسابات كاجل مُعدة!")
        return None, 0
    
    current_index = state.get("current_account_index", 0)
    
    for i in range(len(KAGGLE_ACCOUNTS)):
        idx = (current_index + i) % len(KAGGLE_ACCOUNTS)
        account = KAGGLE_ACCOUNTS[idx]
        username = account["username"]
        
        # حساب ساعات GPU المستخدمة لهذا الحساب هذا الأسبوع
        now = datetime.now()
        week_start = now - timedelta(days=now.weekday())
        
        weekly_hours = 0
        for session in state.get("history", []):
            if session.get("worker_id", "").endswith(username):
                try:
                    session_time = datetime.fromisoformat(session["timestamp"].replace("Z", "+00:00"))
                    if session_time > week_start:
                        weekly_hours += session.get("gpu_hours", 0)
                except:
                    pass
        
        remaining = WEEKLY_GPU_LIMIT_PER_ACCOUNT - weekly_hours
        
        if remaining >= 2:
            print(f"✅ حساب متاح: {username} (متبقي {remaining:.1f} ساعة)")
            return account, remaining
    
    print("❌ جميع الحسابات استُنفدت حصصها الأسبوعية!")
    return None, 0

# ====== فحص آخر جلسة ======
def check_last_session(state):
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
def launch_kaggle_notebook(account):
    username = account["username"]
    key = account["key"]
    
    try:
        os.makedirs("/root/.kaggle", exist_ok=True)
        with open("/root/.kaggle/kaggle.json", "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod("/root/.kaggle/kaggle.json", 0o600)
        
        subprocess.run(["pip", "install", "-q", "kaggle"], check=True)
        
        # فحص حالة Notebook الحالي
        result = subprocess.run([
            "kaggle", "kernels", "status", f"{username}/xandra_worker"
        ], capture_output=True, text=True)
        
        print(f"📊 حالة Kaggle ({username}): {result.stdout}")
        
        status_text = result.stdout.lower()
        if "complete" in status_text or "error" in status_text or "canceled" in status_text or "no kernels found" in status_text:
            print(f"🚀 إطلاق Notebook جديد على {username}...")
            
            metadata = {
                "id": f"{username}/xandra_worker",
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
            
            result = subprocess.run([
                "kaggle", "kernels", "push", "-p", "."
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ تم إطلاق Notebook على {username}")
                return True
            else:
                print(f"❌ فشل الإطلاق: {result.stderr}")
                return False
        else:
            print(f"⚠️ Notebook يعمل بالفعل على {username}")
            return True
        
    except Exception as e:
        print(f"❌ خطأ في Kaggle API: {e}")
        return False

# ====== فحص المراحل المهمة ======
def check_milestones(state):
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
    print(f"🎵 Xandra Maestro (Multi-Account) - {datetime.now()}")
    print(f"👥 حسابات كاجل المتاحة: {len(KAGGLE_ACCOUNTS)}")
    
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
    
    # 4. اختيار حساب متاح (Rotation)
    result = select_available_account(state)
    if not result or result[0] is None:
        print("⚠️ جميع الحسابات مشغولة أو استُنفدت")
        if not state.get("all_accounts_exhausted_alerted"):
            send_email(
                "⚠️ جميع حسابات كاجل استُنفدت",
                f"تم استنفاد جميع حصص GPU الأسبوعية.\nعدد الحسابات: {len(KAGGLE_ACCOUNTS)}\nستتجدد الحصص الأسبوع القادم."
            )
            state["all_accounts_exhausted_alerted"] = True
            update_state(state)
        return
    
    account, remaining_hours = result
    
    # 5. فحص آخر جلسة
    if not check_last_session(state):
        print(f"⏳ لم تمر {MIN_HOURS_BETWEEN_SESSIONS} ساعات بعد على آخر جلسة")
        return
    
    # 6. إطلاق Notebook على الحساب المحدد
    print(f"🚀 إطلاق جلسة تدريب جديدة على {account['username']}...")
    state = acquire_lock(state, account["username"])
    state["current_account_index"] = account["index"]
    state["all_accounts_exhausted_alerted"] = False
    update_state(state)
    
    success = launch_kaggle_notebook(account)
    
    if success:
        print(f"✅ تم إطلاق الجلسة بنجاح على {account['username']}")
    else:
        print(f"❌ فشل الإطلاق على {account['username']}")
        send_email(
            f"🚨 فشل إطلاق جلسة Kaggle ({account['username']})",
            f"حدث خطأ عند محاولة إطلاق Notebook على حساب {account['username']}.\nسيحاول المايسترو حساباً آخر في الدورة القادمة."
        )
        state["training_lock"] = {
            "is_locked": False, "locked_by": None,
            "locked_at": None, "expires_at": None
        }
        update_state(state)

if __name__ == "__main__":
    main()
