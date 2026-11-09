"""
╔═══════════════════════════════════════════════════════════════╗
║   Xandra Hope - Maestro (Smart Load Balancer + Calculated Quota)║
║   Copyright (c) 2026 Ahmed Sayed Kamel Mahmoud &               ║
║                  Александра Евгеньевна Острикова              ║
║   All Rights Reserved. License: XHNSL v1.0                   ║
╚═══════════════════════════════════════════════════════════════╝
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

HF_REPO = os.environ.get("HF_REPO")
HF_TOKEN = os.environ.get("HF_TOKEN_OWNER")

GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_PASS")
EMAIL_TO = os.environ.get("EMAIL_TO", GMAIL_USER)

KAGGLE_ACCOUNTS = []
for i in range(1, 7):
    username = os.environ.get(f"KAGGLE_USERNAME_{i}" if i > 1 else "KAGGLE_USERNAME")
    key = os.environ.get(f"KAGGLE_KEY_{i}" if i > 1 else "KAGGLE_KEY")
    if username and key:
        KAGGLE_ACCOUNTS.append({"username": username, "key": key, "index": i, "status": "active"})

WEEKLY_GPU_LIMIT_PER_ACCOUNT = 30
LOCK_TIMEOUT_HOURS = 8
MIN_HOURS_BETWEEN_SESSIONS = 2

def send_email(subject, body):
    if not GMAIL_USER or not GMAIL_PASS: return
    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = EMAIL_TO
        msg['Subject'] = f"🎵 Xandra Maestro: {subject}"
        msg.attach(MIMEText(body.replace('\n', '<br>'), 'html', 'utf-8'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, EMAIL_TO, msg.as_string())
        server.quit()
    except: pass

def get_state():
    try:
        path = hf_hub_download(repo_id=HF_REPO, filename="state.json", repo_type="model", token=HF_TOKEN)
        with open(path) as f: return json.load(f)
    except:
        return {
            "current_step": 0, "target_steps": 500000, "current_loss": None,
            "history": [], "best_loss": float('inf'), "updated_at": None,
            "gpu_hours_used": 0, "training_lock": {"is_locked": False},
            "current_account_index": 0, "account_usage": {}, "disabled_accounts": []
        }

def update_state(state):
    state["updated_at"] = datetime.now().isoformat()
    with open("state.json", "w") as f: json.dump(state, f, indent=2, ensure_ascii=False)
    upload_file(path_or_fileobj="state.json", path_in_repo="state.json", repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)

def check_lock(state):
    lock = state.get("training_lock", {})
    if not lock.get("is_locked"): return False, None
    expires_at = lock.get("expires_at")
    if expires_at:
        expires_time = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        if datetime.now(expires_time.tzinfo) > expires_time: return False, "منتهي الصلاحية"
    return True, lock.get("locked_by")

def acquire_lock(state, account_name):
    state["training_lock"] = {
        "is_locked": True,
        "locked_by": f"maestro-{account_name}",
        "locked_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(hours=LOCK_TIMEOUT_HOURS)).isoformat()
    }
    return state

def sync_account_quotas(state):
    """حساب أرقام الاستهلاك الفعلية لكل حساب من سجل الجلسات (history)"""
    print("🔄 حساب أرقام الاستهلاك الفعلية من السجل...")
    account_usage = state.get("account_usage", {})
    
    now = datetime.now()
    week_start = now - timedelta(days=now.weekday())
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # تهيئة الحسابات الحالية في الـ Secrets
    for acc in KAGGLE_ACCOUNTS:
        if acc["username"] not in account_usage:
            account_usage[acc["username"]] = {"used": 0, "remaining": 30}
    
    # حساب الساعات الأسبوعية لكل حساب من الـ history
    weekly_hours = {acc["username"]: 0 for acc in KAGGLE_ACCOUNTS}
    
    for session in state.get("history", []):
        worker_id = session.get("worker_id", "")
        gpu_hours = session.get("gpu_hours", 0)
        timestamp_str = session.get("timestamp", "")
        
        account_name = None
        if worker_id.startswith("kaggle-"):
            account_name = worker_id[7:]
        elif worker_id.startswith("maestro-"):
            account_name = worker_id[8:]
            
        if account_name and account_name in weekly_hours:
            try:
                session_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if session_time > week_start:
                    weekly_hours[account_name] += gpu_hours
            except: pass
            
    for username, hours_used in weekly_hours.items():
        remaining = max(0, 30 - hours_used)
        account_usage[username] = {
            "used": round(hours_used, 2),
            "remaining": round(remaining, 2)
        }
        print(f"  ✅ {username}: مستخدم {round(hours_used, 2)}h | متبقي {round(remaining, 2)}h")
        
    state["account_usage"] = account_usage
    return state

def select_available_account(state):
    if not KAGGLE_ACCOUNTS: return None, 0, []
    
    disabled_accounts = state.get("disabled_accounts", [])
    active_accounts = [acc for acc in KAGGLE_ACCOUNTS if acc["username"] not in disabled_accounts]
    
    if not active_accounts: return None, 0, []
    
    account_stats = []
    real_usage = state.get("account_usage", {})
    
    for acc in active_accounts:
        username = acc["username"]
        remaining = real_usage.get(username, {}).get("remaining", 30)
        
        account_stats.append({
            "username": username,
            "remaining": remaining,
            "weekly_hours": real_usage.get(username, {}).get("used", 0)
        })
    
    # ترتيب الحسابات بناءً على أعلى ساعات متبقية
    account_stats.sort(key=lambda x: x["remaining"], reverse=True)
    
    if account_stats and account_stats[0]["remaining"] >= 2:
        best_account = account_stats[0]
        for i, acc in enumerate(KAGGLE_ACCOUNTS):
            if acc["username"] == best_account["username"]:
                return KAGGLE_ACCOUNTS[i], best_account["remaining"], account_stats
    
    return None, 0, account_stats

def check_last_session(state):
    history = state.get("history", [])
    if not history: return True
    last = history[-1]
    try:
        last_time = datetime.fromisoformat(last["timestamp"].replace("Z", "+00:00"))
        hours_since = (datetime.now(last_time.tzinfo) - last_time).total_seconds() / 3600
        return hours_since >= MIN_HOURS_BETWEEN_SESSIONS
    except: return True

def launch_kaggle_notebook(account):
    username = account["username"]
    key = account["key"]
    try:
        os.makedirs("/root/.kaggle", exist_ok=True)
        with open("/root/.kaggle/kaggle.json", "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod("/root/.kaggle/kaggle.json", 0o600)
        subprocess.run(["pip", "install", "-q", "kaggle"], check=True)
        
        # حقن اسم المستخدم داخل الكود قبل الرفع
        with open("xandra_worker.py", "r") as f:
            worker_code = f.read()
        worker_code = worker_code.replace('__KAGGLE_USERNAME__', username)
        with open("xandra_worker.py", "w") as f:
            f.write(worker_code)
        
        result = subprocess.run(["kaggle", "kernels", "status", f"{username}/xandra_worker"], capture_output=True, text=True)
        status_text = result.stdout.lower()
        if "complete" in status_text or "error" in status_text or "canceled" in status_text or "no kernels found" in status_text:
            metadata = {
                "id": f"{username}/xandra_worker", "title": "Xandra Worker",
                "code_file": "xandra_worker.py", "language": "python",
                "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_internet": True
            }
            with open("kernel-metadata.json", "w") as f: json.dump(metadata, f)
            result = subprocess.run(["kaggle", "kernels", "push", "-p", "."], capture_output=True, text=True)
            if result.returncode == 0: return True
        return True
    except: return False

def main():
    state = get_state()
    
    # 1. تحديث أرقام الاستهلاك الحقيقية بحسابها من الـ history
    state = sync_account_quotas(state)
    
    # تقرير حالة الحسابات
    result = select_available_account(state)
    if result and result[2]:
        print("📊 تقرير الحسابات:")
        for acc_stat in result[2]:
            print(f"  - {acc_stat['username']}: {acc_stat['remaining']:.1f}h متبقية")
    
    # حفظ الحالة بعد تحديث الأرقام
    update_state(state)
    
    is_locked, locked_by = check_lock(state)
    if is_locked:
        print(f"⏳ النظام مشغول بواسطة: {locked_by}")
        return
    
    account, remaining_hours, _ = result
    if not account:
        print("⚠️ جميع الحسابات مشغولة أو استُنفدت")
        if not state.get("all_accounts_exhausted_alerted"):
            send_email("⚠️ جميع حسابات كاجل استُنفدت", "تم استنفاد جميع حصص GPU الأسبوعية.")
            state["all_accounts_exhausted_alerted"] = True
            update_state(state)
        return
    
    if not check_last_session(state):
        print(f"⏳ لم تمر {MIN_HOURS_BETWEEN_SESSIONS} ساعات بعد على آخر جلسة")
        return
    
    print(f"🚀 إطلاق جلسة تدريب جديدة على {account['username']} (متبقي {remaining_hours:.1f}h)")
    state = acquire_lock(state, account["username"])
    state["current_account_index"] = account["index"]
    state["all_accounts_exhausted_alerted"] = False
    update_state(state)
    
    success = launch_kaggle_notebook(account)
    if not success:
        send_email(f"🚨 فشل إطلاق جلسة Kaggle ({account['username']})", "حدث خطأ.")
        state["training_lock"] = {"is_locked": False, "locked_by": None, "locked_at": None, "expires_at": None}
        update_state(state)

if __name__ == "__main__":
    main()
