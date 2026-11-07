"""
Xandra Worker Registrar
========================
يقرأ بيانات الصديق من GitHub Issue ويضيفه لـ workers_config.yaml
"""
import os
import re
import yaml
import requests
from datetime import datetime

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
ISSUE_BODY = os.environ.get("ISSUE_BODY", "")
ISSUE_TITLE = os.environ.get("ISSUE_TITLE", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
OWNER_CHAT_ID = os.environ.get("OWNER_CHAT_ID")

REPO = os.environ.get("GITHUB_REPOSITORY", "Xandra-Flow/xandra-maestro")
CONFIG_PATH = "maestro/workers_config.yaml"


def parse_issue_body(body):
    """استخراج البيانات من نص الـ Issue"""
    data = {}
    patterns = {
        "worker_id": r"\*\*Worker ID:\*\*\s*`([^`]+)`",
        "hf_username": r"\*\*HF Username:\*\*\s*`([^`]+)`",
        "hf_token": r"\*\*HF Token:\*\*\s*`([^`]+)`",
        "timestamp": r"\*\*Timestamp:\*\*\s*`([^`]+)`",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, body)
        if match:
            data[key] = match.group(1).strip()
    return data


def update_workers_config(worker_data):
    """قراءة workers_config.yaml وإضافة Worker جديد"""
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    
    # 1. قراءة الملف الحالي من GitHub API
    url = f"https://api.github.com/repos/{REPO}/contents/{CONFIG_PATH}"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"فشل قراءة workers_config.yaml: {response.text}")
    
    file_data = response.json()
    sha = file_data["sha"]
    
    # فك الترميز (base64 -> utf-8)
    import base64
    content = base64.b64decode(file_data["content"]).decode("utf-8")
    config = yaml.safe_load(content)
    
    # 2. التحقق من عدم التكرار
    for w in config.get("workers", []):
        if w.get("hf_username") == worker_data["hf_username"]:
            print(f"⚠️ الصديق {worker_data['hf_username']} مسجل بالفعل!")
            return False
    
    # 3. إضافة الـ Worker الجديد
    new_worker = {
        "id": worker_data["worker_id"],
        "service": "kaggle",
        "enabled": True,
        "hf_username": worker_data["hf_username"],
        "hf_token": worker_data["hf_token"],
        "added_at": datetime.now().isoformat(),
        "total_steps_trained": 0,
        "total_sessions": 0,
        "status": "active"
    }
    config["workers"].append(new_worker)
    
    # 4. رفع الملف المُحدّث
    new_content = yaml.dump(config, default_flow_style=False, allow_unicode=True)
    new_content_encoded = base64.b64encode(new_content.encode("utf-8")).decode("utf-8")
    
    update_data = {
        "message": f"Add worker: {worker_data['worker_id']} ({worker_data['hf_username']})",
        "content": new_content_encoded,
        "sha": sha
    }
    
    update_response = requests.put(url, headers=headers, json=update_data)
    
    if update_response.status_code not in [200, 201]:
        raise Exception(f"فشل رفع الملف: {update_response.text}")
    
    return True


def send_telegram_notification(worker_data):
    """إرسال إشعار للمالك"""
    if not TELEGRAM_BOT_TOKEN or not OWNER_CHAT_ID:
        return
    
    message = (
        "🎉 *صديق جديد انضم للنظام!*\n\n"
        f"🆔 المعرّف: `{worker_data['worker_id']}`\n"
        f"👤 حساب HF: `{worker_data['hf_username']}`\n"
        f"📅 الوقت: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n\n"
        "✅ تمت إضافته لـ workers_config.yaml تلقائياً."
    )
    
    payload = {
        "chat_id": OWNER_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json=payload
    )


def main():
    print(f"📥 معالجة Issue: {ISSUE_TITLE}")
    print(f"📄 محتوى Issue:\n{ISSUE_BODY}\n")
    
    worker_data = parse_issue_body(ISSUE_BODY)
    
    if not worker_data.get("worker_id") or not worker_data.get("hf_username"):
        print("❌ لم يتم العثور على بيانات صالحة في الـ Issue")
        exit(1)
    
    print(f"✅ بيانات مستخرجة:")
    for k, v in worker_data.items():
        if k == "hf_token":
            print(f"  {k}: {v[:10]}...{v[-4:]}")
        else:
            print(f"  {k}: {v}")
    
    try:
        added = update_workers_config(worker_data)
        if added:
            print("\n✅ تمت إضافة الصديق بنجاح!")
            send_telegram_notification(worker_data)
        else:
            print("\n⚠️ الصديق مسجل مسبقاً")
    except Exception as e:
        print(f"\n❌ فشل الإضافة: {e}")
        exit(1)


if __name__ == "__main__":
    main()
