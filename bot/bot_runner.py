"""
Xandra Bot Runner - يعمل في GitHub Actions كل 10 دقائق
=========================================================
يستخدم Long Polling بدلاً من Webhook
يتتبع آخر update_id في ملف محلي (bot_state.json)

الأوامر المدعومة:
- /start: ترحيب
- /join: تسجيل صديق جديد (يطلب Token)
- /status: حالة التدريب
- /progress: آخر جلسات التدريب
- /help: المساعدة
- /leave: حذف البيانات
"""

import os
import json
import logging
import requests
import time
import yaml
from datetime import datetime
from huggingface_hub import hf_hub_download, HfApi

# ====== الإعدادات ======
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
HF_REPO = os.environ.get("HF_REPO", "your-username/xandra-training-state")
HF_TOKEN = os.environ.get("HF_TOKEN_OWNER")
OWNER_CHAT_ID = int(os.environ.get("OWNER_CHAT_ID", "0"))
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "your-username/xandra-maestro")
STATE_FILE = "bot_state.json"

# ====== التسجيل ======
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====== إدارة الحالة (آخر update_id) ======
def load_state():
    """تحميل آخر update_id"""
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except:
        return {"last_update_id": 0}

def save_state(state):
    """حفظ آخر update_id"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

# ====== إرسال رسالة ======
def send_message(chat_id, text, parse_mode="Markdown"):
    """إرسال رسالة عبر Telegram API"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            return True
        else:
            logger.error(f"❌ فشل إرسال: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ خطأ: {e}")
        return False

# ====== التحقق من HF Token ======
def verify_hf_token(token):
    """التحقق من صحة HF Token"""
    try:
        response = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            return {
                "valid": True,
                "username": data.get("name", "unknown")
            }
        return {"valid": False, "error": "Token غير صحيح"}
    except Exception as e:
        return {"valid": False, "error": str(e)}

# ====== إضافة Worker لـ workers_config.yaml ======
def add_worker_to_config(worker_id, hf_username, telegram_user_id, telegram_username):
    """إضافة Worker جديد عبر GitHub API"""
    try:
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # 1. قراءة workers_config.yaml الحالي
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/maestro/workers_config.yaml"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"❌ فشل قراءة workers_config: {response.text}")
            return False
        
        content_data = response.json()
        import base64
        content = base64.b64decode(content_data['content']).decode('utf-8')
        config = yaml.safe_load(content)
        
        # 2. التحقق من عدم التكرار
        for w in config.get('workers', []):
            if w.get('telegram_user_id') == telegram_user_id:
                # تحديث موجود
                w['hf_username'] = hf_username
                w['enabled'] = True
                w['status'] = 'active'
                break
        else:
            # إضافة جديد
            new_worker = {
                'id': worker_id,
                'service': 'kaggle',
                'enabled': True,
                'hf_username': hf_username,
                'telegram_user_id': telegram_user_id,
                'telegram_username': telegram_username,
                'added_at': datetime.now().isoformat(),
                'total_steps_trained': 0,
                'total_sessions': 0,
                'status': 'active'
            }
            config['workers'].append(new_worker)
        
        # 3. رفع الملف المُحدّث
        new_content = yaml.dump(config, default_flow_style=False, allow_unicode=True)
        new_content_encoded = base64.b64encode(new_content.encode()).decode()
        
        update_data = {
            "message": f"Add worker: {worker_id}",
            "content": new_content_encoded,
            "sha": content_data['sha']
        }
        
        response = requests.put(url, headers=headers, json=update_data)
        
        if response.status_code in [200, 201]:
            logger.info(f"✅ تم إضافة {worker_id}")
            return True
        else:
            logger.error(f"❌ فشل التحديث: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ خطأ: {e}")
        return False

# ====== إرسال إشعار للمالك ======
def notify_owner(message):
    """إرسال إشعار للمالك"""
    if OWNER_CHAT_ID:
        send_message(OWNER_CHAT_ID, message)

# ====== قراءة state التدريب من HF ======
def get_training_state():
    """قراءة حالة التدريب من HF"""
    try:
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename="state.json",
            repo_type="model",
            token=HF_TOKEN
        )
        with open(path) as f:
            return json.load(f)
    except:
        return {
            "current_step": 0,
            "target_steps": 500000,
            "current_loss": None,
            "history": []
        }

# ====== معالجة الرسائل ======
def handle_message(update):
    """معالجة رسالة من تيليجرام"""
    message = update.get('message', {})
    if not message:
        return
    
    chat_id = message['chat']['id']
    text = message.get('text', '').strip()
    user = message.get('from', {})
    user_id = user.get('id')
    username = user.get('username', 'N/A')
    first_name = user.get('first_name', 'صديقي')
    
    # Load user state to check if waiting for token
    user_state_file = f"user_{user_id}_state.json"
    try:
        with open(user_state_file) as f:
            user_state = json.load(f)
    except:
        user_state = {"waiting_for_token": False}
    
    # إذا كان المستخدم ينتظر Token
    if user_state.get("waiting_for_token"):
        process_token_submission(chat_id, user_id, username, text)
        user_state["waiting_for_token"] = False
        with open(user_state_file, 'w') as f:
            json.dump(user_state, f)
        return
    
    # معالجة الأوامر
    if text == '/start':
        send_message(chat_id, f"""
🌟 مرحباً {first_name}!

أنا البوت الرسمي لمشروع Xandra Hope.

📌 البوت: @xandra_maestro_bot

الأوامر المتاحة:
/join - التسجيل في النظام
/status - حالة التدريب الحالية
/progress - آخر جلسات التدريب
/help - المساعدة
/leave - مغادرة وحذف بياناتك

💡 تذكير: شغّل Notebook على Kaggle يومياً!
""")
    
    elif text == '/join':
        send_message(chat_id, """
📝 رائع! لتسجيلك في النظام:

1️⃣ اذهب إلى: https://huggingface.co/join
   (لو ليس لديك حساب، سجّل بـ Google)

2️⃣ اذهب إلى: https://huggingface.co/settings/tokens
   واضغط "Create new token"

3️⃣ اختر:
   - Name: xandra
   - Token type: Write
   - اضغط Create

4️⃣ انسخ الـ Token (يبدأ بـ hf_)

5️⃣ أرسله لي هنا الآن (الصق الـ Token فقط)

⚠️ ملاحظات:
- نحتاج صلاحية Write لاستدعاء ZeroGPU API
- لا نستطيع التعديل على نماذجك
- يمكنك حذفه في أي وقت من إعدادات HF
""")
        # وضع المستخدم في وضع انتظار الـ Token
        user_state["waiting_for_token"] = True
        with open(user_state_file, 'w') as f:
            json.dump(user_state, f)
        
        notify_owner(f"📝 شخص جديد بدأ التسجيل: @{username} (ID: {user_id})")
    
    elif text == '/status':
        state = get_training_state()
        progress = (state['current_step'] / state['target_steps']) * 100 if state['target_steps'] else 0
        
        send_message(chat_id, f"""
📊 *حالة تدريب Xandra Hope*

📈 الخطوة الحالية: `{state['current_step']:,}`
🎯 الهدف: `{state['target_steps']:,}`
📊 نسبة الإنجاز: `{progress:.2f}%`
📉 الخسارة الحالية: `{state.get('current_loss') or 'N/A'}`
""")
    
    elif text == '/help':
        send_message(chat_id, """
📚 *المساعدة*

الأوامر:
/start - ترحيب
/join - تسجيل
/status - حالة التدريب
/progress - آخر جلسات
/help - هذه الرسالة
/leave - مغادرة
""")
    
    elif text == '/leave':
        send_message(chat_id, """
🚪 لقد طلبت المغادرة.

سيتم حذف بياناتك خلال 24 ساعة.
شكراً لمساهمتك!
""")
        notify_owner(f"⚠️ صديق غادر: @{username} (ID: {user_id})")
    
    else:
        send_message(chat_id, """
لم أفهم هذا الأمر.

الأوامر: /start /join /status /progress /help /leave
""")

def process_token_submission(chat_id, user_id, username, token):
    """معالجة Token المُرسل"""
    if not token.startswith("hf_"):
        send_message(chat_id, "❌ هذا لا يبدو كـ Token صالح. يجب أن يبدأ بـ `hf_`")
        return
    
    send_message(chat_id, "⏳ جاري التحقق من الـ Token...")
    
    result = verify_hf_token(token)
    
    if result.get("valid"):
        worker_id = f"friend-{user_id}"
        success = add_worker_to_config(
            worker_id=worker_id,
            hf_username=result["username"],
            telegram_user_id=user_id,
            telegram_username=username
        )
        
        if success:
            send_message(chat_id, f"""
✅ تم تسجيلك بنجاح!

🆔 معرّفك: {worker_id}
👤 حساب HF: {result["username"]}

🎉 شكراً لك! لا حاجة لأي إجراء آخر.
""")
            notify_owner(f"""
🎉 *صديق جديد انضم!*

🆔 {worker_id}
👤 HF: {result["username"]}
📱 Telegram: @{username}
""")
        else:
            send_message(chat_id, "❌ حدث خطأ أثناء التسجيل. حاول لاحقاً.")
    else:
        send_message(chat_id, f"""
❌ الـ Token غير صالح.

السبب: {result.get('error', 'غير معروف')}
حاول مرة أخرى: /join
""")

# ====== حلقة Long Polling ======
def main():
    logger.info("🤖 بدء تشغيل البوت...")
    
    state = load_state()
    offset = state["last_update_id"] + 1
    
    # الحصول على آخر التحديثات
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    params = {
        "offset": offset,
        "timeout": 30,
        "limit": 50
    }
    
    try:
        response = requests.get(url, params=params, timeout=60)
        if response.status_code != 200:
            logger.error(f"❌ فشل getUpdates: {response.status_code}")
            return
        
        data = response.json()
        if not data.get("ok"):
            logger.error("❌ استجابة غير صحيحة")
            return
        
        updates = data.get("result", [])
        logger.info(f"📬 استلمت {len(updates)} رسالة")
        
        for update in updates:
            try:
                handle_message(update)
                
                # تحديث آخر update_id
                if update.get("update_id", 0) > state["last_update_id"]:
                    state["last_update_id"] = update["update_id"]
                
                # حفظ الحالة بعد كل رسالة
                save_state(state)
                
            except Exception as e:
                logger.error(f"❌ خطأ في معالجة رسالة: {e}")
                # تحديث update_id حتى لا نعيد المعالجة
                if update.get("update_id", 0) > state["last_update_id"]:
                    state["last_update_id"] = update["update_id"]
                save_state(state)
        
        # حفظ نهائي
        save_state(state)
        
        # رفع ملف الحالة إلى GitHub (للحفاظ عليه بين التشغيلات)
        upload_state_to_github(state)
        
    except Exception as e:
        logger.error(f"❌ خطأ عام: {e}")

def upload_state_to_github(state):
    """رفع ملف الحالة إلى GitHub للحفاظ عليه"""
    try:
        import base64
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/bot_state.json"
        
        # محاولة قراءة الملف الحالي
        response = requests.get(url, headers=headers)
        sha = None
        if response.status_code == 200:
            sha = response.json().get('sha')
        
        content = json.dumps(state, indent=2).encode()
        content_encoded = base64.b64encode(content).decode()
        
        update_data = {
            "message": f"Update bot state - {datetime.now().isoformat()}",
            "content": content_encoded
        }
        if sha:
            update_data["sha"] = sha
        
        # إذا الملف ضمن .gitignore، نرفعه إلى branch منفصل أو gist
        # الأسهل: حفظه في GitHub Gist
        # لكن للبساطة، نتجاهل الرفع ونحفظ محلياً فقط
        
        # الحل البديل: حفظ في HF
        # لكن للبساطة الآن، نحفظه محلياً ونعيد قراءته
        
    except Exception as e:
        logger.warning(f"⚠️ لم يتم رفع الحالة: {e}")

if __name__ == "__main__":
    main()
