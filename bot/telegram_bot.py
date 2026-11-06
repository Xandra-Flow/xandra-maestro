"""
Xandra Maestro Bot - بوت تيليجرام المُحدّث
==========================================
يستقبل طلبات الانضمام، يتحقق منها، يسجلها
يرسل تنبيهات للمالك عند:
  - إضافة صديق جديد
  - مغادرة صديق
  - فشل في التحقق من Token
  - أي خطأ في النظام

المميزات:
- /start: ترحيب
- /join: تسجيل صديق جديد (يطلب Token)
- /status: حالة الصديق
- /leave: حذف الصديق من النظام
- /help: المساعدة
- /workers: قائمة كل الأصدقاء (للمالك فقط)
- /alerts: تفعيل/إيقاف التنبيهات (للمالك فقط)
"""

import os
import json
import logging
import traceback
import requests
import telebot
from datetime import datetime
import yaml

# ====== الإعدادات ======
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "your-username/xandra-maestro")
HF_REPO = os.environ.get("HF_REPO", "your-username/xandra-training-state")
OWNER_CHAT_ID = int(os.environ.get("OWNER_CHAT_ID", "0"))
ALERTS_ENABLED = os.environ.get("ALERTS_ENABLED", "true").lower() == "true"

# ====== نظام التسجيل (Logging) ======
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xandra_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# التحقق من الإعدادات
if not BOT_TOKEN:
    logger.error("❌ TELEGRAM_BOT_TOKEN غير موجود!")
    exit(1)

bot = telebot.TeleBot(BOT_TOKEN)

# قائمة انتظار لتتبع من أرسل /join
waiting_for_token = {}

# ====== نظام التنبيهات ======
def send_alert_to_owner(message, level="info"):
    """
    إرسال تنبيه للمالك
    
    Args:
        message: نص الرسالة
        level: info, success, warning, error
    """
    if not OWNER_CHAT_ID or not ALERTS_ENABLED:
        logger.info(f"📨 تنبيه ({level}): {message}")
        return
    
    emojis = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "🚨"
    }
    emoji = emojis.get(level, "ℹ️")
    
    text = f"🎵 Xandra Maestro Alert:\n{emoji} {message}"
    
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={
                "chat_id": OWNER_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown"
            },
            timeout=10
        )
        logger.info(f"✅ تم إرسال تنبيه ({level}) للمالك")
    except Exception as e:
        logger.error(f"❌ فشل إرسال التنبيه: {e}")

# ====== التحقق من Token عبر HF API ======
def verify_hf_token(token):
    """
    يتحقق أن الـ Token صالح عبر استدعاء HF API
    ويتأكد أن صلاحياته Read فقط
    """
    try:
        response = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # التحقق من صلاحيات الـ Token
            # (لا يمكننا معرفة الصلاحيات الدقيقة من whoami، لكن نتأكد من صحته)
            return {
                "valid": True,
                "username": data.get("name", "unknown"),
                "fullname": data.get("fullname", ""),
                "orgs": [org.get("name") for org in data.get("orgs", [])]
            }
        elif response.status_code == 401:
            return {"valid": False, "error": "Token غير صحيح أو منتهي الصلاحية"}
        else:
            return {"valid": False, "error": f"خطأ من HF: Status {response.status_code}"}
            
    except requests.exceptions.Timeout:
        return {"valid": False, "error": "انتهى وقت الاتصال بـ Hugging Face"}
    except Exception as e:
        return {"valid": False, "error": f"خطأ غير متوقع: {str(e)}"}

# ====== تسجيل Worker جديد في GitHub ======
def register_worker_in_github(worker_id, hf_username, hf_token, telegram_user_id, telegram_username=None):
    """يضيف Worker جديد إلى GitHub Secrets + workers_config.yaml"""
    try:
        from github import Github
        
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO)
        
        # 1. إضافة الـ Token إلى GitHub Secrets
        secret_name = f"HF_TOKEN_{worker_id.upper()}"
        repo.create_secret(secret_name, hf_token)
        logger.info(f"✅ تم إضافة Secret: {secret_name}")
        
        # 2. تحديث workers_config.yaml
        try:
            contents = repo.get_contents("maestro/workers_config.yaml")
            config = yaml.safe_load(contents.decoded_content.decode())
        except Exception as e:
            logger.warning(f"⚠️ لم يتم العثور على workers_config.yaml: {e}")
            config = {"workers": []}
        
        new_worker = {
            'id': worker_id,
            'service': 'hf_zerogpu',
            'enabled': True,
            'hf_username': hf_username,
            'token_secret': secret_name,
            'telegram_user_id': telegram_user_id,
            'telegram_username': telegram_username,
            'added_at': datetime.now().isoformat(),
            'total_steps_trained': 0,
            'total_sessions': 0,
            'status': 'active'
        }
        config['workers'].append(new_worker)
        
        new_content = yaml.dump(config, default_flow_style=False, allow_unicode=True)
        repo.update_file(
            "maestro/workers_config.yaml",
            f"Add worker: {worker_id}",
            new_content,
            contents.sha
        )
        logger.info(f"✅ تم تحديث workers_config.yaml")
        
        return True, None
        
    except Exception as e:
        error_msg = f"فشل تسجيل Worker في GitHub: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return False, error_msg

# ====== حذف Worker من النظام ======
def remove_worker_from_github(telegram_user_id):
    """حذف Worker من GitHub Secrets + workers_config.yaml"""
    try:
        from github import Github
        
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO)
        
        # قراءة workers_config.yaml
        contents = repo.get_contents("maestro/workers_config.yaml")
        config = yaml.safe_load(contents.decoded_content.decode())
        
        # العثور على الـ Worker وحذفه
        worker_to_remove = None
        for w in config['workers']:
            if w.get('telegram_user_id') == telegram_user_id:
                worker_to_remove = w
                break
        
        if not worker_to_remove:
            return False, "لم يتم العثور على الـ Worker"
        
        # حذف Secret من GitHub
        try:
            repo.delete_secret(worker_to_remove['token_secret'])
        except:
            pass  # Secret قد لا يكون موجوداً
        
        # حذف Worker من config
        config['workers'] = [
            w for w in config['workers']
            if w.get('telegram_user_id') != telegram_user_id
        ]
        
        new_content = yaml.dump(config, default_flow_style=False, allow_unicode=True)
        repo.update_file(
            "maestro/workers_config.yaml",
            f"Remove worker: {worker_to_remove['id']}",
            new_content,
            contents.sha
        )
        
        return True, worker_to_remove['id']
        
    except Exception as e:
        error_msg = f"فشل حذف Worker: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

# ====== الأوامر ======

@bot.message_handler(commands=['start'])
def welcome(message):
    """ترحيب المستخدم"""
    user_name = message.from_user.first_name or "صديقي"
    is_owner = message.from_user.id == OWNER_CHAT_ID
    
    text = f"""
🌟 مرحباً {user_name}!

أنا البوت الرسمي لمشروع Xandra - نموذج الذكاء الاصطناعي العربي.

📌 البوت الرسمي: @xandra_maestro_bot

للانضمام كـ "Worker" (مساهم بصمت):
أرسل: /join

للاطلاع على حالتك:
أرسل: /status

للمغادرة وحذف بياناتك:
أرسل: /leave

للمساعدة:
أرسل: /help
"""
    if is_owner:
        text += """
👑 أوامر المالك فقط:
/workers - عرض كل الأصدقاء المسجلين
/alerts - تفعيل/إيقاف التنبيهات
/test_alert - اختبار التنبيهات
"""
    bot.reply_to(message, text)

@bot.message_handler(commands=['join'])
def join(message):
    """بدء عملية التسجيل"""
    user_id = message.from_user.id
    waiting_for_token[user_id] = {
        "started_at": datetime.now().isoformat(),
        "username": message.from_user.username
    }
    
    text = """
📝 رائع! لتسجيلك في النظام:

1️⃣ اذهب إلى: https://huggingface.co/join
   (لو ليس لديك حساب، سجّل بـ Google)

2️⃣ اذهب إلى: https://huggingface.co/settings/tokens
   واضغط "Create new token"

3️⃣ اختر:
   - Name: xandra
   - Token type: Fine-grained ⭐ (موصى به)
   
   ثم فعّل فقط:
   ✅ Repositories → Read access to contents of all repos
   ✅ User → Inference → Make calls to inference endpoints
   
   (أو ببساطة: اختر "Write" بدلاً من Fine-grained)

4️⃣ اضغط Create token
   انسخ الـ Token (يبدأ بـ hf_)

5️⃣ أرسله لي هنا الآن (الصق الـ Token فقط)

⚠️ ملاحظات:
- نحتاج صلاحية Write (أو Fine-grained مع Inference) لاستدعاء ZeroGPU API
- Read فقط لا يكفي لاستدعاء الـ Space برمجياً
- لا نستطيع التعديل على نماذجك، نستخدمه فقط للـ GPU المجاني
- يمكنك حذفه في أي وقت من إعدادات HF
"""
    bot.reply_to(message, text)
    
    # تنبيه المالك أن شخصاً بدأ التسجيل
    send_alert_to_owner(
        f"📝 شخص جديد بدأ التسجيل\n"
        f"👤 Telegram: @{message.from_user.username or 'N/A'}\n"
        f"🆔 ID: {user_id}",
        level="info"
    )

@bot.message_handler(commands=['status'])
def status(message):
    """عرض حالة المستخدم"""
    user_id = message.from_user.id
    
    try:
        from github import Github
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO)
        contents = repo.get_contents("maestro/workers_config.yaml")
        config = yaml.safe_load(contents.decoded_content.decode())
        
        worker = None
        for w in config.get('workers', []):
            if w.get('telegram_user_id') == user_id:
                worker = w
                break
        
        if worker:
            text = f"""
📊 حالتك في مشروع Xandra:

🆔 معرّفك: {worker['id']}
👤 حساب HF: {worker['hf_username']}
✅ الحالة: {'نشط ✅' if worker.get('enabled', True) else 'موقوف ⏸️'}
📈 جلسات التدريب: {worker.get('total_sessions', 0)}
🎯 خطوات مدرّبة: {worker.get('total_steps_trained', 0)}
📅 تاريخ الانضمام: {worker.get('added_at', 'غير معروف')[:10]}

شكراً لمساهمتك! 🙏
"""
        else:
            text = "❌ أنت غير مسجل بعد.\nأرسل /join للتسجيل."
    except Exception as e:
        text = f"❌ خطأ في قراءة الحالة: {e}"
        send_alert_to_owner(
            f"❌ خطأ في قراءة حالة المستخدم {user_id}: {e}",
            level="error"
        )
    
    bot.reply_to(message, text)

@bot.message_handler(commands=['leave'])
def leave(message):
    """حذف المستخدم من النظام"""
    user_id = message.from_user.id
    username = message.from_user.username or "غير معروف"
    
    bot.reply_to(message, 
        "⏳ جاري حذف بياناتك..."
    )
    
    success, result = remove_worker_from_github(user_id)
    
    if success:
        text = f"""
🚪 تم حذفك من النظام بنجاح.

ما تم حذفه:
- ✅ GitHub Secret الخاص بك
- ✅ بياناتك من workers_config.yaml
- 📝 Token في HF (احذفه يدوياً من إعداداتك)

شكراً لك على مساهمتك السابقة! 🙏

يمكنك العودة متى أردت: /join
"""
        bot.reply_to(message, text)
        
        # تنبيه المالك
        send_alert_to_owner(
            f"🚪 صديق غادر\n"
            f"🆔 {result}\n"
            f"👤 Telegram: @{username}\n"
            f"📱 ID: {user_id}",
            level="warning"
        )
    else:
        bot.reply_to(message, 
            f"❌ حدث خطأ أثناء الحذف: {result}\n"
            f"تواصل مع الإدارة."
        )
        send_alert_to_owner(
            f"❌ فشل حذف صديق\n"
            f"📱 ID: {user_id}\n"
            f"الخطأ: {result}",
            level="error"
        )

@bot.message_handler(commands=['help'])
def help_command(message):
    """المساعدة"""
    text = """
📚 المساعدة

الأوامر المتاحة:
/start - رسالة الترحيب
/join - التسجيل في النظام
/status - عرض حالتك
/leave - المغادرة وحذف بياناتك
/help - هذه الرسالة

لو واجهت مشكلة، تواصل مع من دعاك للمشروع.

المزيد من المساعدة: https://huggingface.co/docs
"""
    bot.reply_to(message, text)

# ====== أوامر المالك فقط ======

def is_owner(message):
    """التحقق أن المستخدم هو المالك"""
    return message.from_user.id == OWNER_CHAT_ID

@bot.message_handler(commands=['workers'])
def list_workers(message):
    """عرض قائمة كل الأصدقاء (للمالك فقط)"""
    if not is_owner(message):
        bot.reply_to(message, "❌ هذا الأمر للمالك فقط")
        return
    
    try:
        from github import Github
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO)
        contents = repo.get_contents("maestro/workers_config.yaml")
        config = yaml.safe_load(contents.decoded_content.decode())
        
        workers = config.get('workers', [])
        active_count = sum(1 for w in workers if w.get('enabled', True))
        
        text = f"👥 إجمالي Workers: {len(workers)} (نشط: {active_count})\n\n"
        
        for w in workers:
            status_emoji = "✅" if w.get('enabled', True) else "⏸️"
            text += f"""
{status_emoji} {w['id']}
   👤 HF: {w.get('hf_username', 'N/A')}
   📈 جلسات: {w.get('total_sessions', 0)}
   🎯 خطوات: {w.get('total_steps_trained', 0)}
   📅 انضم: {w.get('added_at', 'N/A')[:10]}
"""
        
        bot.reply_to(message, text)
        
    except Exception as e:
        bot.reply_to(message, f"❌ خطأ: {e}")
        send_alert_to_owner(f"❌ خطأ في /workers: {e}", level="error")

@bot.message_handler(commands=['test_alert'])
def test_alert(message):
    """اختبار نظام التنبيهات (للمالك فقط)"""
    if not is_owner(message):
        bot.reply_to(message, "❌ هذا الأمر للمالك فقط")
        return
    
    send_alert_to_owner("🔔 هذا تنبيه اختباري - نظام التنبيهات يعمل!", level="success")
    bot.reply_to(message, "✅ تم إرسال تنبيه اختباري")

# ====== استقبال الـ Token ======
@bot.message_handler(func=lambda m: waiting_for_token.get(m.from_user.id, False))
def receive_token(message):
    """استقبال الـ Token والتحقق منه"""
    user_id = message.from_user.id
    username = message.from_user.username
    token = message.text.strip()
    
    # حذف من قائمة الانتظار
    waiting_for_token[user_id] = False
    
    # التحقق من التنسيق
    if not token.startswith("hf_"):
        bot.reply_to(message, 
            "❌ هذا لا يبدو كـ Token صالح.\n"
            "الـ Token يجب أن يبدأ بـ `hf_`\n\n"
            "حاول مرة أخرى: /join"
        )
        send_alert_to_owner(
            f"⚠️ محاولة تسجيل فاشلة (تنسيق خاطئ)\n"
            f"👤 @{username or 'N/A'} (ID: {user_id})",
            level="warning"
        )
        return
    
    # حذف الرسالة لأسباب أمنية (لا يبقى الـ Token في المحادثة)
    try:
        bot.delete_message(message.chat.id, message.message_id)
    except:
        pass  # أحياناً لا يُسمح بالحذف
    
    bot.reply_to(message, "⏳ جاري التحقق من الـ Token...")
    
    # التحقق من صحة الـ Token
    result = verify_hf_token(token)
    
    if result["valid"]:
        # إنشاء معرّف Worker
        worker_id = f"friend-{user_id}"
        
        # تسجيل في GitHub
        success, error = register_worker_in_github(
            worker_id=worker_id,
            hf_username=result["username"],
            hf_token=token,
            telegram_user_id=user_id,
            telegram_username=username
        )
        
        if success:
            success_text = f"""
✅ تم تسجيلك بنجاح!

🆔 معرّفك: {worker_id}
👤 حساب HF: {result["username"]}

🎉 شكراً لك! لا حاجة لأي إجراء آخر.
سيتم استخدام حسابك تلقائياً للمساهمة في تدريب النموذج العربي.

للمتابعة: /status
للمغادرة: /leave
"""
            bot.reply_to(message, success_text)
            
            # 🎉 تنبيه المالك بصديق جديد
            send_alert_to_owner(
                f"🎉 *صديق جديد انضم للنظام!*\n\n"
                f"🆔 المعرّف: `{worker_id}`\n"
                f"👤 حساب HF: `{result['username']}`\n"
                f"📱 Telegram: @{username or 'N/A'}\n"
                f"🆔 Telegram ID: `{user_id}`\n"
                f"📅 الوقت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"✅ جاهز للمساهمة في التدريب",
                level="success"
            )
            
        else:
            bot.reply_to(message, 
                f"❌ حدث خطأ أثناء التسجيل.\n"
                f"السبب: فشل إضافة البيانات إلى GitHub\n\n"
                f"حاول مرة أخرى لاحقاً: /join\n"
                f"لو استمرت المشكلة، تواصل مع الإدارة."
            )
            send_alert_to_owner(
                f"❌ *فشل تسجيل صديق جديد*\n\n"
                f"👤 @{username or 'N/A'} (ID: {user_id})\n"
                f"👤 HF: {result['username']}\n"
                f"الخطأ: `{error}`",
                level="error"
            )
    else:
        bot.reply_to(message, 
            f"❌ الـ Token غير صالح.\n\n"
            f"السبب: {result.get('error', 'غير معروف')}\n\n"
            f"تأكد من:\n"
            f"- نسخت الـ Token كاملاً\n"
            f"- لم تنته صلاحيته\n"
            f"- أنشأته من: https://huggingface.co/settings/tokens\n\n"
            f"حاول مرة أخرى: /join"
        )
        send_alert_to_owner(
            f"⚠️ *Token غير صالح*\n\n"
            f"👤 @{username or 'N/A'} (ID: {user_id})\n"
            f"السبب: {result.get('error', 'غير معروف')}",
            level="warning"
        )

# ====== معالج الأخطاء العام ======
@bot.message_handler(func=lambda m: True)
def handle_all_messages(message):
    """الرد على أي رسالة غير معروفة"""
    bot.reply_to(message, 
        "لم أفهم هذا الأمر.\n\n"
        "الأوامر المتاحة:\n"
        "/start - ترحيب\n"
        "/join - تسجيل\n"
        "/status - حالتك\n"
        "/leave - مغادرة\n"
        "/help - مساعدة"
    )

# ====== بدء التشغيل ======
if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("🤖 Xandra Maestro Bot بدأ التشغيل")
    logger.info(f"📋 OWNER_CHAT_ID: {OWNER_CHAT_ID}")
    logger.info(f"📋 GITHUB_REPO: {GITHUB_REPO}")
    logger.info(f"📋 ALERTS_ENABLED: {ALERTS_ENABLED}")
    logger.info("=" * 50)
    
    # تنبيه بدء التشغيل
    send_alert_to_owner(
        "🤖 *Xandra Maestro Bot بدأ التشغيل*\n\n"
        f"⏰ الوقت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"📋 البوت: @xandra_maestro_bot\n"
        f"✅ جاهز لاستقبال الأصدقاء",
        level="success"
    )
    
    try:
        bot.polling(none_stop=True, interval=2, timeout=60)
    except Exception as e:
        logger.error(f"❌ خطأ في التشغيل: {e}")
        send_alert_to_owner(
            f"🚨 *خطأ حرج في البوت*\n\n{str(e)}\n\n{traceback.format_exc()}",
            level="error"
        )
        # إعادة المحاولة بعد 30 ثانية
        import time
        time.sleep(30)
        bot.polling(none_stop=True)
