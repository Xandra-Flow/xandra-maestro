"""
Xandra Telegram Bot - الإصدار 2.0 (Render.com)
================================================
يعمل على Render.com Free Tier
يدعم: /start, /status, /progress, /chat, /test_model, /help

مميزات خاصة بـ Render:
- يستخدم Webhook بدلاً من polling (أكثر كفاءة)
- يبقى نائماً ويستيقظ عند الرسائل
- يحفظ state في ملف محلي (Render persistent disk)
"""

import os
import json
import logging
import requests
from flask import Flask, request, jsonify
from huggingface_hub import hf_hub_download
import telebot

# ====== الإعدادات ======
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
HF_REPO = os.environ.get("HF_REPO", "your-username/xandra-training-state")
HF_TOKEN = os.environ.get("HF_TOKEN_OWNER")
OWNER_CHAT_ID = int(os.environ.get("OWNER_CHAT_ID", "0"))
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "")

# ====== التسجيل ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== Flask App ======
app = Flask(__name__)
bot = telebot.TeleBot(BOT_TOKEN)

# ====== قراءة state من HF ======
def get_training_state():
    """قراءة آخر حالة تدريب من HF"""
    try:
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename="state.json",
            repo_type="model",
            token=HF_TOKEN
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"⚠️ لا توجد حالة: {e}")
        return {
            "current_step": 0,
            "target_steps": 500000,
            "current_loss": None,
            "history": [],
            "best_loss": float('inf')
        }

# ====== الأوامر ======

@bot.message_handler(commands=['start'])
def welcome(message):
    """ترحيب"""
    user_name = message.from_user.first_name or "صديقي"
    text = f"""
🌟 مرحباً {user_name}!

أنا البوت الرسمي لمشروع Xandra Hope.

📌 البوت: @xandra_maestro_bot

الأوامر المتاحة:
/status - حالة التدريب الحالية
/progress - تقرير التقدم
/help - المساعدة

💡 تذكير: شغّل Notebook على Kaggle يومياً!
"""
    bot.reply_to(message, text)

@bot.message_handler(commands=['status'])
def status(message):
    """حالة التدريب"""
    state = get_training_state()
    
    progress = (state['current_step'] / state['target_steps']) * 100
    steps_remaining = state['target_steps'] - state['current_step']
    
    # تقدير الوقت المتبقي
    history = state.get('history', [])
    if history and len(history) >= 2:
        avg_steps_per_session = sum(h.get('steps_done', 0) for h in history[-7:]) / min(7, len(history))
        if avg_steps_per_session > 0:
            days_remaining = steps_remaining / avg_steps_per_session
            time_estimate = f"~{int(days_remaining)} يوم"
        else:
            time_estimate = "غير محدد"
    else:
        time_estimate = "غير محدد"
    
    text = f"""
📊 *حالة تدريب Xandra Hope*

📈 الخطوة الحالية: `{state['current_step']:,}`
🎯 الهدف: `{state['target_steps']:,}`
📊 نسبة الإنجاز: `{progress:.2f}%`
📉 الخسارة الحالية: `{state.get('current_loss', 'N/A')}`
🏆 أفضل خسارة: `{state.get('best_loss', 'N/A')}`
⏳ الوقت المتبقي: `{time_estimate}`
📅 آخر تحديث: `{state.get('last_updated', 'غير معروف')[:19]}`

💡 لتشغيل جلسة تدريب:
1. اذهب إلى Kaggle
2. افتح Notebook: xandra_worker
3. اضغط Run All
"""
    bot.reply_to(message, text, parse_mode="Markdown")

@bot.message_handler(commands=['progress'])
def progress(message):
    """تقرير مفصل عن التقدم"""
    state = get_training_state()
    history = state.get('history', [])
    
    if not history:
        bot.reply_to(message, "❌ لا يوجد سجل تدريب بعد.")
        return
    
    # آخر 5 جلسات
    recent = history[-5:]
    
    text = "📈 *آخر جلسات التدريب:*\n\n"
    for h in reversed(recent):
        text += (
            f"📅 `{h.get('timestamp', 'N/A')[:19]}`\n"
            f"   خطوة: `{h.get('final_step', 'N/A'):,}`\n"
            f"   خسارة: `{h.get('final_loss', 'N/A'):.4f}`\n"
            f"   خطوات اليوم: `{h.get('steps_done', 'N/A')}`\n\n"
        )
    
    text += (
        f"📊 *إجمالي:*\n"
        f"   الخطوة الحالية: `{state['current_step']:,}`\n"
        f"   نسبة الإنجاز: `{(state['current_step']/state['target_steps']*100):.2f}%`"
    )
    
    bot.reply_to(message, text, parse_mode="Markdown")

@bot.message_handler(commands=['help'])
def help_command(message):
    """المساعدة"""
    text = """
📚 *المساعدة - مشروع Xandra Hope*

🤖 *الأوامر:*
/start - رسالة الترحيب
/status - حالة التدريب الحالية
/progress - آخر جلسات التدريب
/help - هذه الرسالة

💡 *كيف أساهم في التدريب؟*

1. اذهب إلى https://www.kaggle.com/
2. افتح Notebook: `xandra_worker`
3. اضغط **Run All** (Ctrl+F9)
4. اغلق المتصفح بأمان
5. سيعمل Notebook 12 ساعة تلقائياً

📌 *النتيجة:*
- ~500-1000 خطوة تدريب
- حفظ checkpoint على HF
- إشعار تلقائي لك عبر هذا البوت

❓ *أسئلة شائعة:*

س: كم مرة يجب التشغيل؟
ج: يومياً إن أمكن. حتى مرة كل 3 أيام كافية.

س: ماذا لو فشلت الجلسة؟
ج: لا مشكلة، الشغّل مرة أخرى. النظام يحفظ التقدم.

س: هل أحتاج GPU قوي؟
ج: لا، T4 على Kaggle كافٍ تماماً.
"""
    bot.reply_to(message, text, parse_mode="Markdown")

@bot.message_handler(commands=['test_model'])
def test_model(message):
    """اختبار النموذج (تحت التطوير)"""
    text = """
🧪 *اختبار النموذج*

هذه الميزة قيد التطوير.

في الوقت الحالي، يمكنك:
1. تنزيل آخر checkpoint من HF
2. تحميله محلياً
3. اختباره بـ Python

📎 الرابط:
https://huggingface.co/your-username/xandra-training-state/blob/main/checkpoints/latest.pt

📌 أو انتظر التحديث القادم سيحتوي على اختبار مباشر.
"""
    bot.reply_to(message, text, parse_mode="Markdown")

@bot.message_handler(func=lambda m: True)
def handle_all(message):
    """الرد على أي رسالة غير معروفة"""
    bot.reply_to(message, 
        "لم أفهم هذا الأمر.\n\n"
        "الأوامر: /start /status /progress /help"
    )

# ====== Webhook endpoints ======

@app.route('/', methods=['GET'])
def index():
    """صفحة رئيسية بسيطة"""
    return """
    <html>
        <head><title>Xandra Maestro Bot</title></head>
        <body style="font-family: Arial; text-align: center; padding: 50px;">
            <h1>🎵 Xandra Maestro Bot</h1>
            <p>البوت يعمل بنجاح ✅</p>
            <p>افتح البوت على تيليجرام: <a href="https://t.me/xandra_maestro_bot">@xandra_maestro_bot</a></p>
        </body>
    </html>
    """

@app.route('/webhook/' + BOT_TOKEN, methods=['POST'])
def webhook():
    """استقبال رسائل تيليجرام عبر Webhook"""
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return jsonify({"status": "ok"})
    return jsonify({"status": "error"}), 400

@app.route('/set_webhook', methods=['GET'])
def set_webhook():
    """ضبط الـ Webhook (شغّله مرة واحدة بعد النشر)"""
    if not RENDER_EXTERNAL_URL:
        return "❌ RENDER_EXTERNAL_URL غير مُعد"
    
    webhook_url = f"{RENDER_EXTERNAL_URL}/webhook/{BOT_TOKEN}"
    response = bot.set_webhook(url=webhook_url)
    
    if response:
        return f"✅ تم ضبط Webhook: {webhook_url}"
    return "❌ فشل ضبط Webhook"

@app.route('/health', methods=['GET'])
def health():
    """فحص الصحة (لـ Render)"""
    return jsonify({
        "status": "ok",
        "bot": "xandra_maestro",
        "version": "2.0"
    })

# ====== التشغيل ======
if __name__ == '__main__':
    # على Render، نستخدم Webhook
    port = int(os.environ.get('PORT', 5000))
    
    if RENDER_EXTERNAL_URL:
        # ضبط الـ webhook تلقائياً
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook/{BOT_TOKEN}"
        bot.remove_webhook()
        bot.set_webhook(url=webhook_url)
        logger.info(f"✅ Webhook مضبوط: {webhook_url}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
