"""
Xandra Maestro - الإصدار 3.0 (GitHub Actions)
يراقب state من HF، يرسل إشعارات، ويكتشف النضج
"""
import os
import json
import logging
import requests
from datetime import datetime
from huggingface_hub import hf_hub_download, HfApi

HF_REPO = os.environ.get("HF_REPO")
HF_TOKEN = os.environ.get("HF_TOKEN_OWNER")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
OWNER_CHAT_ID = os.environ.get("OWNER_CHAT_ID")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STEP_MILESTONES = [100, 500, 1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000]
LOSS_MILESTONES = [10, 5, 4, 3, 2.5, 2, 1.5]

def send_alert(message, level="info"):
    if not TELEGRAM_BOT_TOKEN or not OWNER_CHAT_ID:
        logger.info(f"📨 ({level}): {message}")
        return
    
    emojis = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "🚨", "celebrate": "🎉", "progress": "📊"}
    emoji = emojis.get(level, "ℹ️")
    text = f"🎵 Xandra Maestro:\n{emoji} {message}"
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        response = requests.post(url, json={"chat_id": OWNER_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=15)
        if response.status_code == 200: logger.info(f"✅ تم إرسال إشعار ({level})")
    except Exception as e:
        logger.error(f"❌ فشل إرسال: {e}")

def get_training_state():
    try:
        path = hf_hub_download(repo_id=HF_REPO, filename="state.json", repo_type="model", token=HF_TOKEN)
        with open(path) as f: return json.load(f)
    except Exception as e:
        logger.warning(f"⚠️ لا توجد حالة: {e}")
        return {"current_step": 0, "target_steps": 500000, "current_loss": None, "history": [], "best_loss": float('inf'), "last_updated": None}

def check_milestones(state):
    step = state.get('current_step', 0)
    loss = state.get('current_loss')
    try:
        with open("celebrated.json") as f: celebrated = set(json.load(f))
    except: celebrated = set()
    
    for milestone in STEP_MILESTONES:
        key = f"step_{milestone}"
        if step >= milestone and key not in celebrated:
            progress = (step / state['target_steps']) * 100 if state['target_steps'] else 0
            send_alert(f"*🎉 مرحلة مهمة!*\n\nوصلنا للخطوة `{milestone:,}`\n📊 نسبة الإنجاز: `{progress:.2f}%`", level="celebrate")
            celebrated.add(key)
    
    if loss is not None:
        for milestone in LOSS_MILESTONES:
            key = f"loss_{milestone}"
            if loss < milestone and key not in celebrated:
                messages = {10: "النموذج تخرج من مرحلة الهراء!", 5: "النموذج بدأ ينتج نصاً مفهوماً!", 4: "النموذج بدأ يفهم العربية!", 3: "النموذج أصبح جيداً!", 2.5: "النموذج أصبح قوياً جداً!", 2: "النموذج وصل لنضج متقدم!", 1.5: "النموذج في مستوى ممتاز!"}
                send_alert(f"*🎉 معلم خسارة مهم!*\n\nالخسارة وصلت لـ أقل من `{milestone}`\n📉 الخسارة: `{loss:.4f}`\n📈 الخطوة: `{step:,}`\n\n📌 {messages.get(milestone, 'إنجاز!')}", level="celebrate")
                celebrated.add(key)
    
    with open("celebrated.json", 'w') as f: json.dump(list(celebrated), f)

def check_stale(state):
    history = state.get('history', [])
    if history:
        last_session = history[-1]
        try:
            last_time_str = last_session.get('timestamp', '').replace('Z', '+00:00')
            last_time = datetime.fromisoformat(last_time_str)
            hours_since = (datetime.now(last_time.tzinfo) - last_time).total_seconds() / 3600
            if hours_since > 48:
                send_alert(f"*⚠️ توقف التدريب!*\n\nآخر جلسة كانت منذ `{hours_since:.0f}` ساعة\n📌 يجب تشغيل Notebook!", level="warning")
        except Exception as e:
            logger.warning(f"⚠️ خطأ في حساب الوقت: {e}")

def main():
    logger.info(f"🎵 Xandra Maestro v3.0 - {datetime.now()}")
    state = get_training_state()
    logger.info(f"📊 الخطوة الحالية: {state.get('current_step', 0)}")
    check_milestones(state)
    check_stale(state)
    logger.info("✅ انتهت الدورة")

if __name__ == "__main__":
    main()
