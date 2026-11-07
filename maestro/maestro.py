"""
Xandra Maestro - الإصدار 3.0 (GitHub Actions)
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
    """إرسال إشعار للمالك عبر Telegram API مباشرة"""
    if not TELEGRAM_BOT_TOKEN or not OWNER_CHAT_ID:
        logger.info(f"📨 ({level}): {message}")
        return
    
    emojis = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "🚨", "celebrate": "🎉", "progress": "📊"}
    emoji = emojis.get(level, "ℹ️")
    text = f"🎵 Xandra Maestro:\n{emoji} {message}"
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        response = requests.post(url, json={
            "chat_id": OWNER_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown"
        }, timeout=15)
        if response.status_code == 200:
            logger.info(f"✅ تم إرسال إشعار ({level})")
        else:
            logger.error(f"❌ فشل: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ فشل إرسال: {e}")

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
            "best_loss": float('inf'),
            "last_updated": None
        }

def check_milestones(state):
    """فحص المراحل الاحتفالية"""
    step = state.get('current_step', 0)
    loss = state.get('current_loss')
    
    # قراءة المراحل المحتفل بها
    try:
        with open("celebrated.json") as f:
            celebrated = set(json.load(f))
    except:
        celebrated = set()
    
    # فحص مراحل الخطوات
    for milestone in STEP_MILESTONES:
        key = f"step_{milestone}"
        if step >= milestone and key not in celebrated:
            progress = (step / state['target_steps']) * 100 if state['target_steps'] else 0
            send_alert(
                f"*🎉 مرحلة مهمة!*\n\n"
                f"وصلنا للخطوة `{milestone:,}`\n"
                f"📊 نسبة الإنجاز: `{progress:.2f}%`\n"
                f"📉 الخسارة: `{loss or 'N/A'}`",
                level="celebrate"
            )
            celebrated.add(key)
    
    # فحص مراحل الخسارة
    if loss is not None:
        for milestone in LOSS_MILESTONES:
            key = f"loss_{milestone}"
            if loss < milestone and key not in celebrated:
                messages = {
                    10: "النموذج تخرج من مرحلة الهراء!",
                    5: "النموذج بدأ ينتج نصاً مفهوماً!",
                    4: "النموذج بدأ يفهم العربية!",
                    3: "النموذج أصبح جيداً!",
                    2.5: "النموذج أصبح قوياً جداً!",
                    2: "النموذج وصل لنضج متقدم!",
                    1.5: "النموذج في مستوى ممتاز!"
                }
                msg = messages.get(milestone, "إنجاز!")
                send_alert(
                    f"*🎉 معلم خسارة مهم!*\n\n"
                    f"الخسارة وصلت لـ أقل من `{milestone}`\n"
                    f"📉 الخسارة: `{loss:.4f}`\n"
                    f"📈 الخطوة: `{step:,}`\n\n"
                    f"📌 {msg}",
                    level="celebrate"
                )
                celebrated.add(key)
    
    # حفظ المراحل
    with open("celebrated.json", 'w') as f:
        json.dump(list(celebrated), f)

def check_stale(state):
    """فحص هل آخر تحديث قديم"""
    history = state.get('history', [])
    if history:
        last_session = history[-1]
        try:
            last_time_str = last_session.get('timestamp', '')
            # محاولة تحليل التاريخ
            last_time_str = last_time_str.replace('Z', '+00:00')
            last_time = datetime.fromisoformat(last_time_str)
            hours_since = (datetime.now(last_time.tzinfo) - last_time).total_seconds() / 3600
            if hours_since > 48:
                send_alert(
                    f"*⚠️ توقف التدريب!*\n\n"
                    f"آخر جلسة كانت منذ `{hours_since:.0f}` ساعة\n"
                    f"📌 يجب تشغيل Notebook على Kaggle!",
                    level="warning"
                )
        except Exception as e:
            logger.warning(f"⚠️ خطأ في حساب الوقت: {e}")

def send_daily_report(state):
    """إرسال تقرير يومي"""
    history = state.get('history', [])
    if not history:
        return
    
    # آخر 24 ساعة
    now = datetime.now()
    last_24h = []
    for h in history:
        try:
            ts = h.get('timestamp', '').replace('Z', '+00:00')
            t = datetime.fromisoformat(ts)
            if (now - t.replace(tzinfo=None)).total_seconds() < 86400:
                last_24h.append(h)
        except:
            pass
    
    if not last_24h:
        return
    
    steps_today = sum(h.get('steps_done', 0) for h in last_24h)
    progress = (state['current_step'] / state['target_steps']) * 100 if state['target_steps'] else 0
    
    send_alert(
        f"*📊 تقرير اليوم*\n\n"
        f"📈 جلسات اليوم: `{len(last_24h)}`\n"
        f"🎯 خطوات اليوم: `{steps_today:,}`\n"
        f"📊 إجمالي الخطوات: `{state['current_step']:,}`\n"
        f"📊 نسبة الإنجاز: `{progress:.2f}%`\n"
        f"📉 الخسارة: `{state.get('current_loss') or 'N/A'}`",
        level="progress"
    )

def main():
    logger.info(f"🎵 Xandra Maestro v3.0 - {datetime.now()}")
    
    state = get_training_state()
    logger.info(f"📊 الخطوة الحالية: {state.get('current_step', 0)}")
    
    check_milestones(state)
    check_stale(state)
    
    # تقرير يومي (نُرسله إذا كانت آخر تقرير قبل 24 ساعة)
    try:
        with open("last_daily_report.txt") as f:
            last_report = datetime.fromisoformat(f.read().strip())
        if (datetime.now() - last_report).total_seconds() >= 86400:
            send_daily_report(state)
            with open("last_daily_report.txt", 'w') as f:
                f.write(datetime.now().isoformat())
    except:
        send_daily_report(state)
        with open("last_daily_report.txt", 'w') as f:
            f.write(datetime.now().isoformat())
    
    logger.info("✅ انتهت الدورة")

if __name__ == "__main__":
    main()
