"""
Xandra Maestro - الإصدار 2.0 (مبسّط)
====================================
يعمل على GitHub Actions كل ساعة
يقرأ state من HF، يرسل إشعارات Telegram

الاختلافات عن v1.0:
- لا يطلق تدريباً تلقائياً (لا يمكنه ذلك على Kaggle)
- يراقب التقدم فقط
- يكتشف النضج (Loss يستقر)
- يرسل تنبيهات احتفالية
"""

import os
import json
import logging
import requests
from datetime import datetime
from huggingface_hub import hf_hub_download, HfApi

# ====== الإعدادات ======
HF_REPO = os.environ.get("HF_REPO", "your-username/xandra-training-state")
HF_TOKEN = os.environ.get("HF_TOKEN_OWNER")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
OWNER_CHAT_ID = os.environ.get("OWNER_CHAT_ID")

# ====== التسجيل ======
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====== قائمة المراحل الاحتفالية ======
STEP_MILESTONES = [100, 500, 1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000]
LOSS_MILESTONES = [10, 5, 4, 3, 2.5, 2, 1.5]

# ملف لتتبع المراحل التي تم الاحتفال بها
CELEBRATED_FILE = "celebrated_milestones.json"

def load_cebrated():
    """تحميل قائمة المراحل المحتفل بها"""
    try:
        with open(CELEBRATED_FILE) as f:
            return set(json.load(f))
    except:
        return set()

def save_celebrated(celebrated_set):
    """حفظ قائمة المراحل المحتفل بها"""
    with open(CELEBRATED_FILE, 'w') as f:
        json.dump(list(celebrated_set), f)

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
            "best_loss": float('inf'),
            "last_updated": None
        }

# ====== إرسال إشعار Telegram ======
def send_alert(message, level="info"):
    """إرسال تنبيه للمالك"""
    if not TELEGRAM_BOT_TOKEN or not OWNER_CHAT_ID:
        logger.info(f"📨 ({level}): {message}")
        return
    
    emojis = {
        "info": "ℹ️", "success": "✅", "warning": "⚠️", 
        "error": "🚨", "celebrate": "🎉", "progress": "📊"
    }
    emoji = emojis.get(level, "ℹ️")
    
    text = f"🎵 Xandra Maestro:\n{emoji} {message}"
    
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": OWNER_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown"
            },
            timeout=15
        )
        if response.status_code == 200:
            logger.info(f"✅ تم إرسال إشعار ({level})")
        else:
            logger.error(f"❌ فشل إرسال: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ فشل إرسال الإشعار: {e}")

# ====== فحص مراحل الاحتفال ======
def check_milestones(state, celebrated):
    """فحص هل وصلنا لمرحلة جديدة يجب الاحتفال بها"""
    step = state.get('current_step', 0)
    loss = state.get('current_loss')
    
    # فحص مراحل الخطوات
    for milestone in STEP_MILESTONES:
        key = f"step_{milestone}"
        if step >= milestone and key not in celebrated:
            celebrate_step(milestone, step, state)
            celebrated.add(key)
    
    # فحص مراحل الخسارة
    if loss is not None:
        for milestone in LOSS_MILESTONES:
            key = f"loss_{milestone}"
            if loss < milestone and key not in celebrated:
                celebrate_loss(milestone, loss, step)
                celebrated.add(key)
    
    # فحص النضج (Loss يستقر)
    check_convergence(state, celebrated)
    
    # فحص اكتمال التدريب
    if step >= state.get('target_steps', 500000):
        key = "training_complete"
        if key not in celebrated:
            celebrate_completion(state)
            celebrated.add(key)

def celebrate_step(milestone, current_step, state):
    """الاحتفال بمرحلة خطوات"""
    progress = (current_step / state['target_steps']) * 100
    send_alert(
        f"*🎉 مرحلة مهمة!*\n\n"
        f"وصلنا للخطوة `{milestone:,}`\n"
        f"📊 نسبة الإنجاز: `{progress:.2f}%`\n"
        f"🎯 الهدف: `{state['target_steps']:,}`\n"
        f"📉 الخسارة الحالية: `{state.get('current_loss', 'N/A')}`",
        level="celebrate"
    )

def celebrate_loss(milestone, current_loss, step):
    """الاحتفال بمرحلة خسارة"""
    messages = {
        10: "النموذج تخرج من مرحلة الهراء!",
        5: "النموذج بدأ ينتج نصاً مفهوماً!",
        4: "النموذج بدأ يفهم اللغة العربية!",
        3: "النموذج أصبح جيداً!",
        2.5: "النموذج أصبح قوياً جداً!",
        2: "النموذج وصل لنضج متقدم!",
        1.5: "النموذج في مستوى ممتاز!"
    }
    msg = messages.get(milestone, "إنجاز!")
    
    send_alert(
        f"*🎉 معلم خسارة مهم!*\n\n"
        f"الخسارة وصلت لـ أقل من `{milestone}`\n"
        f"📉 الخسارة الحالية: `{current_loss:.4f}`\n"
        f"📈 الخطوة: `{step:,}`\n\n"
        f"📌 {msg}",
        level="celebrate"
    )

def check_convergence(state, celebrated):
    """فحص هل النموذج نضج (Loss يستقر)"""
    history = state.get('history', [])
    if len(history) < 5:
        return
    
    # آخر 5 سجلات
    recent = history[-5:]
    losses = [h.get('final_loss', 0) for h in recent]
    
    if all(losses):
        avg_recent = sum(losses) / len(losses)
        avg_old = sum(losses[:2]) / 2
        improvement = (avg_old - avg_recent) / avg_old * 100
        
        # لو التحسن أقل من 1% خلال 5 جلسات
        if abs(improvement) < 1 and "convergence_warning" not in celebrated:
            send_alert(
                f"*🎯 قد يكون النموذج نضج!*\n\n"
                f"آخر 5 جلسات لم تُظهر تحسناً ملحوظاً:\n"
                f"  متوسط أول جلستين: `{losses[0]:.4f}`\n"
                f"  متوسط آخر جلسة: `{losses[-1]:.4f}`\n"
                f"  نسبة التحسن: `{improvement:.2f}%`\n\n"
                f"📌 توصية: اختبر النموذج يدوياً.\n"
                f"  أرسل `/test_model` للبوت لتجربة النموذج.",
                level="warning"
            )
            # نضيف علامة مؤقتاً (لمدة 24 ساعة مثلاً)
            celebrated.add("convergence_warning")

def celebrate_completion(state):
    """الاحتفال بإكمال التدريب"""
    send_alert(
        f"*🎊🎊🎊 اكتمل التدريب! 🎊🎊🎊*\n\n"
        f"الخطوة النهائية: `{state['current_step']:,}`\n"
        f"الخسارة النهائية: `{state.get('current_loss', 'N/A')}`\n"
        f"أفضل خسارة: `{state.get('best_loss', 'N/A')}`\n\n"
        f"🎯 النموذج جاهز للنشر!\n"
        f"📌 الخطوات التالية:\n"
        f"  1. `/export` لتصدير GGUF\n"
        f"  2. `/deploy` للنشر على Render\n"
        f"  3. `/test` لاختبار شامل",
        level="celebrate"
    )

# ====== تقرير يومي ======
def send_daily_report(state):
    """إرسال تقرير يومي عن التقدم"""
    history = state.get('history', [])
    if not history:
        return
    
    # آخر 24 ساعة
    last_24h = [h for h in history if h.get('timestamp') and 
                (datetime.now() - datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00'))).total_seconds() < 86400]
    
    if not last_24h:
        return
    
    steps_today = sum(h.get('steps_done', 0) for h in last_24h)
    sessions_today = len(last_24h)
    
    send_alert(
        f"*📊 تقرير اليوم*\n\n"
        f"📈 جلسات اليوم: `{sessions_today}`\n"
        f"🎯 خطوات اليوم: `{steps_today:,}`\n"
        f"📊 إجمالي الخطوات: `{state['current_step']:,}`\n"
        f"🎯 الهدف: `{state['target_steps']:,}`\n"
        f"📊 نسبة الإنجاز: `{(state['current_step']/state['target_steps']*100):.2f}%`\n"
        f"📉 الخسارة الحالية: `{state.get('current_loss', 'N/A')}`\n\n"
        f"📌 تابع تشغيل Notebook يومياً!",
        level="progress"
    )

# ====== الحلقة الرئيسية ======
def main():
    logger.info(f"🎵 Xandra Maestro v2.0 - {datetime.now()}")
    
    # قراءة الحالة
    state = get_training_state()
    logger.info(f"📊 الخطوة الحالية: {state.get('current_step', 0)}")
    
    # فحص المراحل
    celebrated = load_celebrated()
    check_milestones(state, celebrated)
    save_celebrated(celebrated)
    
    # تقرير يومي (مرة كل 24 ساعة)
    if state.get('last_updated'):
        try:
            last_update = datetime.fromisoformat(state['last_updated'].replace('Z', '+00:00'))
            hours_since = (datetime.now() - last_update).total_seconds() / 3600
            if hours_since >= 24:
                send_daily_report(state)
                state['last_daily_report'] = datetime.now().isoformat()
        except:
            pass
    
    # فحص هل آخر تحديث قديم جداً (تنبيه)
    history = state.get('history', [])
    if history:
        last_session = history[-1]
        try:
            last_time = datetime.fromisoformat(last_session['timestamp'].replace('Z', '+00:00'))
            hours_since = (datetime.now() - last_time).total_seconds() / 3600
            if hours_since > 48:
                send_alert(
                    f"*⚠️ توقف التدريب!*\n\n"
                    f"آخر جلسة كانت منذ `{hours_since:.0f}` ساعة\n"
                    f"📌 يجب تشغيل Notebook على Kaggle!",
                    level="warning"
                )
        except:
            pass
    
    logger.info("✅ انتهت الدورة")

if __name__ == "__main__":
    main()
