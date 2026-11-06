"""
Xandra Maestro - المنسّق المركزي المُحدّث
==========================================
يقرأ قائمة Workers من workers_config.yaml
يستدعي HF ZeroGPU Space عبر API لكل Worker
يجمع النتائج، يرفعها على HF، يرسل إشعارات Telegram

التنبيهات الإضافية:
- بدء مهمة تدريب
- نجاح مهمة تدريب
- فشل مهمة تدريب
- Token أصبح غير صالح
- استنفاد حصة Worker
- أخطاء GitHub Actions
"""

import os
import yaml
import json
import time
import logging
import traceback
import requests
from datetime import datetime
from gradio_client import Client
from huggingface_hub import HfApi, upload_file, hf_hub_download

# ====== الإعدادات ======
HF_REPO = os.environ.get("HF_REPO", "your-username/xandra-training-state")
OWNER_TOKEN = os.environ.get("HF_TOKEN_OWNER")
ZERO_GPU_SPACE = os.environ.get("ZERO_GPU_SPACE", "your-username/xandra-trainer")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
OWNER_CHAT_ID = os.environ.get("OWNER_CHAT_ID")
MAX_RETRIES = 3

# ====== نظام التسجيل ======
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xandra_maestro.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

hf_api = HfApi(token=OWNER_TOKEN)

# ====== نظام التنبيهات ======
def send_alert(message, level="info"):
    """
    إرسال تنبيه للمالك عبر Telegram
    
    Args:
        message: نص الرسالة
        level: info, success, warning, error
    """
    if not TELEGRAM_BOT_TOKEN or not OWNER_CHAT_ID:
        logger.info(f"📨 تنبيه ({level}): {message}")
        return
    
    emojis = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "🚨",
        "training": "🎓",
        "progress": "📊"
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
            logger.info(f"✅ تم إرسال تنبيه ({level})")
        else:
            logger.error(f"❌ فشل إرسال: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"❌ فشل إرسال التنبيه: {e}")

# ====== قراءة حالة المشروع ======
def get_training_state():
    """قراءة آخر حالة تدريب من HF"""
    try:
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename="state.json",
            repo_type="model",
            token=OWNER_TOKEN
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"⚠️ لا توجد حالة سابقة: {e}")
        return {
            "current_step": 0,
            "target_steps": 500000,
            "current_loss": None,
            "last_checkpoint": None,
            "history": [],
            "status": "ready",
            "started_at": None
        }

def save_training_state(state):
    """حفظ الحالة المُحدّثة على HF"""
    state["updated_at"] = datetime.now().isoformat()
    with open("state.json", "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    
    try:
        upload_file(
            path_or_fileobj="state.json",
            path_in_repo="state.json",
            repo_id=HF_REPO,
            repo_type="model",
            token=OWNER_TOKEN
        )
        logger.info("💾 تم حفظ الحالة على HF")
    except Exception as e:
        logger.error(f"❌ فشل حفظ الحالة: {e}")
        send_alert(
            f"❌ فشل حفظ الحالة على HF\n\n{str(e)}",
            level="error"
        )

# ====== قراءة Workers ======
def load_workers():
    """قراءة جميع Workers المُفعّلين"""
    try:
        with open("maestro/workers_config.yaml") as f:
            config = yaml.safe_load(f)
        workers = [w for w in config.get('workers', []) if w.get('enabled', True)]
        logger.info(f"👥 Workers متاحون: {len(workers)}")
        return workers
    except Exception as e:
        logger.error(f"❌ فشل قراءة workers_config: {e}")
        send_alert(
            f"❌ فشل قراءة workers_config.yaml\n\n{str(e)}",
            level="error"
        )
        return []

# ====== الحصول على Token Worker ======
def get_worker_token(worker):
    """استخراج Token العامل من Environment (في GitHub Actions)"""
    secret_name = worker.get('token_secret', '')
    return os.environ.get(secret_name, "")

# ====== التحقق من صحة Token ======
def check_token_valid(token):
    """التحقق من أن الـ Token ما زال صالحاً"""
    try:
        response = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15
        )
        return response.status_code == 200
    except:
        return False

# ====== استدعاء ZeroGPU Space ======
def call_zerogpu_space(worker_token, worker_id, current_step, max_steps=500):
    """
    استدعاء الـ Space عبر gradio_client باستخدام Token الصديق
    
    Returns:
        dict: نتائج التدريب أو خطأ
    """
    try:
        logger.info(f"📤 استدعاء Space لـ {worker_id} من الخطوة {current_step}")
        
        # التحقق من صحة Token أولاً
        if not check_token_valid(worker_token):
            return {
                "success": False,
                "error": "TOKEN_INVALID",
                "message": f"Token الصديق {worker_id} غير صالح"
            }
        
        # إنشاء Client باستخدام Token الصديق
        client = Client(
            ZERO_GPU_SPACE,
            hf_token=worker_token
        )
        
        # استدعاء دالة التدريب
        result = client.predict(
            current_step=current_step,
            max_steps=max_steps,
            api_name="/train_step"
        )
        
        logger.info(f"✅ نتيجة {worker_id}: {result}")
        return result
        
    except Exception as e:
        error_str = str(e).lower()
        
        # تحديد نوع الخطأ
        if "401" in error_str or "unauthorized" in error_str:
            return {
                "success": False,
                "error": "TOKEN_INVALID",
                "message": f"Token {worker_id} مرفوض من HF"
            }
        elif "quota" in error_str or "rate limit" in error_str:
            return {
                "success": False,
                "error": "QUOTA_EXCEEDED",
                "message": f"حصة {worker_id} استُنفدت"
            }
        elif "timeout" in error_str:
            return {
                "success": False,
                "error": "TIMEOUT",
                "message": f"انتهى وقت {worker_id}"
            }
        else:
            return {
                "success": False,
                "error": "UNKNOWN",
                "message": f"خطأ غير معروف لـ {worker_id}: {str(e)}"
            }

# ====== معالجة نتيجة التدريب ======
def process_training_result(result, worker, state):
    """معالجة نتيجة التدريب من الـ Space"""
    
    if not result.get("success"):
        error_type = result.get("error", "UNKNOWN")
        error_message = result.get("message", "غير معروف")
        
        # تنبيه حسب نوع الخطأ
        if error_type == "TOKEN_INVALID":
            # تنبيه حرج: Token الصديق أصبح غير صالح
            send_alert(
                f"🚨 *Token غير صالح*\n\n"
                f"👤 الصديق: `{worker['id']}`\n"
                f"👤 HF: `{worker.get('hf_username', 'N/A')}`\n"
                f"📱 Telegram: @{worker.get('telegram_username', 'N/A')}\n\n"
                f"السبب: {error_message}\n\n"
                f"📌 الإجراء: يجب تحديث Token أو إيقاف الـ Worker",
                level="error"
            )
            
        elif error_type == "QUOTA_EXCEEDED":
            # تنبيه عادي: استنفاد الحصة (متوقع)
            logger.info(f"⏳ {worker['id']} استنفد حصته (طبيعي)")
            # لا نرسل تنبيه لهذا (متكرر جداً)
            
        else:
            # تنبيه تحذيري: خطأ غير متوقع
            send_alert(
                f"⚠️ *فشل مهمة تدريب*\n\n"
                f"👤 الصديق: `{worker['id']}`\n"
                f"👤 HF: `{worker.get('hf_username', 'N/A')}\n\n"
                f"الخطأ: {error_message}\n"
                f"النوع: {error_type}",
                level="warning"
            )
        
        # تحديث حالة الـ Worker
        state["status"] = "ready"
        save_training_state(state)
        return state
    
    # ✅ نجح التدريب
    new_step = result.get("final_step", state["current_step"])
    steps_done = result.get("steps_completed", new_step - state["current_step"])
    final_loss = result.get("final_loss")
    
    state["current_step"] = new_step
    state["current_loss"] = final_loss
    state["status"] = "ready"
    state["last_checkpoint"] = datetime.now().isoformat()
    state["history"].append({
        "worker_id": worker['id'],
        "started_at": state.get("started_at"),
        "completed_at": datetime.now().isoformat(),
        "steps_done": steps_done,
        "final_loss": final_loss
    })
    
    # الحفاظ على آخر 50 سجل فقط
    if len(state["history"]) > 50:
        state["history"] = state["history"][-50:]
    
    save_training_state(state)
    
    # حساب نسبة الإنجاز
    progress = (new_step / state["target_steps"]) * 100
    
    # تنبيه نجاح المهمة
    send_alert(
        f"✅ *اكتملت جلسة تدريب*\n\n"
        f"👤 الصديق: `{worker['id']}`\n"
        f"👤 HF: `{worker.get('hf_username', 'N/A')}`\n"
        f"📈 الخطوة الحالية: `{new_step:,}` / `{state['target_steps']:,}`\n"
        f"📊 نسبة الإنجاز: `{progress:.2f}%`\n"
        f"📉 الخسارة: `{final_loss:.4f}`" if final_loss else "📉 الخسارة: N/A"
        + f"\n⚡ خطوات الجلسة: `{steps_done}`\n"
        f"📅 الوقت: {datetime.now().strftime('%H:%M:%S')}",
        level="success"
    )
    
    # تنبيه خاص عند الوصول لمراحل مهمة
    milestones = [1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000]
    if new_step in milestones:
        send_alert(
            f"🎉 *مرحلة مهمة!*\n\n"
            f"وصلنا للخطوة `{new_step:,}`\n"
            f"نسبة الإنجاز: `{progress:.2f}%`",
            level="success"
        )
    
    return state

# ====== الحلقة الرئيسية ======
def main():
    """الحلقة الرئيسية للمايسترو"""
    logger.info("=" * 50)
    logger.info(f"🎵 Xandra Maestro - {datetime.now()}")
    
    try:
        # 1. قراءة الحالة الحالية
        state = get_training_state()
        logger.info(f"📊 الخطوة الحالية: {state['current_step']}")
        
        # 2. لو المشروع انتهى
        if state["current_step"] >= state["target_steps"]:
            send_alert(
                f"🎉 *التدريب اكتمل!*\n\n"
                f"وصلنا للخطوة {state['current_step']}\n"
                f"النموذج جاهز!",
                level="success"
            )
            return
        
        # 3. لو هناك مهمة تعمل، لا تطلق جديدة
        if state.get("status") == "running":
            logger.info("⏳ هناك مهمة تعمل بالفعل")
            return
        
        # 4. تحميل Workers
        workers = load_workers()
        if not workers:
            logger.warning("⚠️ لا يوجد Workers!")
            return
        
        # 5. محاولة كل Worker بالتوالي
        for worker in workers:
            token = get_worker_token(worker)
            if not token:
                logger.warning(f"⚠️ لا يوجد Token لـ {worker['id']}")
                continue
            
            logger.info(f"🔄 محاولة {worker['id']}...")
            
            # إرسال تنبيه بدء المهمة
            send_alert(
                f"🚀 *بدء مهمة تدريب*\n\n"
                f"👤 الصديق: `{worker['id']}`\n"
                f"👤 HF: `{worker.get('hf_username', 'N/A')}`\n"
                f"📊 الخطوة الحالية: `{state['current_step']:,}`",
                level="training"
            )
            
            # تحديث الحالة
            state["status"] = "running"
            state["started_at"] = datetime.now().isoformat()
            save_training_state(state)
            
            # استدعاء الـ Space مع إعادة المحاولة
            result = None
            for attempt in range(MAX_RETRIES):
                if attempt > 0:
                    logger.info(f"🔄 إعادة محاولة {attempt + 1}/{MAX_RETRIES}")
                    time.sleep(5)
                
                result = call_zerogpu_space(
                    worker_token=token,
                    worker_id=worker['id'],
                    current_step=state["current_step"]
                )
                
                # لو نجحت، اخرج
                if result.get("success"):
                    break
                
                # لو فشل بـ Token غير صالح، لا تعيد المحاولة
                if result.get("error") == "TOKEN_INVALID":
                    break
            
            # معالجة النتيجة
            state = process_training_result(result, worker, state)
            
            # لو نجحت، اخرج من الحلقة
            if result and result.get("success"):
                break
        
        logger.info(f"✅ انتهت الدورة. الخطوة النهائية: {state['current_step']}")
        
    except Exception as e:
        error_msg = f"❌ خطأ حرج في Maestro:\n{str(e)}\n\n{traceback.format_exc()}"
        logger.error(error_msg)
        send_alert(
            f"🚨 *خطأ حرج في Maestro*\n\n"
            f"```\n{str(e)[:500]}\n```",
            level="error"
        )

if __name__ == "__main__":
    main()
