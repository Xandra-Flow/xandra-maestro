# 📖 دليل الإعداد الكامل - الإصدار 2.0

## 🎯 ماذا سنبني؟

نظام تدريب **شبه آلي** مجاني بالكامل:

```
أنت (5 دقائق يومياً) → Kaggle → HF → GitHub Actions → Telegram Bot → أنت
```

---

## 📋 المتطلبات

- ✅ حساب GitHub
- ✅ حساب Hugging Face
- ✅ حساب Kaggle
- ✅ حساب Telegram
- ✅ حساب Render.com (سنسجّل مجاناً)

---

## 🚀 الخطوات بالتفصيل

### الخطوة 1: إنشاء GitHub Repo (5 دقائق)

1. اذهب إلى: https://github.com/new
2. املأ:
   - **Repository name:** `xandra-maestro`
   - **Visibility:** ✅ **Private** (مهم!)
   - **Add README:** ✅
3. اضغط **"Create repository"**

### الخطوة 2: رفع ملفات المشروع (10 دقائق)

1. استخرج `xandra-system-v2/` على جهازك
2. ارفع كل الملفات إلى GitHub Repo:
   - عبر المتصفح: اسحب وأفلت
   - أو عبر Git:
     ```bash
     git clone https://github.com/YOUR_USERNAME/xandra-maestro.git
     cp -r xandra-system-v2/* xandra-maestro/
     cd xandra-maestro
     git add .
     git commit -m "Initial commit - Xandra v2.0"
     git push
     ```

### الخطوة 3: إنشاء HF Model Repo (5 دقائق)

1. اذهب إلى: https://huggingface.co/new-model
2. املأ:
   - **Owner:** حسابك
   - **Name:** `xandra-training-state`
   - **License:** `Apache 2.0`
   - **Visibility:** ✅ **Private**
3. اضغط **"Create model"**

### الخطوة 4: إنشاء HF Token (3 دقائق)

1. اذهب إلى: https://huggingface.co/settings/tokens
2. اضغط **"Create new token"**
3. املأ:
   - **Name:** `xandra_owner`
   - **Token type:** **"Write"** (مهم!)
4. اضغط Create
5. احفظ الـ Token (يبدأ بـ `hf_...`)

### الخطوة 5: إنشاء Telegram Bot (5 دقائق)

1. افتح تيليجرام
2. ابحث عن: **@BotFather**
3. أرسل: `/newbot`
4. اختر اسماً: `Xandra Maestro`
5. اختر username: `xandra_maestro_bot`
6. احفظ الـ **Bot Token** (مثال: `123456:ABC-DEF1234...`)

#### الحصول على Chat ID الخاص بك:

1. ابحث عن: **@userinfobot**
2. أرسل: `/start`
3. احفظ الـ **Id** (مثال: `123456789`)

### الخطوة 6: إنشاء GitHub Token (5 دقائق)

1. اذهب إلى: https://github.com/settings/tokens
2. اضغط **"Generate new token (classic)"**
3. املأ:
   - **Note:** `xandra_maestro`
   - **Expiration:** `90 days`
   - **Scopes:** ✅ `repo` (كامل)
4. اضغط **"Generate token"**
5. احفظ الـ Token (يبدأ بـ `ghp_...`)

### الخطوة 7: إعداد GitHub Secrets (10 دقائق)

اذهب إلى: `GitHub Repo → Settings → Secrets and variables → Actions → New repository secret`

أضف الـ 5 Secrets التالية بالضبط:

| Secret Name | القيمة |
|---|---|
| `HF_REPO` | `your-username/xandra-training-state` |
| `HF_TOKEN_OWNER` | `hf_xxxxxxxxxxxxxxxx` |
| `TELEGRAM_BOT_TOKEN` | `123456:ABC-DEF1234...` |
| `OWNER_CHAT_ID` | `123456789` |
| `GITHUB_TOKEN` | `ghp_xxxxxxxxxxxxxxxx` |

### الخطوة 8: إنشاء حساب Render.com (5 دقائق)

1. اذهب إلى: https://render.com/
2. اضغط **"Sign Up"**
3. سجّل بـ GitHub (الأسهل)
4. أكّد الإيميل

### الخطوة 9: نشر البوت على Render (15 دقيقة)

#### الطريقة A: عبر render.yaml (الأسهل)

1. اذهب إلى: https://dashboard.render.com
2. اضغط **"New +"** → **"Blueprint"**
3. اختر GitHub Repo: `xandra-maestro`
4. Render سيقرأ `render.yaml` تلقائياً
5. سيطلب منك إدخال قيم الـ envVars:
   - `TELEGRAM_BOT_TOKEN`: (القيمة من الخطوة 5)
   - `OWNER_CHAT_ID`: (القيمة من الخطوة 5)
   - `HF_REPO`: `your-username/xandra-training-state`
   - `HF_TOKEN_OWNER`: (القيمة من الخطوة 4)
6. اضغط **"Apply"**
7. انتظر 2-3 دقائق حتى يبني Render الـ service
8. ستحصل على URL مثل: `https://xandra-maestro-bot.onrender.com`

#### الطريقة B: عبر الـ Dashboard

1. اذهب إلى: https://dashboard.render.com
2. اضغط **"New +"** → **"Web Service"**
3. اختر GitHub Repo: `xandra-maestro`
4. املأ:
   - **Name:** `xandra-maestro-bot`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r bot/requirements.txt`
   - **Start Command:** `gunicorn bot.telegram_bot:app --worker-class gevent --workers 1 --timeout 60`
   - **Plan:** `Free`
5. في **Environment Variables**، أضف:
   - `TELEGRAM_BOT_TOKEN` = (القيمة)
   - `OWNER_CHAT_ID` = (القيمة)
   - `HF_REPO` = `your-username/xandra-training-state`
   - `HF_TOKEN_OWNER` = (القيمة)
6. اضغط **"Create Web Service"**
7. انتظر حتى يصبح Status: `Live`

### الخطوة 10: ضبط Telegram Webhook (2 دقيقة)

بعد أن يصبح البوت `Live` على Render:

1. افتح الرابط في المتصفح:
   ```
   https://xandra-maestro-bot.onrender.com/set_webhook
   ```
   (استبدل `xandra-maestro-bot` باسم service الخاص بك)

2. يجب أن ترى:
   ```
   ✅ تم ضبط Webhook: https://xandra-maestro-bot.onrender.com/webhook/xxx
   ```

### الخطوة 11: اختبار البوت (3 دقائق)

1. افتح تيليجرام
2. ابحث عن: `@xandra_maestro_bot`
3. أرسل: `/start`
4. يجب أن تتلقى رسالة الترحيب
5. أرسل: `/status`
6. يجب أن ترى حالة التدريب (سيكون 0 خطوة في البداية)

### الخطوة 12: رفع Notebook على Kaggle (10 دقائق)

1. اذهب إلى: https://www.kaggle.com/code
2. اضغط **"New Notebook"**
3. اضبط الإعدادات:
   - **Accelerator:** `GPU T4 x2`
   - **Internet:** `On`
4. اذهب إلى: **File → Import Notebook**
5. ارفع ملف: `kaggle/xandra_worker.ipynb`
6. احفظ الـ Notebook باسم: `xandra_worker`

### الخطوة 13: إعداد Kaggle Secrets (3 دقائق)

1. في Kaggle Notebook، اذهب إلى: **Add-ons → Secrets → Add Secret**
2. أضف Secretين:
   - **Name:** `HF_TOKEN` | **Value:** (نفس HF_TOKEN_OWNER)
   - **Name:** `HF_REPO` | **Value:** `your-username/xandra-training-state`

### الخطوة 14: أول تشغيل تجريبي (5 دقائق)

1. افتح `xandra_worker.ipynb` على Kaggle
2. اضغط **Run All** (Ctrl+F9)
3. انتظر دقائق حتى يبدأ التدريب
4. ستظهر رسائل مثل:
   ```
   ✅ تم بناء نموذج Xandra-Tiny
   📊 عدد الباراميترات: 3,xxx,xxx
   🚀 بدء التدريب من الخطوة 0
   ```
5. اغلق المتصفح (سيستمر Notebook بالعمل 12 ساعة)

### الخطوة 15: التحقق من النتائج (بعد ساعة)

1. اذهب إلى: https://huggingface.co/your-username/xandra-training-state
2. تحقق من وجود:
   - `state.json`
   - `checkpoints/latest.pt`
3. على تيليجرام، أرسل `/status` للبوت
4. يجب أن ترى الخطوة الحالية

---

## ✅ Checklist للتتبع

انقل هذا الجدول وتابع تقدمك:

```
□ 1. إنشاء GitHub Repo (xandra-maestro)
□ 2. رفع ملفات المشروع لـ GitHub
□ 3. إنشاء HF Model Repo (xandra-training-state)
□ 4. إنشاء HF Token (Write)
□ 5. إنشاء Telegram Bot (@BotFather)
□ 6. الحصول على Chat ID (@userinfobot)
□ 7. إنشاء GitHub Token
□ 8. إعداد 5 GitHub Secrets
□ 9. إنشاء حساب Render.com
□ 10. نشر البوت على Render (render.yaml)
□ 11. ضبط Telegram Webhook
□ 12. اختبار البوت (/start, /status)
□ 13. رفع Notebook على Kaggle
□ 14. إعداد Kaggle Secrets
□ 15. أول تشغيل تجريبي
□ 16. التحقق من النتائج على HF
```

---

## 🎯 ملاحظات مهمة

### ملاحظة 1: حدود Render.com Free
- ينام بعد 15 دقيقة خمول
- يستيقظ تلقائياً عند استقبال رسالة (Webhook)
- 750 ساعة شهرياً (يكفي لـ 1 service)

### ملاحظة 2: حدود Kaggle
- 30 ساعة GPU أسبوعياً
- 12 ساعة كحد أقصى للجلسة
- يتجدد يوم الاثنين

### ملاحظة 3: GitHub Actions
- 2000 دقيقة شهرياً مجاناً
- الـ Maestro يستخدم ~10 دقائق/ساعة = ~720 دقيقة/شهر
- كافٍ تماماً

### ملاحظة 4: Hugging Face
- Model Repo مجاني
- حجم غير محدود (عملياً)
- Read/Write مجاني للـ private repos

---

## ❓ المشاكل الشائعة وحلولها

### المشكلة: البوت لا يستجيب
**الحل:**
1. تأكد أن Render service `Live` وليس `Suspended`
2. اضغط على الرابط: `https://your-app.onrender.com/health`
3. لو لم يعمل، شغّل `/set_webhook` مرة أخرى

### المشكلة: Kaggle Notebook فشل
**الحل:**
1. تأكد أن GPU مفعّل
2. تأكد أن Internet مفعّل
3. تأكد أن Secrets صحيحة
4. راجع الأخطاء في الـ Output

### المشكلة: GitHub Actions لا يعمل
**الحل:**
1. اذهب إلى Actions → آخر تشغيل
2. شوف الـ logs
3. تأكد أن كل Secrets موجودة وصحيحة

### المشكلة: لا يصل تنبيه تيليجرام
**الحل:**
1. أرسل `/start` للبوت أولاً
2. تأكد أن OWNER_CHAT_ID صحيح
3. تأكد أن TELEGRAM_BOT_TOKEN صحيح

---

## 📞 الدعم

لو واجهت أي مشكلة:
1. أرسل لي رسالة
2. اذكر الخطوة التي وصلت لها
3. أرفق لقطة شاشة للخطأ (إن وجد)

سأساعدك فوراً.
