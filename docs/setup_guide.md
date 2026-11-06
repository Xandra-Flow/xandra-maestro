# 📖 دليل الإعداد الكامل - مشروع Xandra Hope

## 🎯 ما الذي ستبنيه؟

نظام كامل لتدريب نموذج Xandra Hope عبر:
- **Hugging Face ZeroGPU**: GPU مجاني عبر API
- **GitHub Actions**: منسّق يشتغل كل ساعة
- **Telegram Bot**: بوت للأصدقاء للانضمام
- **Cloudflare**: استضافة الواجهة + R2 + D1

---

## 📋 المتطلبات

- حساب GitHub (لديك بالفعل)
- حساب Hugging Face (لديك بالفعل)
- حساب Telegram (لديك بالفعل)
- نطاق على Cloudflare (لديك بالفعل)

---

## 🚀 الخطوة 1: إنشاء HF Spaces (الأهم)

### 1.1 إنشاء Space للتدريب

```
1. اذهب إلى: https://huggingface.co/new-space
2. املأ البيانات:
   - Owner: your-username
   - Space name: xandra-trainer
   - License: Apache 2.0
   - SDK: Gradio
   - Hardware: ZeroGPU (مهم!)
   - Visibility: Private (أنت فقط)
3. اضغط "Create Space"
```

### 1.2 رفع الكود للـ Space

```bash
# استنساخ الـ Space
git clone https://huggingface.co/spaces/your-username/xandra-trainer
cd xandra-trainer

# نسخ الملفات من مجلد space/ في مشروعنا
cp /path/to/xandra-system/space/app.py .
cp /path/to/xandra-system/space/requirements.txt .
cp /path/to/xandra-system/space/README.md .

# رفع الكود
git add .
git commit -m "Initial Xandra Trainer"
git push
```

### 1.3 إنشاء مستودع لحفظ Checkpoints

```
1. اذهب إلى: https://huggingface.co/new-model
2. املأ:
   - Owner: your-username
   - Name: xandra-training-state
   - License: Apache 2.0
   - Visibility: Private
3. اضغط "Create Model"
```

---

## 🚀 الخطوة 2: إنشاء Telegram Bot

### 2.1 إنشاء البوت

```
1. افتح تيليجرام، ابحث عن: @BotFather
2. أرسل: /newbot
3. اختر اسماً: Xandra Maestro
4. اختر username: xandra_maestro_bot (أو أي اسم متاح)
5. احفظ الـ Token (يبدأ بـ أرقام:نص)
```

### 2.2 الحصول على Chat ID الخاص بك

```
1. ابحث عن: @userinfobot
2. أرسل: /start
3. احفظ الـ "Id:" (مثلاً: 123456789)
```

---

## 🚀 الخطوة 3: إنشاء GitHub Repo

### 3.1 إنشاء المستودع

```
1. اذهب إلى: https://github.com/new
2. املأ:
   - Repository name: xandra-maestro
   - Visibility: Private (مهم!)
   - Add README: ✅
3. اضغط "Create repository"
```

### 3.2 رفع الكود

```bash
# استنساخ
git clone https://github.com/your-username/xandra-maestro
cd xandra-maestro

# نسخ الملفات
cp -r /path/to/xandra-system/* .

# رفع
git add .
git commit -m "Initial Maestro setup"
git push
```

---

## 🚀 الخطوة 4: إعداد GitHub Secrets

اذهب إلى: `GitHub Repo → Settings → Secrets and variables → Actions`

أضف الـ Secrets التالية:

| Secret Name | القيمة | من أين؟ |
|---|---|---|
| `HF_REPO` | `your-username/xandra-training-state` | اسم مستودع HF |
| `HF_TOKEN_OWNER` | `hf_xxx...` | HF Settings → Tokens (Full access) |
| `ZERO_GPU_SPACE` | `your-username/xandra-trainer` | اسم الـ Space |
| `TELEGRAM_BOT_TOKEN` | `123456:ABC-xxx...` | من BotFather |
| `OWNER_CHAT_ID` | `123456789` | من @userinfobot |
| `GITHUB_TOKEN` | `ghp_xxx...` | GitHub Settings → Developer → Tokens |
| `GITHUB_REPO` | `your-username/xandra-maestro` | اسم الـ Repo |

---

## 🚀 الخطوة 5: استضافة البوت

البوت يحتاج خادم يعمل 24/7. الخيارات المجانية:

### الخيار A: على HF Spaces (موصى به)

```
1. اذهب إلى: https://huggingface.co/new-space
2. املأ:
   - Space name: xandra-bot
   - SDK: Docker
   - Hardware: CPU basic (Free)
   - Visibility: Private
3. اضغط "Create Space"
4. ارفع الكود من bot/telegram_bot.py
```

### الخيار B: على Cloudflare Workers

```javascript
// worker.js
export default {
  async fetch(request, env) {
    // محاكاة polling للبوت
    const url = `https://api.telegram.org/bot${env.BOT_TOKEN}/getUpdates`;
    const response = await fetch(url);
    const data = await response.json();
    
    // معالجة الرسائل
    for (const update of data.result || []) {
      await handleMessage(update.message, env);
    }
    
    return new Response("OK");
  }
};
```

### الخيار C: على جهازك (مؤقتاً)

```bash
# على جهازك
export TELEGRAM_BOT_TOKEN="..."
export GITHUB_TOKEN="..."
export GITHUB_REPO="your-username/xandra-maestro"
export OWNER_CHAT_ID="..."

python bot/telegram_bot.py
```

---

## 🚀 الخطوة 6: التحقق من العمل

### 6.1 اختبر البوت

```
1. افتح تيليجرام
2. ابحث عن البوت الخاص بك
3. أرسل: /start
4. يجب أن ترى رسالة الترحيب
5. أرسل: /join
6. اتبع التعليمات
```

### 6.2 اختبر الـ Space

```
1. اذهب إلى: https://huggingface.co/spaces/your-username/xandra-trainer
2. اضغط "Submit"
3. يجب أن ترى نتائج التدريب
```

### 6.3 اختبر GitHub Actions

```
1. اذهب إلى GitHub Repo → Actions
2. اضغط على "Xandra Maestro"
3. اضغط "Run workflow"
4. انتظر النتيجة
```

---

## 🎯 كيف يبدأ الأصدقاء؟

### أرسل لكل صديق:

```
مرحباً! أحتاج مساعدتك في مشروع تقني.
سيستغرق منك 5 دقائق فقط، ولن تحتاج لفعل أي شيء بعدها.

1. اذهب إلى: https://huggingface.co/join
2. سجّل بحساب Google
3. اذهب إلى: https://huggingface.co/settings/tokens
4. اضغط "Create new token"
5. الاسم: xandra | النوع: Read
6. اضغط Create، انسخ الـ Token
7. افتح تيليجرام، ابحث عن: @xandra_maestro_bot
8. أرسل /join، ثم الصق الـ Token

انتهى! شكراً لك 🙏
```

---

## 🎯 كيف تشاهد التقدم؟

### على تيليجرام (البوت سيخبرك):

```
🎉 صديق جديد انضم!
🆔 friend-123456789
👤 HF: ahmed_username

🚀 أطلقت مهمة على friend-123456789
📊 الخطوة الحالية: 0

✅ اكتملت جلسة friend-123456789
📈 الخطوة: 500/500000
📉 الخسارة: 4.234
⚡ خطوات الجلسة: 500
📊 نسبة الإنجاز: 0.10%
```

### على GitHub Actions:

```
GitHub Repo → Actions → Xandra Maestro
شاهد كل تشغيل ونتيجته
```

### على Hugging Face:

```
https://huggingface.co/your-username/xandra-training-state
- state.json: آخر حالة
- checkpoints/: ملفات الـ checkpoints
- training_logs/: سجلات التدريب
```

---

## ⚠️ مشاكل شائعة وحلولها

### 1. ZeroGPU لا يعمل

```
السبب: الحساب المجاني لا يملك صلاحية ZeroGPU
الحل:
  - ترقية الحساب إلى PRO ($9/شهر)
  - أو الانتظار (الحصص تتجدد يومياً)
  - أو الاعتماد على أصدقاء أكثر
```

### 2. البوت لا يستجيب

```
السبب: البوت لا يعمل 24/7
الحل:
  - انشر البوت على HF Space (CPU مجاني)
  - أو Cloudflare Workers
```

### 3. GitHub Actions يفشل

```
السبب: Secrets غير مكتملة
الحل: تأكد من جميع Secrets في الخطوة 4
```

### 4. الأصدقاء لا يستطيعون الانضمام

```
السبب: البوت خاص (Private)
الحل: اعرض رابط دعوة مباشر:
t.me/xandra_maestro_bot?start=join
```

---

## 📞 الخطوات التالية بعد الإعداد

1. **اختبر وحدك أولاً** (شغّل Maestro، شوف التدريب يشتغل)
2. **ادعو 1-2 صديق** للاختبار
3. **بعد النجاح**، ادعو المزيد
4. **راقب التقدم** على تيليجرام
5. **بعد 1000 خطوة**، قيّم النموذج
6. **بعد 50000 خطوة**، ابدأ في النشر

---

## 🎉 مبروك!

عند إتمام هذه الخطوات، لديك نظام كامل:
- ✅ بوت تيليجرام للأصدقاء
- ✅ Space ZeroGPU للتدريب
- ✅ Maestro على GitHub Actions
- ✅ تخزين على Hugging Face
- ✅ إشعارات تلقائية

التكلفة: $0
الزمن المطلوب لإكمال الإعداد: ~2-3 ساعات
