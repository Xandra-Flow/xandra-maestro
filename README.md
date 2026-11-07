# 🎯 Xandra Hope - المشروع الكامل

## 📁 البنية

```
xandra-final-package/
│
├── cloudflare-worker/
│   └── worker.js          # الكود الكامل (صفحة التسجيل + الدليل + الإحصائيات)
│
├── kaggle/
│   └── xandra_worker_kaggle.py   # كود التدريب على Kaggle
│
├── colab/
│   └── xandra_worker_colab.py   # كود التدريب على Colab (بديل)
│
├── maestro/
│   ├── maestro.py                 # المنسّق (GitHub Actions)
│   ├── process_registration.py    # معالج تسجيل الأصدقاء
│   └── workers_config.yaml        # سجل Workers
│
├── .github/workflows/
│   ├── maestro.yml                # كل ساعة: مراقبة + إشعارات
│   └── registrar.yml              # عند الـ Issues: تسجيل الأصدقاء
│
└── README.md (هذا الملف)
```

## 🚀 كيف تستخدم كل ملف؟

### 1. Cloudflare Worker (الاستضافة)
- الملف: `cloudflare-worker/worker.js`
- انسخه بالكامل والصقه في Cloudflare Worker جديد
- أضف المتغيرات: `GH_PAT` و `GITHUB_REPO`
- النتيجة: صفحة تسجيل + دليل + إحصائيات على رابط واحد

### 2. Kaggle (للتدريب - 30 ساعة أسبوعياً)
- الملف: `kaggle/xandra_worker_kaggle.py`
- اذهب إلى kaggle.com/code → New Notebook
- اضبط: Accelerator → GPU T4 | Internet → On
- أضف Secrets: HF_TOKEN, HF_REPO
- انسخ الكود في خلية واضغط Run All

### 3. Colab (للتدريب - بديل 12 ساعة يومياً)
- الملف: `colab/xandra_worker_colab.py`
- اذهب إلى colab.research.google.com
- Runtime → Change runtime type → T4 GPU
- عدّل HF_TOKEN في أعلى الكود
- انسخ الكود واضغط Run All

### 4. GitHub Actions (المايسترو)
- ارفع ملفات `.github/workflows/` و `maestro/` على GitHub
- أضف GitHub Secrets (GH_PAT, HF_TOKEN_OWNER, إلخ)
- النتيجة: نظام آلي بالكامل

## 📋 GitHub Secrets المطلوبة

| Secret | القيمة |
|---|---|
| `GH_PAT` | ghp_... (PAT بصلاحية repo) |
| `HF_TOKEN_OWNER` | hf_... (توكن HF الخاص بك) |
| `HF_REPO` | smart-force/xandra-training-state |
| `KAGGLE_USERNAME` | اسم مستخدم Kaggle |
| `KAGGLE_KEY` | مفتاح Kaggle API |
| `TELEGRAM_BOT_TOKEN` | توكن البوت |
| `OWNER_CHAT_ID` | 6194697593 |

## 🎯 الروابط النهائية

| الرابط | الوصف |
|---|---|
| `https://[worker].workers.dev/` | صفحة التسجيل |
| `https://[worker].workers.dev/guide` | الدليل الكامل |
| `https://[worker].workers.dev/stats` | الإحصائيات الحية |
