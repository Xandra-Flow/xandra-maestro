# 🎯 Xandra Hope System v3.0 - GitHub Only

نظام تدريب شبه آلي مجاني بالكامل عبر GitHub فقط (لا PythonAnywhere!)

## ✨ المميزات الجديدة

- ✅ مجاني 100% للأبد (لا تجديد شهري)
- ✅ صفحة HTML بسيطة للأصدقاء على GitHub Pages
- ✅ بوت تيليجرام يعمل على GitHub Actions (كل 10 دقائق)
- ✅ Maestro يعمل على GitHub Actions (كل ساعة)
- ✅ لا WSGI، لا Reload، لا تعقيد
- ✅ كل التحديثات تتم عبر GitHub فقط

## 📁 البنية

```
xandra-github/
├── docs/                              # GitHub Pages
│   ├── index.html                     # صفحة تسجيل الأصدقاء
│   └── style.css
├── bot/
│   ├── bot_runner.py                  # البوت (Long Polling)
│   └── requirements.txt
├── maestro/
│   ├── maestro.py                     # المنسّق
│   └── workers_config.yaml            # سجل Workers
├── .github/workflows/
│   ├── bot.yml                        # كل 10 دقائق
│   └── maestro.yml                    # كل ساعة
└── README.md
```

## 🚀 كيف يعمل؟

```
1. صديق يفتح صفحة GitHub Pages
2. يضغط "افتح البوت على تيليجرام"
3. يرسل /join للبوت
4. يرسل HF Token
5. البوت (في GitHub Actions):
   - يتحقق من الـ Token
   - يحدّث workers_config.yaml عبر GitHub API
   - يرسل إشعار للمالك
6. المايسترو (كل ساعة):
   - يقرأ state من HF
   - يرسل إشعارات التقدم
7. أنت تشوف التقدم على تيليجرام
```

## 📋 الإعداد (15 دقيقة)

### 1. رفع الكود إلى GitHub
1. أنشئ Repo خاص: `xandra-maestro`
2. ارفع كل ملفات `xandra-github/`

### 2. تفعيل GitHub Pages
1. اذهب إلى: Repo → Settings → Pages
2. Source: Deploy from a branch
3. Branch: `main` / `docs` folder
4. اضغط Save
5. انتظر دقائق، ستحصل على رابط: `https://YOUR_USERNAME.github.io/xandra-maestro/`

### 3. إعداد GitHub Secrets
اذهب إلى: Repo → Settings → Secrets → Actions

| Secret | القيمة |
|---|---|
| `TELEGRAM_BOT_TOKEN` | توكن البوت |
| `OWNER_CHAT_ID` | شات آي دي |
| `HF_REPO` | `smart-force/xandra-training-state` |
| `HF_TOKEN_OWNER` | توكن HF (Write) |
| `GITHUB_REPO` | `YOUR_USERNAME/xandra-maestro` |
| `GH_PAT` | GitHub Personal Access Token (للبوت لتعديل الملفات) |

### 4. إنشاء GitHub PAT (للبوت)
1. اذهب إلى: https://github.com/settings/tokens (أو fine-grained)
2. أنشئ token بصلاحية `repo` (كامل)
3. احفظه كـ `GH_PAT` في Secrets

### 5. اختبار النظام
1. افتح: `https://YOUR_USERNAME.github.io/xandra-maestro/`
2. اضغط "افتح البوت على تيليجرام"
3. أرسل `/start` للبوت
4. انتظر دقائق (البوت يعمل كل 10 دقائق)
5. يجب أن تتلقى رد!

## 🎯 لماذا بدون PythonAnywhere؟

| المعيار | PythonAnywhere | GitHub Only |
|---|---|---|
| التجديد الشهري | ❌ مطلوب | ✅ غير مطلوب |
| التعقيد | ⚠️ WSGI + Reload | ✅ بسيط |
| الـ Token في WSGI | ❌ مزعج | ✅ في Secrets |
| صفحة الأصدقاء | ❌ لا توجد | ✅ GitHub Pages |
| التحديث | يدوي | ✅ عبر Git |

## 📊 حدود GitHub Actions المجانية

- 2000 دقيقة/شهر للـ Actions
- البوت: ~3 دقائق × 6/ساعة × 24/يوم = ~4320 دقيقة/شهر ⚠️
- هذا يتجاوز الحد المجاني!

**الحل:** شغّل البوت كل 30 دقيقة بدلاً من 10 دقائق:
- تعديل `bot.yml`: `cron: "*/30 * * * *"`
- النتيجة: ~1440 دقيقة/شهر (ضمن الحد المجاني)

## 📜 الترخيص
Apache 2.0
