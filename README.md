# 🎯 Xandra Hope System

نظام تدريب أوتوماتيكي كامل لبناء نموذج Xandra Hope السيادي.

## 📁 البنية

```
xandra-system/
│
├── bot/                          # بوت تيليجرام
│   └── telegram_bot.py           # استقبال تسجيل الأصدقاء
│
├── space/                        # HF Space (ZeroGPU)
│   ├── app.py                    # كود التدريب
│   ├── requirements.txt          # متطلبات Python
│   └── README.md                 # إعدادات الـ Space
│
├── maestro/                      # المنسّق المركزي
│   ├── maestro.py                # الكود الرئيسي
│   └── workers_config.yaml       # سجل الأصدقاء
│
├── .github/workflows/
│   └── maestro.yml               # جدولة GitHub Actions
│
├── docs/                         # التوثيق
│   ├── setup_guide.md            # دليل الإعداد الكامل
│   └── friend_guide.md           # دليل الصديق (5 دقائق فقط)
│
└── README.md                     # هذا الملف
```

## 🚀 البدء السريع

1. **اقرأ**: `docs/setup_guide.md` - دليل الإعداد الكامل
2. **أنشئ الـ Spaces على HF** (خطوة 1)
3. **أنشئ البوت** (خطوة 2)
4. **أنشئ GitHub Repo + Secrets** (خطوة 3-4)
5. **ارفع الكود** (الملفات الموجودة هنا)
6. **اختبر** (خطوة 6)
7. **ادعو الأصدقاء** باستخدام `docs/friend_guide.md`

## 🎯 المميزات

- ✅ مجاني 100%
- ✅ صديقك يسجّل مرة واحدة (5 دقائق) ثم لا شيء
- ✅ أتمتة كاملة عبر GitHub Actions
- ✅ إشعارات تيليجرام فورية
- ✅ تخزين مركزي على Hugging Face
- ✅ قابل للتوسع (إضافة أصدقاء بسهولة)
- ✅ آمن (Tokens بصلاحية Read فقط)

## 📊 كيف يعمل النظام؟

```
صديق يسجّل في HF → يعطيك Token → ينتهي دوره
              ↓
        البوت يحفظ Token
              ↓
   GitHub Actions يعمل كل ساعة
              ↓
   Maestro يقرأ Tokens الأصدقاء
              ↓
   يستدعي ZeroGPU Space لكل صديق
              ↓
   Space يدرّب النموذج 500 خطوة
              ↓
   يحفظ checkpoint على HF
              ↓
   Maestro يرسل إشعار لك على تيليجرام
```

## 🔐 الأمان

- Tokens بصلاحية "Read" فقط
- مستودعاتك Private على HF
- GitHub Repo Private
- يمكن سحب أي صديق فوراً (/leave)

## 📚 المراجع

- [Hugging Face ZeroGPU Docs](https://huggingface.co/docs/hub/spaces-zero-gpu)
- [GitHub Actions Docs](https://docs.github.com/actions)
- [Telegram Bot API](https://core.telegram.org/bots/api)

## 📜 الترخيص

Apache 2.0 - يمكنك استخدام هذا الكود لأي غرض، بما في ذلك التجاري.
