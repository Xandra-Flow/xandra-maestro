# 🎯 Xandra Hope System v2.0

نظام تدريب شبه آلي مجاني بالكامل لبناء نموذج Xandra Hope.

## 📁 البنية

```
xandra-system-v2/
│
├── bot/                          # بوت تيليجرام (Render.com)
│   ├── telegram_bot.py           # Flask + Webhook
│   ├── requirements.txt
│
├── kaggle/                       # Notebook التدريب
│   └── xandra_worker.ipynb       # جاهز للتشغيل
│
├── maestro/                      # المنسّق
│   └── maestro.py                # يراقب + يرسل إشعارات
│
├── .github/workflows/
│   └── maestro.yml               # جدولة كل ساعة
│
├── render.yaml                   # إعدادات Render.com
│
└── docs/
    ├── MASTER_PLAN_v2.md         # الخطة الكاملة
    ├── setup_guide.md             # دليل الإعداد
    ├── daily_workflow.md          # دليل العمل اليومي
    └── UPDATES_v2.0.md            # شرح التحديثات
```

## 🚀 البدء السريع

1. **اقرأ** `docs/MASTER_PLAN_v2.md` - الخطة الكاملة
2. **اتبع** `docs/setup_guide.md` - خطوة بخطوة للإعداد
3. **يومياً** `docs/daily_workflow.md` - الروتين اليومي

## 🎯 المميزات

- ✅ مجاني 100% (لا HF PRO، لا بطاقات بنكية)
- ✅ 5 دقائق يومياً فقط منك
- ✅ أتمتة شبه كاملة (Kaggle → HF → GitHub → Telegram)
- ✅ إشعارات تلقائية بكل تقدم
- ✅ احتفالات في المراحل المهمة
- ✅ تتبع Loss والتنبيه عند النضج

## 📊 كيف يعمل؟

```
أنت (5 دقائق) → تشغّل Notebook على Kaggle
                    ↓
        Notebook يدرّب 12 ساعة تلقائياً
                    ↓
        يحفظ checkpoints على HF (مجاني)
                    ↓
        Maestro على GitHub Actions (كل ساعة)
                    ↓
        يقرأ state ويرسل إشعارات Telegram
                    ↓
        أنت تشوف التقدم على البوت
```

## 🔧 المتطلبات

- GitHub account
- Hugging Face account  
- Kaggle account
- Telegram account
- Render.com account (مجاني)

## 📜 الترخيص

Apache 2.0
