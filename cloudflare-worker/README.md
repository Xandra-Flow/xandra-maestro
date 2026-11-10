# 🚀 Xandra Hope - Cloudflare Worker

Cloudflare Worker واحد يقوم بكل شيء:
1. ✅ يعرض صفحة التسجيل (HTML)
2. ✅ يستقبل بيانات الصديق (POST /register)
3. ✅ يتحقق من HF Token عبر HF API
4. ✅ يحدّث `workers_config.yaml` على GitHub تلقائياً
5. ✅ يرسل إشعار للمالك على تيليجرام

## 📦 الملفات

```
cloudflare-worker/
├── worker.js          # الكود الكامل (HTML + API)
└── wrangler.toml      # إعدادات Cloudflare (اختياري)
```

## 🚀 النشر على Cloudflare Workers

### الطريقة 1: عبر Dashboard (الأسهل)

1. اذهب إلى: https://dash.cloudflare.com
2. Workers & Pages → Create application → Create Worker
3. اسم: `xandra-join`
4. اضغط **Deploy**
5. اضغط **Edit code**
6. احذف الكود الافتراضي
7. الصق محتوى `worker.js` كاملاً
8. اضغط **Save and Deploy**
9. ستحصل على رابط مثل: `https://xandra-join.smart-brute-force.workers.dev/`

### إضافة المتغيرات (Environment Variables):

1. في صفحة الـ Worker: Settings → Variables
2. أضف:

| Variable | القيمة |
|---|---|
| `GITHUB_TOKEN` | ghp_... (GitHub PAT بصلاحية repo) |
| `GITHUB_REPO` | `Xandra-Flow/xandra-maestro` |
| `TELEGRAM_BOT_TOKEN` | توكن البوت (اختياري) |
| `OWNER_CHAT_ID` | شات آي دي (اختياري) |

3. اضغط **Save and Deploy**

### الطريقة 2: عبر Wrangler CLI

```bash
# تثبيت Wrangler
npm install -g wrangler

# تسجيل دخول
wrangler login

# إنشاء Worker
wrangler deploy worker.js --name xandra-join --compatibility-date 2024-01-01
```

## 🎯 كيف يعمل؟

```
1. صديق يفتح: https://xandra-join.smart-brute-force.workers.dev/
2. يرى صفحة جميلة
3. يملأ النموذج:
   - اسم المستخدم في HF
   - HF Token
4. يضغط "تسجيل الآن"
5. الـ Worker:
   - يتحقق من الـ Token عبر HF API
   - يضيف الصديق إلى workers_config.yaml على GitHub
   - يرسل إشعار للمالك على تيليجرام (لو مفعل)
6. الصديق يرى رسالة: "✅ تم تسجيلك بنجاح!"
```

## ✅ المميزات

- مجاني 100% (100K طلب/يوم)
- لا PythonAnywhere
- لا Webhook
- لا GitHub Actions للبوت (اختياري)
- التسجيل مباشر من الصفحة
- سريع جداً (CDN عالمي)
- نطاقك الخاص (لو ربطته)

## 🔗 الروابط

- الصفحة: `https://xandra-join.smart-brute-force.workers.dev/`
- الحالة: `https://xandra-join.smart-brute-force.workers.dev/status`
