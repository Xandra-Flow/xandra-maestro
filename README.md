# 🎯 Xandra Hope — نظام التدريب الموزّع

> **نظام تدريب آلي متعدد المنصات** لـ HEPL (Hybrid Expert-Parallel Learning) على معمارية Qwen2.5-Coder-1.5B-Instruct، يجمع بين Kaggle (GPU T4)، Hugging Face Hub (الحالة والمستودعات)، Cloudflare Workers (لوحة التحكم)، و GitHub Actions (المنسّق).
>
> **الإصدار:** v9.2 (Stage 2 SFT + HEPL) — إعادة التدريب من 150K إلى 180K
> **الترخيص:** XHNSL v1.0 (حقوق الطبع محفوظة)

---

## 📑 الفهرس

1. [نظرة معمارية](#نظرة-معمارية)
2. [بنية المستودع](#بنية-المستودع)
3. [مراحل التدريب السبع](#مراحل-التدريب-السبع)
4. [المتطلبات والإعداد](#المتطلبات-والإعداد)
5. [GitHub Secrets المطلوبة](#github-secrets-المطلوبة)
6. [Cloudflare Worker Secrets](#cloudflare-worker-secrets)
7. [التشغيل السريع](#التشغيل-السريع)
8. [مسارات API (Worker)](#مسارات-api-worker)
9. [مراقبة الصحة والتشخيص](#مراقبة-الصحة-والتشخيص)
10. [إعدادات الذاكرة الآمنة](#إعدادات-الذاكرة-الآمنة)
11. [استكشاف الأخطاء](#استكشاف-الأخطاء)
12. [الترخيص والإسناد](#الترخيص-وإسناد)

---

## نظرة معمارية

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Xandra Hope Training System                      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│  GitHub       │         │  Cloudflare   │         │  Hugging      │
│  Actions      │         │  Worker       │         │  Face Hub     │
│  (maestro.py) │         │  (worker.js)  │         │  (state.json) │
│  كل 30 دقيقة  │         │  لوحة تحكم    │         │  مستودع الحالة │
└───────┬───────┘         └───────┬───────┘         └───────┬───────┘
        │                         │                         │
        │ يُطلق                    │ يقرأ/يُحدّث               │ يُزامن
        ▼                         ▼                         ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│  Kaggle       │         │  GitHub API   │         │  HF Repos     │
│  (6 حسابات)   │         │  (workflows)  │         │  • training-  │
│  GPU T4 x2    │         │               │         │    state      │
│  xandra_      │         └───────────────┘         │  • arabic-     │
│  worker.py    │                                    │    corpus     │
└───────┬───────┘                                    │  • tech-corpus │
        │                                            │  • sft-corpus │
        └────────────────────────────────────────────►│  • core-      │
                                                     │    private    │
                                                     └───────────────┘
```

### المسؤوليات:

| المكوّن | الدور | التكرار |
|---|---|---|
| **maestro.py** (GitHub Actions) | منسّق: يفحص حالة الحسابات، يُطلق جلسات Kaggle، يُرسل تقارير بالبريد | كل 30 دقيقة |
| **xandra_worker.py** (Kaggle) | عامل التدريب: يحمل state.json، يدرّب HEPL، يرفع checkpoints | جلسة واحدة = ~9 ساعات |
| **worker.js** (Cloudflare) | لوحة تحكم ويب + API: عرض الحالة، تعديل الإعدادات، إطلاق manual | عند الطلب |
| **state.json** (HF Hub) | مصدر الحقيقة: current_step، loss، training_lock، telemetry | تحديث مستمر |

---

## بنية المستودع

```
xandra-maestro/
│
├── xandra_worker.py                  # عامل التدريب (يُرفع إلى Kaggle)
│
├── kaggle-worker/
│   ├── hepl_architecture.py           # معمارية HEPL (نموذج + خسائر)
│   ├── ldr_referee.py                 # حكم الطبقة 3 (Lateral Dialectic)
│   └── memory_b_cdd.py                # Memory-B و CDD
│
├── maestro/
│   └── maestro.py                     # المنسّق (يعمل على GitHub Actions)
│
├── cloudflare-worker/
│   ├── worker.js                      # لوحة تحكم + API (v4.5 مع STAGE_PRESETS)
│   └── README.md                      # وثائق Worker
│
├── .github/workflows/
│   ├── maestro.yml                    # كل 30 دقيقة: مراقبة + إطلاق + تحقق
│   ├── distill_corpus.yml             # عند الطلب: تقطير معرفي من Qwen2.5-7B
│   └── corpus_report.yml              # أسبوعياً: تقرير حالة Corpus
│
├── scripts/
│   ├── analyze_corpus.py              # تحليل إحصائيات Corpus
│   ├── distill_sft.py                 # سكربت التقطير (يعمل على GitHub Actions)
│   └── eval_hepl.py                  # تقييم نموذج HEPL
│
├── training_sources.json              # مصادر التدريب (4 مصادر بأوزان)
├── TEACHERS_AND_GOALS.md              # المعلمون والأهداف التعليمية
├── TRAINING_GUIDE.md                  # دليل التدريب الكامل
└── README.md                          # هذا الملف
```

---

## مراحل التدريب السبع

| المرحلة | الملف | الـ LR | الـ Target | Batch / Grad | المدة |
|---|---|---|---|---|---|
| **1. Pretrain** | `xandra_worker.py` | 3e-4 | 150K | 8 / 16 | ~570 د |
| **2. SFT + HEPL** ← *الحالية* | `xandra_worker.py` | 1.5e-5 | **180K** | **1 / 16** | ~570 د |
| **3. Memory-B & CDD** | `memory_b_cdd.py` | 1e-5 | 163K | 1 / 16 | ~570 د |
| **4. LDR Alignment** | `ldr_referee.py` | 5e-6 | 165K | 1 / 16 | ~570 د |
| **5. Scaling 1.5B** | `xandra_worker.py` | 1.5e-4 | 100K | 1 / 16 | ~570 د |
| **6. QLoRA 7B** | `xandra_worker.py` | 2e-4 | 20K | 1 / 16 | ~570 د |
| **7. QLoRA 32B** | `xandra_worker.py` | 2e-4 | 25K | 1 / 16 | ~570 د |

> **ملاحظة:** المرحلة 2 تعمل حالياً (إعادة تدريب v9.2 من 150K إلى 180K). الإعدادات `batch=1, grad=16` تضمن استهلاك VRAM آمن ≤ 2 GB / 16 GB على T4.

---

## المتطلبات والإعداد

### الحسابات والمنصات المطلوبة:

| المنصة | الحساب | الاستخدام |
|---|---|---|
| **GitHub** | حساب واحد + repo `xandra-maestro` | تشغيل maestro.yml + workflows |
| **Hugging Face** | حساب `smart-force` (أو حسابك) | مستودعات state.json و corpora |
| **Kaggle** | 6 حسابات (KGAT_ tokens) | جلسات GPU T4 × 2 (30 ساعة/أسبوع لكل حساب) |
| **Cloudflare** | حساب مجاني | Worker للوحة التحكم |
| **Gmail** | حساب مع App Password | إرسال تقارير maestro |

---

## GitHub Secrets المطلوبة

أضف هذه Secrets في `Settings → Secrets and variables → Actions` في الـ repo:

### Hugging Face:

| Secret | القيمة | الوصف |
|---|---|---|
| `HF_REPO` | `smart-force/xandra-training-state` | مستودع state.json |
| `HF_TOKEN_OWNER` | `hf_xxx...` | HF access token (صلاحية write) |

### Kaggle (الحسابات 1-6):

> **ملاحظة:** KGAT_ tokens تُولّد من `kaggle.com/settings/api` (التنسيق الجديد).

| Secret | القيمة |
|---|---|
| `KAGGLE_API_TOKEN` | `KGAT_xxx...` |
| `KAGGLE_API_TOKEN_2` ... `KAGGLE_API_TOKEN_6` | `KGAT_xxx...` |
| `KAGGLE_USERNAME` ... `KAGGLE_USERNAME_6` | اسم المستخدم (للعرض فقط) |
| `KAGGLE_KEY` ... `KAGGLE_KEY_6` | نفس `KGAT_` token (fallback) |

### البريد:

| Secret | القيمة |
|---|---|
| `GMAIL_USER` | `your-email@gmail.com` |
| `GMAIL_PASS` | App Password من Google (16 حرف) |
| `EMAIL_TO` | بريد استقبال التقارير |

---

## Cloudflare Worker Secrets

في Cloudflare Workers → Settings → Variables:

| Variable | القيمة | الوصف |
|---|---|---|
| `HF_TOKEN_OWNER` | `hf_xxx...` | HF token لقراءة/كتابة state.json |
| `GITHUB_REPO` | `owner/xandra-maestro` | repo للوصول إلى Actions API |
| `GH_PAT` | `ghp_xxx...` | GitHub PAT بصلاحية `repo` + `actions` |

> **ملاحظة:** ضع هذه كـ **Secrets** وليس Variables (مشفّرة).

---

## التشغيل السريع

### 1. إعداد المنصات (مرة واحدة):

```bash
# أ. أنشئ مستودع HF (smart-force/xandra-training-state)
#    وارفع state.json أولي (البنية في xandra_worker.py:def get_state())

# ب. أنشئ 6 Kaggle API tokens (KGAT_ format) من kaggle.com/settings/api

# ج. أنشئ GitHub PAT بصلاحية repo + actions

# د. أنشئ Gmail App Password من Google Account → Security → App passwords
```

### 2. رفع الكود إلى GitHub:

```bash
git clone <your-repo>.git xandra-maestro
cd xandra-maestro
# انسخ محتوى هذا المستودع
git add .
git commit -m "feat: v9.2 Stage 2 retrain — batch=1/grad=16, OCR filter, distill 50%"
git push origin main
```

### 3. إضافة GitHub Secrets:

أضف كل Secrets المذكورة في القسم أعلاه.

### 4. نشر Cloudflare Worker:

```bash
# من Cloudflare Dashboard → Workers & Pages → Create
# الصق worker.js → Deploy
# أضف Variables/Secrets الثلاثة
```

### 5. إطلاق أول جلسة:

```bash
# من GitHub Actions → "Xandra Maestro" → Run workflow
# أو انتظر أول تشغيل تلقائي (خلال 30 دقيقة)
```

### 6. المراقبة:

افتح رابط Worker: `https://<your-worker>.workers.dev/` → سترى:
- حالة كل حساب Kaggle (نشط/مستخدم/معطّل)
- التقدم الحالي (current_step / target_steps)
- منحنى الخسارة (loss_history)
- ساعات GPU المستخدمة
- زر "Force Update" للتحديث الفوري

---

## مسارات API (Worker)

| الطريقة | المسار | الوصف |
|---|---|---|
| `GET` | `/` | لوحة التحكم الرئيسية (Dashboard) |
| `GET` | `/stats` | صفحة الإحصائيات |
| `GET` | `/settings` | تعديل إعدادات التدريب |
| `GET` | `/corpus` | عرض حالة Corpus |
| `GET` | `/api/guide` | دليل التشغيل (JSON) |
| `POST` | `/api/admin` | عمليات إدارية (pause/resume/lock/unlock) |
| `POST` | `/api/force-update` | إطلاق maestro.yml يدوياً |
| `POST` | `/api/scan-corpus` | فحص حالة Corpus |

---

## مراقبة الصحة والتشخيص

### مؤشرات النجاح (راقبها في اللوجز):

- ✅ `Batch size: 2 (effective: 32)` + `Gradient accumulation: 16 steps`
- ✅ `VRAM Check Step 2] Allocated: ~2 GB / 16 GB`
- ✅ `🚫 تخطّي N سجل OCR ملتصق` (دليل أن فلتر OCR يعمل)
- ✅ `L_KL < 5` و `L_gate ~ 0.5-1.0` في نهاية الجلسة
- ✅ `M_real_norm ratio > 1.0` (KillSwitch لم يطلق)

### مؤشرات الفشل (توقف فوراً):

- ❌ `OutOfMemoryError` → تحقق من batch/grad (يجب أن يكونا 1/16)
- ❌ `ScatterGather index out of bounds` → تحقق من vocab_size (151646)
- ❌ `M_real_norm ratio < 0.5` لـ > 200 خطوة → Boabab توقف
- ❌ `KillSwitch triggered` → راجع telemetry في state.json

### استهلاك GitHub Actions:

- **v4.4 (الحالي):** 1,720 دقيقة/شهر (هامش آمن تحت 2,000 المجانية)
- راقب: GitHub Settings → Billing → Actions
- لو اقتربت من 1,800 → فعّل خطة الطوارئ في `MINUTES_SOLUTION.md`

---

## إعدادات الذاكرة الآمنة

### لماذا `batch=1, grad=16` (وليس `batch=2, grad=8`)؟

`worker.py` السطر 786 يضرب `batch_size × gpu_count` (DataParallel):

| الإعداد | GPU يرى seqs/step | effective batch | VRAM peak | النتيجة |
|---|---|---|---|---|
| ❌ `batch=2, grad=8` | **2 sequences** | 32 | ~3 GB | محفوف بالمخاطر |
| ✅ `batch=1, grad=16` | **1 sequence** | 32 | **1.97 GB / 16** | هامش 88% آمن |

كلاهما يعطي `effective batch = 32`، لكن الأول أخف على VRAM بـ 50%.

### تحديث STAGE_PRESETS في worker.js:

```javascript
// stage 2/3/4 جميعها:
{ batch: 1, grad: 16, maxLen: 1024, precision: "float16", parallel: "auto" }
```

---

## استكشاف الأخطاء

### 1. الـ Job "متجمّد" في GitHub Actions (12 دقيقة صمت)

**السبب:** `PYTHONUNBUFFERED` لم يُضبط.
**الحل:** تأكد أن maestro.yml يحتوي:
```yaml
PYTHONUNBUFFERED: '1'
PYTHONIOENCODING: utf-8
```

### 2. الـ worker يسقط بـ OOM

**السبب:** `batch > 1` أو `maxLen > 1024`.
**الحل:** من لوحة Worker → Settings → تأكد `batch_size = 1`, `grad_accum_steps = 16`.

### 3. ScatterGather index out of bounds

**السبب:** `vocab_size` غير متطابق بين student و teacher.
**الحل:** تأكد `vocab_size = 151646` في `hepl_architecture.py` (السطر 383).

### 4. لا تصل بريد التقارير

**السبب:** Gmail App Password غير صحيح.
**الحل:** أعد توليد App Password من Google Account → Security → App passwords.

### 5. Kaggle API: "401 Unauthorized"

**السبب:** KGAT_ token منتهي الصلاحية.
**الحل:** ولّد token جديد من `kaggle.com/settings/api` وحدّث `KAGGLE_API_TOKEN_N`.

### 6. نص عربي ملتصق بعد التدريب

**السبب:** نصوص OCR ملتصقة في data/*.parquet دخلت التدريب.
**الحل:** تأكد أن `xandra_worker.py` (v9.2) يحتوي فلتر `_is_bad_stream_row` (السطر 905).

---

## الترخيص وإسناد

```
Xandra Hope Neural Software License (XHNSL) v1.0
Copyright (c) 2026 Ahmed Sayed Kamel Mahmoud &
               Александра Евгеньевна Острикова
All Rights Reserved.
```

- **الكود**: حقوق الطبع محفوظة للمؤلفين. الاستخدام الشخصي والبحثي مسموح. إعادة التوزيع التجارية تتطلب إذناً صريحاً.
- **النماذج المُدرّبة**: ملكية المؤلفين، تُنشر عبر Hugging Face Hub وفق شروطهم.
- **البيانات**: المصادر الخارجية (Wikipedia, Shamela) تخضع لتراخيصها الأصلية.

### المراجع التقنية:

- **HEPL Architecture**: `kaggle-worker/hepl_architecture.py` (القسم 6 من ورقة البحث)
- **LDR Referee**: `kaggle-worker/ldr_referee.py` (المعادلات 12-20)
- **Memory-B CDD**: `kaggle-worker/memory_b_cdd.py`
- **دليل التدريب الكامل**: `TRAINING_GUIDE.md`
- **المعلمون والأهداف**: `TEACHERS_AND_GOALS.md`

---

## تاريخ الإصدارات

| الإصدار | التاريخ | التغييرات |
|---|---|---|
| v9.2 | 2026-11-20 | إعادة تدريب Stage 2 من 150K→180K + فلتر OCR + تقطير 50% |
| v4.4 | 2026-11-20 | maestro.yml: cron 15→30د + timeout 15→14د (حل مشكلة الدقائق) |
| v4.3 | 2026-11-19 | إصلاح PYTHONUNBUFFERED للبث الحي للّوجز |
| v4.5 | 2026-11-18 | worker.js: STAGE_PRESETS الآمنة (batch=1, grad=16) |

---

*آخر تحديث: 2026-11-20*
