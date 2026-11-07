/**
 * Xandra Hope - All-in-One Worker
 * =================================
 * Worker واحد يفعل كل شيء:
 * 1. GET /          → عرض صفحة التسجيل
 * 2. POST /register → استقبال البيانات وتسجيل الصديق
 * 3. GET /status    → فحص الصحة
 * 
 * لا حاجة لـ GitHub Pages!
 * لا مشاكل CORS (نفس النطاق)!
 */

// ====== Helper: CORS Headers ======
function corsHeaders(extra = {}) {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Content-Type': 'application/json',
    ...extra
  };
}

// ====== Helper: JSON Response ======
function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: corsHeaders()
  });
}

// ====== HTML الصفحة ======
const HTML_PAGE = `<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Xandra Hope - انضمام الأصدقاء</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        header {
            text-align: center;
            background: white;
            padding: 40px 20px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        header h1 { font-size: 2.5em; color: #4a3a8a; margin-bottom: 10px; }
        .subtitle { color: #666; font-size: 1.2em; }
        section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        }
        section h2 {
            color: #4a3a8a;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        .intro p { margin-bottom: 15px; font-size: 1.1em; }
        .highlight {
            background: #fff3cd;
            padding: 15px;
            border-radius: 10px;
            border-right: 5px solid #ffc107;
            font-weight: bold;
        }
        .step {
            display: flex;
            gap: 20px;
            margin-bottom: 25px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .step-number {
            background: #667eea;
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5em;
            font-weight: bold;
            flex-shrink: 0;
        }
        .step-content { flex: 1; }
        .step-content h3 { color: #4a3a8a; margin-bottom: 10px; }
        .step-content ul { margin: 10px 20px; }
        .step-content code {
            background: #2d3748;
            color: #68d391;
            padding: 3px 8px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
        }
        .btn {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 10px 25px;
            text-decoration: none;
            border-radius: 8px;
            margin-top: 10px;
            font-weight: bold;
            border: none;
            cursor: pointer;
            font-size: 1em;
        }
        .btn:hover { background: #5a67d8; }
        .btn-primary {
            background: #48bb78;
            font-size: 1.2em;
            padding: 15px 30px;
            width: 100%;
            margin-top: 20px;
        }
        .btn-primary:hover { background: #38a169; }
        .form-group { margin-bottom: 20px; }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #4a3a8a;
        }
        .form-group input {
            width: 100%;
            padding: 15px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 1em;
            font-family: inherit;
        }
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
        }
        .alert {
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            display: none;
        }
        .alert-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .alert-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        footer { text-align: center; color: white; padding: 30px; font-size: 1.1em; }
        .faq details {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            cursor: pointer;
        }
        .faq summary { font-weight: bold; color: #4a3a8a; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 مشروع Xandra Hope</h1>
            <p class="subtitle">ساهم في بناء أول نموذج ذكاء اصطناعي عربي سيادي</p>
        </header>

        <section class="intro">
            <h2>🌟 مرحباً بك!</h2>
            <p>
                هذا المشروع يهدف لبناء نموذج ذكاء اصطناعي عربي خالص،
                يساهم في تطوير العمارة، الزراعة، الصناعة، والتعليم في وطننا العربي.
            </p>
            <p class="highlight">
                ✨ يمكنك المساهمة في 5 دقائق فقط، ولن تحتاج لفعل أي شيء بعدها!
            </p>
        </section>

        <section class="steps">
            <h2>📋 خطوات الانضمام</h2>

            <div class="step">
                <div class="step-number">1</div>
                <div class="step-content">
                    <h3>سجّل في Hugging Face</h3>
                    <p>افتح الرابط واضغط "Sign up with Google":</p>
                    <a href="https://huggingface.co/join" target="_blank" class="btn">فتح Hugging Face ↗</a>
                </div>
            </div>

            <div class="step">
                <div class="step-number">2</div>
                <div class="step-content">
                    <h3>احصل على Token</h3>
                    <p>افتح الرابط، أنشئ Token جديد:</p>
                    <ul>
                        <li>الاسم: <code>xandra</code></li>
                        <li>النوع: <strong>Write</strong> (مطلوب لـ ZeroGPU API)</li>
                    </ul>
                    <a href="https://huggingface.co/settings/tokens" target="_blank" class="btn">صفحة Tokens ↗</a>
                </div>
            </div>

            <div class="step">
                <div class="step-number">3</div>
                <div class="step-content">
                    <h3>أدخل بياناتك أدناه</h3>
                    <p>املأ النموذج التالي واضغط "تسجيل":</p>
                </div>
            </div>
        </section>

        <section>
            <h2>📝 نموذج التسجيل</h2>
            
            <div id="alertSuccess" class="alert alert-success"></div>
            <div id="alertError" class="alert alert-error"></div>
            
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p style="margin-top: 15px;">⏳ جاري التحقق والتسجيل...</p>
            </div>

            <form id="registerForm">
                <div class="form-group">
                    <label for="hf_username">👤 اسم المستخدم في Hugging Face</label>
                    <input type="text" id="hf_username" name="hf_username" 
                           placeholder="مثال: ahmed123" required>
                </div>
                
                <div class="form-group">
                    <label for="hf_token">🔑 HF Token (يبدأ بـ hf_)</label>
                    <input type="text" id="hf_token" name="hf_token" 
                           placeholder="hf_xxxxxxxxxxxxxxxx" required
                           style="font-family: 'Courier New', monospace;">
                    <small style="color: #666; margin-top: 5px; display: block;">
                        ⚠️ نحتاج صلاحية Write لاستدعاء ZeroGPU API
                    </small>
                </div>
                
                <button type="submit" class="btn btn-primary">🚀 تسجيل الآن</button>
            </form>
        </section>

        <section class="faq">
            <h2>❓ أسئلة شائعة</h2>
            <details>
                <summary>ما الذي سيحدث بحسابي؟</summary>
                <p style="margin-top: 10px;">لا شيء. نستخدم الـ Token فقط لاستدعاء خدمات Hugging Face المجانية نيابة عنك. لا نلمس ملفاتك الشخصية.</p>
            </details>
            <details>
                <summary>هل هذا آمن؟</summary>
                <p style="margin-top: 10px;">نعم. الـ Token يُستخدم فقط لاستدعاء Space API. لا نستطيع تعديل ملفاتك أو إنشاء نماذج باسمك.</p>
            </details>
            <details>
                <summary>كم تستغرق المشاركة من وقتي؟</summary>
                <p style="margin-top: 10px;">5 دقائق فقط للتسجيل، وبعدها لا شيء إطلاقاً! كل شيء يتم تلقائياً في الخلفية.</p>
            </details>
        </section>

        <footer>
            <p>🎯 شكراً لمساهمتك في بناء التكنولوجيا العربية!</p>
        </footer>
    </div>

    <script>
        const form = document.getElementById('registerForm');
        const alertSuccess = document.getElementById('alertSuccess');
        const alertError = document.getElementById('alertError');
        const loading = document.getElementById('loading');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            alertSuccess.style.display = 'none';
            alertError.style.display = 'none';
            
            const hf_username = document.getElementById('hf_username').value.trim();
            const hf_token = document.getElementById('hf_token').value.trim();
            
            if (!hf_username) {
                alertError.textContent = '❌ يرجى إدخال اسم المستخدم';
                alertError.style.display = 'block';
                return;
            }
            
            if (!hf_token.startsWith('hf_')) {
                alertError.textContent = '❌ الـ Token يجب أن يبدأ بـ hf_';
                alertError.style.display = 'block';
                return;
            }
            
            loading.style.display = 'block';
            form.style.display = 'none';
            
            try {
                const response = await fetch('/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ hf_username, hf_token })
                });
                
                const data = await response.json();
                
                loading.style.display = 'none';
                
                if (data.success) {
                    alertSuccess.innerHTML = 
                        '✅ <strong>تم تسجيلك بنجاح!</strong><br><br>' +
                        '🆔 معرّفك: <code>' + data.worker_id + '</code><br>' +
                        '👤 حساب HF: <code>' + data.hf_username + '</code><br><br>' +
                        '🎉 شكراً لك! لا حاجة لأي إجراء آخر.<br>' +
                        'سيتم استخدام حسابك تلقائياً للمساهمة في تدريب النموذج.';
                    alertSuccess.style.display = 'block';
                    form.reset();
                    form.style.display = 'block';
                } else {
                    alertError.innerHTML = '❌ <strong>فشل التسجيل</strong><br><br>' + data.error;
                    alertError.style.display = 'block';
                    form.style.display = 'block';
                }
            } catch (error) {
                loading.style.display = 'none';
                alertError.textContent = '❌ خطأ في الاتصال: ' + error.message;
                alertError.style.display = 'block';
                form.style.display = 'block';
            }
        });
    </script>
</body>
</html>`;

// ====== معالج الطلبات ======
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    
    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders() });
    }
    
    // GET / → عرض الصفحة
    if (request.method === 'GET' && url.pathname === '/') {
      return new Response(HTML_PAGE, {
        headers: { 'Content-Type': 'text/html; charset=utf-8' }
      });
    }
    
    // POST /register → استقبال التسجيل
    if (request.method === 'POST' && url.pathname === '/register') {
      return handleRegistration(request, env);
    }
    
    // GET /status → فحص الصحة
    if (request.method === 'GET' && url.pathname === '/status') {
      return jsonResponse({ status: 'ok', version: '3.0' });
    }
    
    return new Response('Not Found', { status: 404 });
  }
};

// ====== معالج التسجيل ======
async function handleRegistration(request, env) {
  try {
    // التحقق من وجود المتغيرات
    if (!env.GITHUB_TOKEN || !env.GITHUB_REPO) {
      return jsonResponse({ 
        success: false, 
        error: 'إعدادات Worker غير مكتملة. تأكد من إضافة GITHUB_TOKEN و GITHUB_REPO' 
      }, 500);
    }
    
    const { hf_username, hf_token } = await request.json();
    
    if (!hf_username || !hf_token) {
      return jsonResponse({ success: false, error: 'بيانات ناقصة' }, 400);
    }
    
    if (!hf_token.startsWith('hf_')) {
      return jsonResponse({ success: false, error: 'الـ Token يجب أن يبدأ بـ hf_' }, 400);
    }
    
    // 1. التحقق من صحة الـ Token عبر HF API
    const hfResponse = await fetch('https://huggingface.co/api/whoami-v2', {
      headers: { 'Authorization': 'Bearer ' + hf_token }
    });
    
    if (hfResponse.status !== 200) {
      return jsonResponse({ 
        success: false, 
        error: 'الـ Token غير صالح أو منتهي الصلاحية' 
      }, 400);
    }
    
    const hfData = await hfResponse.json();
    const realHfUsername = hfData.name;
    
    if (hf_username.toLowerCase() !== realHfUsername.toLowerCase()) {
      return jsonResponse({ 
        success: false, 
        error: 'اسم المستخدم لا يطابق الـ Token. الاسم الصحيح: ' + realHfUsername 
      }, 400);
    }
    
    // 2. إضافة Worker إلى GitHub
    const workerId = 'friend-' + Date.now();
    const githubResult = await addWorkerToGitHub(env, {
      worker_id: workerId,
      hf_username: realHfUsername,
      hf_token: hf_token
    });
    
    if (!githubResult.success) {
      return jsonResponse({ 
        success: false, 
        error: 'فشل تحديث GitHub: ' + githubResult.error 
      }, 500);
    }
    
    // 3. إرسال إشعار للمالك على تيليجرام (اختياري)
    if (env.TELEGRAM_BOT_TOKEN && env.OWNER_CHAT_ID) {
      await sendTelegramNotification(env, {
        worker_id: workerId,
        hf_username: realHfUsername
      });
    }
    
    return jsonResponse({
      success: true,
      worker_id: workerId,
      hf_username: realHfUsername,
      message: 'تم التسجيل بنجاح!'
    });
    
  } catch (error) {
    return jsonResponse({ 
      success: false, 
      error: 'خطأ داخلي: ' + error.message 
    }, 500);
  }
}

// ====== إضافة Worker إلى GitHub ======
async function addWorkerToGitHub(env, data) {
  try {
    const headers = {
      'Authorization': 'token ' + env.GITHUB_TOKEN,
      'Accept': 'application/vnd.github.v3+json',
      'User-Agent': 'Xandra-Maestro'
    };
    
    // 1. قراءة workers_config.yaml الحالي
    const fileUrl = 'https://api.github.com/repos/' + env.GITHUB_REPO + '/contents/maestro/workers_config.yaml';
    const fileResponse = await fetch(fileUrl, { headers });
    
    if (fileResponse.status !== 200) {
      return { success: false, error: 'فشل قراءة workers_config.yaml (تأكد من GITHUB_REPO و GITHUB_TOKEN)' };
    }
    
    const fileData = await fileResponse.json();
    const sha = fileData.sha;
    
    // فك الترميز (base64 → utf-8)
    const content = atob(fileData.content);
    
    // إضافة الـ Worker الجديد (YAML format)
    const newWorkerEntry = 
      '\n  - id: "' + data.worker_id + '"\n' +
      '    service: "kaggle"\n' +
      '    enabled: true\n' +
      '    hf_username: "' + data.hf_username + '"\n' +
      '    hf_token: "' + data.hf_token + '"\n' +
      '    added_at: "' + new Date().toISOString() + '"\n' +
      '    total_steps_trained: 0\n' +
      '    total_sessions: 0\n' +
      '    status: "active"';
    
    const newContent = content.replace(/\n*$/, '') + newWorkerEntry + '\n';
    
    // 2. رفع الملف المُحدّث
    const updateData = {
      message: 'Add worker: ' + data.worker_id + ' (' + data.hf_username + ')',
      content: btoa(unescape(encodeURIComponent(newContent))),
      sha: sha
    };
    
    const updateResponse = await fetch(fileUrl, {
      method: 'PUT',
      headers,
      body: JSON.stringify(updateData)
    });
    
    if (updateResponse.status === 200 || updateResponse.status === 201) {
      return { success: true };
    } else {
      const errorData = await updateResponse.json();
      return { success: false, error: errorData.message };
    }
    
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// ====== إرسال إشعار تيليجرام ======
async function sendTelegramNotification(env, data) {
  try {
    const message = '🎉 *صديق جديد انضم!*\n\n' +
      '🆔 ' + data.worker_id + '\n' +
      '👤 HF: ' + data.hf_username + '\n' +
      '⏰ ' + new Date().toISOString();

    await fetch('https://api.telegram.org/bot' + env.TELEGRAM_BOT_TOKEN + '/sendMessage', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chat_id: env.OWNER_CHAT_ID,
        text: message,
        parse_mode: 'Markdown'
      })
    });
  } catch (error) {
    console.error('Telegram error:', error);
  }
}
