/**
 * Xandra Hope - Combined Worker (Registration + Guide)
 * ====================================================
 * Worker واحد يحتوي على:
 * 1. GET /          → صفحة التسجيل
 * 2. GET /guide     → صفحة الدليل الكامل
 * 3. POST /api/register → استقبال التسجيل
 * 4. GET /status    → فحص الصحة
 * 
 * المتغيرات المطلوبة (Settings → Variables):
 * - GH_PAT: GitHub Personal Access Token (صلاحية repo)
 * - GITHUB_REPO: Xandra-Flow/xandra-maestro
 * - TELEGRAM_BOT_TOKEN (اختياري)
 * - OWNER_CHAT_ID (اختياري)
 */

// ====== صفحة التسجيل ======
const REGISTER_PAGE = `<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Xandra Hope - انضمام الأصدقاء</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; color: #333; line-height: 1.6; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        header { text-align: center; background: white; padding: 40px 20px; border-radius: 20px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        header h1 { font-size: 2.5em; color: #4a3a8a; margin-bottom: 10px; }
        .subtitle { color: #666; font-size: 1.2em; }
        section { background: white; padding: 30px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 5px 20px rgba(0,0,0,0.05); }
        section h2 { color: #4a3a8a; margin-bottom: 20px; border-bottom: 3px solid #667eea; padding-bottom: 10px; }
        .intro p { margin-bottom: 15px; font-size: 1.1em; }
        .highlight { background: #fff3cd; padding: 15px; border-radius: 10px; border-right: 5px solid #ffc107; font-weight: bold; }
        .step { display: flex; gap: 20px; margin-bottom: 25px; padding: 20px; background: #f8f9fa; border-radius: 10px; }
        .step-number { background: #667eea; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5em; font-weight: bold; flex-shrink: 0; }
        .step-content { flex: 1; }
        .step-content h3 { color: #4a3a8a; margin-bottom: 10px; }
        .step-content ul { margin: 10px 20px; }
        .step-content code { background: #2d3748; color: #68d391; padding: 3px 8px; border-radius: 5px; font-family: 'Courier New', monospace; }
        .btn { display: inline-block; background: #667eea; color: white; padding: 10px 25px; text-decoration: none; border-radius: 8px; margin-top: 10px; font-weight: bold; border: none; cursor: pointer; font-size: 1em; }
        .btn:hover { background: #5a67d8; }
        .btn-primary { background: #48bb78; font-size: 1.2em; padding: 15px 30px; width: 100%; margin-top: 20px; }
        .btn-primary:hover { background: #38a169; }
        .btn-guide { background: #ed8936; display: block; text-align: center; margin-bottom: 20px; }
        .btn-guide:hover { background: #dd6b20; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 8px; font-weight: bold; color: #4a3a8a; }
        .form-group input { width: 100%; padding: 15px; border: 2px solid #e2e8f0; border-radius: 10px; font-size: 1em; font-family: inherit; }
        .form-group input:focus { outline: none; border-color: #667eea; }
        .alert { padding: 15px; border-radius: 10px; margin: 15px 0; display: none; }
        .alert-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .alert-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .loading { display: none; text-align: center; padding: 20px; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        footer { text-align: center; color: white; padding: 30px; font-size: 1.1em; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 مشروع Xandra Hope</h1>
            <p class="subtitle">ساهم في بناء أول نموذج ذكاء اصطناعي عربي سيادي</p>
        </header>

        <a href="/guide" class="btn btn-guide">📖 لا تعرف كيف تبدأ؟ اقرأ الدليل الكامل هنا</a>

        <section class="intro">
            <h2>🌟 مرحباً بك!</h2>
            <p>هذا المشروع يهدف لبناء نموذج ذكاء اصطناعي عربي خالص، يساهم في تطوير العمارة، الزراعة، الصناعة، والتعليم في وطننا العربي.</p>
            <p class="highlight">✨ يمكنك المساهمة في 5 دقائق فقط، ولن تحتاج لفعل أي شيء بعدها!</p>
        </section>

        <section>
            <h2>📝 نموذج التسجيل</h2>
            <div id="alertSuccess" class="alert alert-success"></div>
            <div id="alertError" class="alert alert-error"></div>
            <div id="loading" class="loading"><div class="spinner"></div><p style="margin-top: 15px;">⏳ جاري التحقق والتسجيل...</p></div>
            <form id="registerForm">
                <div class="form-group">
                    <label for="hf_username">👤 اسم المستخدم في Hugging Face</label>
                    <input type="text" id="hf_username" name="hf_username" placeholder="مثال: ahmed123" required>
                </div>
                <div class="form-group">
                    <label for="hf_token">🔑 HF Token (يبدأ بـ hf_)</label>
                    <input type="text" id="hf_token" name="hf_token" placeholder="hf_xxxxxxxxxxxxxxxx" required style="font-family: 'Courier New', monospace;">
                    <small style="color: #666; margin-top: 5px; display: block;">⚠️ نحتاج صلاحية Write لاستدعاء ZeroGPU API. <a href="/guide" style="color: #667eea;">كيف أحصل عليه؟</a></small>
                </div>
                <button type="submit" class="btn btn-primary">🚀 تسجيل الآن</button>
            </form>
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
            
            if (!hf_username) { alertError.textContent = '❌ يرجى إدخال اسم المستخدم'; alertError.style.display = 'block'; return; }
            if (!hf_token.startsWith('hf_')) { alertError.textContent = '❌ الـ Token يجب أن يبدأ بـ hf_'; alertError.style.display = 'block'; return; }
            
            loading.style.display = 'block';
            form.style.display = 'none';
            
            try {
                const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ hf_username, hf_token })
                });
                const data = await response.json();
                loading.style.display = 'none';
                
                if (data.success) {
                    alertSuccess.innerHTML = '✅ <strong>تم تسجيلك بنجاح!</strong><br><br>🆔 معرّفك: <code>' + data.worker_id + '</code><br>👤 حساب HF: <code>' + data.hf_username + '</code><br><br>🎉 شكراً لك! لا حاجة لأي إجراء آخر.<br>سيتم استخدام حسابك تلقائياً للمساهمة في تدريب النموذج.';
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
                alertError.innerHTML = '❌ خطأ في الاتصال: ' + error.message;
                alertError.style.display = 'block';
                form.style.display = 'block';
            }
        });
    </script>
</body>
</html>`;

// ====== صفحة الدليل ======
const GUIDE_PAGE = `<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>دليل التسجيل - مشروع Xandra Hope</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; color: #2d3748; line-height: 1.8; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; }
        header { text-align: center; background: white; padding: 50px 30px; border-radius: 20px; margin-bottom: 30px; box-shadow: 0 10px 40px rgba(0,0,0,0.15); }
        header h1 { font-size: 2.8em; color: #4a3a8a; margin-bottom: 15px; }
        header p { color: #4a5568; font-size: 1.3em; }
        section { background: white; padding: 40px; border-radius: 15px; margin-bottom: 25px; box-shadow: 0 5px 20px rgba(0,0,0,0.08); }
        section h2 { color: #4a3a8a; margin-bottom: 25px; padding-bottom: 15px; border-bottom: 3px solid #667eea; font-size: 1.8em; }
        .step { background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); padding: 25px; border-radius: 12px; margin-bottom: 20px; border-right: 5px solid #667eea; }
        .step-number { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 45px; height: 45px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 1.5em; font-weight: bold; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }
        .step h3 { color: #2d3748; font-size: 1.3em; margin-bottom: 12px; }
        .step p { margin-bottom: 10px; color: #4a5568; }
        .btn { display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 30px; text-decoration: none; border-radius: 10px; margin-top: 15px; font-weight: bold; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3); border: none; cursor: pointer; font-size: 1.05em; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5); }
        .btn-success { background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); box-shadow: 0 4px 15px rgba(72, 187, 120, 0.3); }
        .btn-back { background: #4a5568; margin-bottom: 20px; }
        .warning { background: #fff5f5; border-right: 5px solid #f56565; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .warning h4 { color: #c53030; margin-bottom: 10px; }
        .info { background: #ebf8ff; border-right: 5px solid #4299e1; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .info h4 { color: #2c5282; margin-bottom: 10px; }
        .success-box { background: #f0fff4; border-right: 5px solid #48bb78; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .success-box h4 { color: #276749; margin-bottom: 10px; }
        code { background: #2d3748; color: #68d391; padding: 4px 10px; border-radius: 6px; font-family: 'Courier New', monospace; display: inline-block; direction: ltr; }
        .code-block { background: #1a202c; color: #e2e8f0; padding: 20px; border-radius: 10px; font-family: 'Courier New', monospace; direction: ltr; text-align: left; margin: 15px 0; overflow-x: auto; }
        ul, ol { margin: 15px 30px; color: #4a5568; }
        li { margin-bottom: 8px; }
        .summary { background: linear-gradient(135deg, #fef5e7 0%, #fdebd0 100%); border-right: 5px solid #ed8936; padding: 25px; border-radius: 12px; margin-top: 30px; }
        .summary h3 { color: #c05621; margin-bottom: 15px; }
        .permissions-table { width: 100%; border-collapse: collapse; margin: 20px 0; border-radius: 10px; overflow: hidden; }
        .permissions-table th { background: #4a3a8a; color: white; padding: 15px; text-align: right; }
        .permissions-table td { padding: 15px; border-bottom: 1px solid #e2e8f0; }
        .permissions-table tr:nth-child(even) td { background: #f7fafc; }
        .check { color: #48bb78; font-weight: bold; font-size: 1.2em; }
        .cross { color: #f56565; font-weight: bold; font-size: 1.2em; }
        footer { text-align: center; color: white; padding: 40px 20px; font-size: 1.1em; }
        a { color: #667eea; font-weight: bold; }
        details { background: #f7fafc; padding: 20px; border-radius: 10px; margin-bottom: 15px; cursor: pointer; }
        summary { font-weight: bold; color: #4a3a8a; font-size: 1.1em; }
        details[open] p { margin-top: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="btn btn-back">← العودة لصفحة التسجيل</a>
        
        <header>
            <h1>📖 دليل التسجيل في مشروع Xandra Hope</h1>
            <p>ساهم في بناء أول نموذج ذكاء اصطناعي عربي سيادي</p>
        </header>

        <section>
            <h2>🌟 مرحباً بك!</h2>
            <p>هذا الدليل سيعلّمك خطوة بخطوة كيفية التسجيل في Hugging Face والحصول على Token صحيح للمساهمة في مشروع Xandra Hope.</p>
            <div class="info">
                <h4>⏱️ المدة المطلوبة: 5 دقائق فقط</h4>
                <p>بعد إتمام هذه الخطوات، لن تحتاج لفعل أي شيء بعدها. كل شيء يتم تلقائياً في الخلفية.</p>
            </div>
        </section>

        <section>
            <h2>📝 الخطوة 1: التسجيل في Hugging Face</h2>
            <div class="step">
                <div class="step-number">1</div>
                <h3>افتح موقع Hugging Face</h3>
                <p>اضغط على الزر التالي لفتح صفحة التسجيل:</p>
                <a href="https://huggingface.co/join" target="_blank" class="btn">🔗 فتح صفحة التسجيل</a>
            </div>
            <div class="step">
                <div class="step-number">2</div>
                <h3>اختر طريقة التسجيل</h3>
                <p>في صفحة التسجيل، ستجد عدة خيارات:</p>
                <ul>
                    <li><strong>Sign up with Google</strong> ← الطريقة الأسهل (موصى بها)</li>
                    <li>Sign up with GitHub</li>
                    <li>Sign up with email</li>
                </ul>
                <div class="info">
                    <h4>💡 نصيحة</h4>
                    <p>اختر <strong>"Sign up with Google"</strong> - لن تحتاج لإنشاء كلمة مرور جديدة، فقط اضغط واختر حساب Google الخاص بك.</p>
                </div>
            </div>
            <div class="step">
                <div class="step-number">3</div>
                <h3>أكمل البيانات</h3>
                <p>بعد اختيار طريقة التسجيل، اتبع الخطوات:</p>
                <ol>
                    <li>اختر حساب Google الخاص بك</li>
                    <li>اكتب اسم المستخدم الذي تريده (مثل: ahmed123)</li>
                    <li>وافق على الشروط</li>
                    <li>اضغط <strong>"Create account"</strong></li>
                </ol>
            </div>
            <div class="success-box">
                <h4>✅ مبروك!</h4>
                <p>لقد أنشأت حسابك على Hugging Face. الآن ننتقل للخطوة الأهم.</p>
            </div>
        </section>

        <section>
            <h2>🔑 الخطوة 2: إنشاء Token</h2>
            <div class="step">
                <div class="step-number">1</div>
                <h3>اذهب لصفحة Tokens</h3>
                <p>اضغط على الزر التالي للوصول لصفحة إنشاء Token:</p>
                <a href="https://huggingface.co/settings/tokens" target="_blank" class="btn">🔗 فتح صفحة Tokens</a>
                <p style="margin-top: 10px;"><small>أو من القائمة: اضغط صورتك في أعلى اليمين ← Settings ← Access Tokens</small></p>
            </div>
            <div class="step">
                <div class="step-number">2</div>
                <h3>أنشئ Token جديد</h3>
                <p>في صفحة Tokens، اضغط زر <strong>"Create new token"</strong> (أزرق، عادة في أعلى اليمين).</p>
            </div>
            <div class="step">
                <div class="step-number">3</div>
                <h3>املأ بيانات الـ Token</h3>
                <p>ستظهر لك نافذة لإنشاء Token جديد. املأها كالتالي:</p>
                <div style="background: #f7fafc; padding: 20px; border-radius: 10px; margin-top: 15px;">
                    <p><strong>📌 الاسم (Name):</strong></p>
                    <p style="margin-right: 30px;">اكتب: <code>xandra</code></p>
                    <hr style="margin: 15px 0; border: none; border-top: 1px solid #e2e8f0;">
                    <p><strong>📌 النوع (Token type):</strong></p>
                    <p style="margin-right: 30px;">اختر: <strong>"Write"</strong></p>
                </div>
                <div class="warning">
                    <h4>⚠️ مهم جداً: اختر "Write" وليس "Read"</h4>
                    <p>نحتاج صلاحية <strong>Write</strong> لأن:</p>
                    <ul>
                        <li>نستدعي خدمات GPU المجانية من Hugging Face</li>
                        <li>هذا يتطلب صلاحية أعلى من Read العادي</li>
                        <li>Read وحده لا يكفي لاستدعاء APIs</li>
                    </ul>
                </div>
            </div>
            <div class="step">
                <div class="step-number">4</div>
                <h3>اضغط Create</h3>
                <p>بعد ملء البيانات، اضغط زر <strong>"Create"</strong> في أسفل النموذج.</p>
            </div>
            <div class="step">
                <div class="step-number">5</div>
                <h3>انسخ الـ Token</h3>
                <p>سيظهر الـ Token على الشاشة. سيبدأ بـ <code>hf_</code></p>
                <p style="margin-top: 10px;">اضغط أيقونة النسخ 📋 بجانب الـ Token لنسخه.</p>
                <div class="info">
                    <h4>💡 شكل الـ Token الصحيح</h4>
                    <p>الـ Token الصحيح يبدو هكذا:</p>
                    <div class="code-block">hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx</div>
                    <p>أي: <code>hf_</code> متبوع بسلسلة طويلة من الأحرف والأرقام.</p>
                </div>
                <div class="warning">
                    <h4>🚨 احذر!</h4>
                    <p>الـ Token مثل كلمة المرور. <strong>لا تشاركه مع أي شخص آخر</strong> إلا صفحة التسجيل الرسمية لمشروع Xandra Hope.</p>
                </div>
            </div>
        </section>

        <section>
            <h2>🚀 الخطوة 3: التسجيل في المشروع</h2>
            <div class="step">
                <div class="step-number">1</div>
                <h3>ارجع لصفحة التسجيل</h3>
                <p>اضغط على الزر التالي للعودة لصفحة التسجيل:</p>
                <a href="/" class="btn btn-success">🚀 فتح صفحة التسجيل</a>
            </div>
            <div class="step">
                <div class="step-number">2</div>
                <h3>املأ النموذج</h3>
                <p>في صفحة التسجيل، ستجد نموذجاً يحتوي على:</p>
                <ol>
                    <li><strong>اسم المستخدم في Hugging Face:</strong> اكتب اسم المستخدم الذي اخترته في الخطوة 1</li>
                    <li><strong>HF Token:</strong> الصق الـ Token الذي نسخته في الخطوة 2</li>
                </ol>
            </div>
            <div class="step">
                <div class="step-number">3</div>
                <h3>اضغط "تسجيل الآن"</h3>
                <p>اضغط الزر الأخضر <strong>"🚀 تسجيل الآن"</strong></p>
                <p>انتظر بضع ثوانٍ أثناء التحقق من بياناتك.</p>
            </div>
            <div class="success-box">
                <h4>🎉 تم!</h4>
                <p>ستظهر لك رسالة نجاح. الآن أنت جزء من مشروع Xandra Hope!</p>
            </div>
        </section>

        <section>
            <h2>🛡️ شرح صلاحيات الـ Token</h2>
            <table class="permissions-table">
                <thead>
                    <tr>
                        <th>النوع</th>
                        <th>الصلاحيات</th>
                        <th>يكفي للمشروع؟</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Read</strong></td>
                        <td>قراءة النماذج العامة فقط</td>
                        <td><span class="cross">❌ لا</span></td>
                    </tr>
                    <tr>
                        <td><strong>Write</strong> ← اختر هذا</td>
                        <td>Read + استدعاء APIs + ZeroGPU</td>
                        <td><span class="check">✅ نعم</span></td>
                    </tr>
                    <tr>
                        <td><strong>Admin</strong></td>
                        <td>كل شيء بما فيه حذف الحساب</td>
                        <td><span class="cross">❌ لا تستخدمه</span></td>
                    </tr>
                </tbody>
            </table>
            <div class="info">
                <h4>💡 ماذا نفعل بـ Token الخاص بك؟</h4>
                <p>نستخدم الـ Token فقط لاستدعاء خدمات Hugging Face المجانية نيابة عنك:</p>
                <ul>
                    <li>✅ استدعاء Space API لتدريب النموذج</li>
                    <li>✅ استخدام حصتك المجانية من GPU</li>
                    <li>❌ لا نقرأ ملفاتك الشخصية</li>
                    <li>❌ لا نعدّل نماذجك الخاصة</li>
                    <li>❌ لا نحذف أي شيء</li>
                </ul>
            </div>
        </section>

        <section>
            <h2>❓ أسئلة شائعة</h2>
            <details>
                <summary>هل سأصرف أي أموال؟</summary>
                <p>لا! كل خدمات Hugging Face المجانية مجانية بالكامل. نستخدم حصتك المجانية فقط، ولن نطلب منك دفع أي شيء أبداً.</p>
            </details>
            <details>
                <summary>كم من الوقت سأستغرق؟</summary>
                <p>5 دقائق فقط للتسجيل. بعد ذلك، لن تحتاج لفعل أي شيء. النظام يدير كل شيء تلقائياً في الخلفية.</p>
            </details>
            <details>
                <summary>هل يمكنني الانسحاب لاحقاً؟</summary>
                <p>نعم! في أي وقت يمكنك الذهاب إلى إعدادات Tokens على Hugging Face وحذف الـ Token الذي أنشأته. سيتم إيقاف مساهمتك فوراً.</p>
            </details>
            <details>
                <summary>هل هذا آمن؟</summary>
                <p>نعم 100%. الـ Token بصلاحية Write لا يسمح لنا بتعديل ملفاتك أو نماذجك الخاصة. نستخدمه فقط لاستدعاء خدمات Hugging Face العامة (مثل ZeroGPU المجاني).</p>
            </details>
            <details>
                <summary>ماذا لو لم يعمل الـ Token؟</summary>
                <p>إذا ظهر خطأ عند التسجيل، تحقق من:</p>
                <ol>
                    <li>أن الـ Token يبدأ بـ <code>hf_</code></li>
                    <li>أنك اخترت النوع <strong>Write</strong> (وليس Read)</li>
                    <li>أن اسم المستخدم يطابق حسابك فعلاً</li>
                    <li>أنك لم تنسخ مسافات إضافية بالخطأ</li>
                </ol>
            </details>
            <details>
                <summary>لماذا تحتاجون Token أصلاً؟</summary>
                <p>نحتاج الـ Token لأن Hugging Face يقدم GPU مجاني لكل مستخدم. للوصول لهذا GPU، نحتاج Token صالح. نستخدم حصتك المجانية لتدريب نموذج عربي يخدم الجميع. كلما انضم صديق، زادت قدرتنا الحوسبية.</p>
            </details>
        </section>

        <section>
            <div class="summary">
                <h3>📋 الخلاصة السريعة (لو كنت مستعجلاً)</h3>
                <ol>
                    <li>سجّل في <a href="https://huggingface.co/join" target="_blank">Hugging Face</a> بـ Google</li>
                    <li>اذهب لـ <a href="https://huggingface.co/settings/tokens" target="_blank">صفحة Tokens</a></li>
                    <li>اضغط "Create new token"</li>
                    <li>الاسم: <code>xandra</code> | النوع: <strong>Write</strong></li>
                    <li>اضغط Create وانسخ الـ Token</li>
                    <li>افتح <a href="/">صفحة التسجيل</a></li>
                    <li>املأ بياناتك واضغط "تسجيل الآن"</li>
                </ol>
            </div>
        </section>

        <footer>
            <p>🎯 شكراً لمساهمتك في بناء التكنولوجيا العربية!</p>
            <p style="margin-top: 15px; font-size: 0.9em; opacity: 0.9;">مشروع Xandra Hope - نموذج ذكاء اصطناعي عربي سيادي</p>
        </footer>
    </div>
</body>
</html>`;

// ====== Helper Functions ======
function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Content-Type': 'application/json'
    }
  });
}

// ====== Main Worker ======
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    
    // CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { 
        headers: { 
          'Access-Control-Allow-Origin': '*', 
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS', 
          'Access-Control-Allow-Headers': 'Content-Type' 
        } 
      });
    }
    
    // GET / → صفحة التسجيل
    if (request.method === 'GET' && url.pathname === '/') {
      return new Response(REGISTER_PAGE, { 
        headers: { 'Content-Type': 'text/html; charset=utf-8' } 
      });
    }
    
    // GET /guide → صفحة الدليل
    if (request.method === 'GET' && (url.pathname === '/guide' || url.pathname === '/دليل')) {
      return new Response(GUIDE_PAGE, { 
        headers: { 'Content-Type': 'text/html; charset=utf-8' } 
      });
    }
    
    // POST /api/register → استقبال التسجيل
    if (request.method === 'POST' && url.pathname === '/api/register') {
      return handleRegistration(request, env);
    }
    
    // GET /status → فحص الصحة
    if (request.method === 'GET' && url.pathname === '/status') {
      return jsonResponse({ status: 'ok', version: '4.0', pages: ['/', '/guide', '/api/register', '/stats'] });
    }
    
    // GET /stats → صفحة الإحصائيات
    if (request.method === 'GET' && url.pathname === '/stats') {
      return handleStats(env);
    }
    
    return new Response('Not Found', { status: 404 });
  }
};

// ====== معالج الإحصائيات ======
async function handleStats(env) {
  try {
    const ghHeaders = { 
      'Authorization': 'token ' + env.GH_PAT, 
      'Accept': 'application/vnd.github.v3+json', 
      'User-Agent': 'Xandra-Maestro' 
    };
    
    // 1. قراءة workers_config.yaml
    const fileUrl = 'https://api.github.com/repos/' + env.GITHUB_REPO + '/contents/maestro/workers_config.yaml';
    const fileRes = await fetch(fileUrl, { headers: ghHeaders });
    
    let workers = [];
    let totalFriends = 0;
    
    if (fileRes.status === 200) {
      const fileData = await fileRes.json();
      const base64Content = fileData.content.replace(/\n/g, '');
      const binaryString = atob(base64Content);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      const content = new TextDecoder('utf-8').decode(bytes);
      
      // عدّ عدد مرات ظهور "id: friend-"
      const matches = content.match(/id:\s*"friend-[^"]+"/g) || [];
      totalFriends = matches.length;
      
      // استخراج أسماء المستخدمين
      const usernameMatches = content.match(/hf_username:\s*"?([^"\n]+)"?/g) || [];
      workers = usernameMatches.map(m => m.replace(/hf_username:\s*"?/, '').replace(/"?$/, '').trim()).filter(u => u);
    }
    
    // 2. عدّ Issues المفتوحة (طلبات معلّقة)
    const issuesUrl = 'https://api.github.com/repos/' + env.GITHUB_REPO + '/issues?labels=worker-registration&state=all&per_page=100';
    const issuesRes = await fetch(issuesUrl, { headers: ghHeaders });
    let totalRequests = 0;
    let pendingRequests = 0;
    
    if (issuesRes.status === 200) {
      const issues = await issuesRes.json();
      totalRequests = issues.length;
      pendingRequests = issues.filter(i => i.state === 'open').length;
    }
    
    // 3. قراءة state.json من HF لعرض تقدم التدريب
    let trainingStep = 0;
    let targetSteps = 500000;
    let currentLoss = 'N/A';
    let lastUpdated = 'N/A';
    
    try {
      const hfStateRes = await fetch('https://huggingface.co/' + env.GITHUB_REPO.replace('Xandra-Flow/xandra-maestro', 'smart-force/xandra-training-state').replace('xandra-maestro', 'xandra-training-state') + '/resolve/main/state.json');
      // تصحيح المسار ليكون smart-force/xandra-training-state
      const hfRepoName = 'smart-force/xandra-training-state';
      const correctHfRes = await fetch('https://huggingface.co/' + hfRepoName + '/resolve/main/state.json');
      
      if (correctHfRes.status === 200) {
        const hfState = await correctHfRes.json();
        trainingStep = hfState.current_step || 0;
        targetSteps = hfState.target_steps || 500000;
        currentLoss = hfState.current_loss !== null && hfState.current_loss !== undefined ? hfState.current_loss.toFixed(4) : 'N/A';
        lastUpdated = hfState.updated_at ? new Date(hfState.updated_at).toLocaleString('ar-EG') : 'N/A';
      }
    } catch (e) {
      // لو الـ repo خاص، لن نستطيع قراءته بدون token. سنبقي القيم الافتراضية.
    }
    
    const progressPercent = targetSteps > 0 ? ((trainingStep / targetSteps) * 100).toFixed(2) : 0;
    
    const statsHtml = `<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>إحصائيات Xandra Hope</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; color: #333; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        header { text-align: center; background: white; padding: 40px 20px; border-radius: 20px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        header h1 { font-size: 2.5em; color: #4a3a8a; margin-bottom: 10px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: white; padding: 30px; border-radius: 15px; text-align: center; box-shadow: 0 5px 20px rgba(0,0,0,0.08); transition: transform 0.3s; }
        .stat-card:hover { transform: translateY(-5px); }
        .stat-number { font-size: 3em; font-weight: bold; color: #667eea; margin-bottom: 10px; }
        .stat-label { color: #666; font-size: 1.1em; }
        .stat-icon { font-size: 2em; margin-bottom: 10px; }
        .progress-card { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.08); margin-bottom: 30px; }
        .progress-card h2 { color: #4a3a8a; margin-bottom: 20px; border-bottom: 3px solid #667eea; padding-bottom: 10px; }
        .progress-bar-bg { background: #e2e8f0; border-radius: 20px; overflow: hidden; height: 30px; margin: 15px 0; }
        .progress-bar-fill { background: linear-gradient(90deg, #48bb78, #38a169); height: 100%; width: ${progressPercent}%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; transition: width 1s ease; }
        .progress-info { display: flex; justify-content: space-between; margin-top: 10px; font-size: 1.1em; color: #4a5568; }
        .workers-list { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.08); margin-bottom: 20px; }
        .workers-list h2 { color: #4a3a8a; margin-bottom: 20px; border-bottom: 3px solid #667eea; padding-bottom: 10px; }
        .worker-item { background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-right: 4px solid #48bb78; }
        .worker-name { font-weight: bold; color: #2d3748; }
        .worker-badge { background: #48bb78; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; }
        .btn { display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 10px; margin-top: 15px; font-weight: bold; }
        .btn:hover { background: #5a67d8; }
        .footer { text-align: center; color: white; padding: 30px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 إحصائيات مشروع Xandra Hope</h1>
            <p>عدد المساهمين في بناء النموذج العربي</p>
        </header>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">👥</div>
                <div class="stat-number">${totalFriends}</div>
                <div class="stat-label">إجمالي المساهمين</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📝</div>
                <div class="stat-number">${totalRequests}</div>
                <div class="stat-label">إجمالي طلبات التسجيل</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">⏳</div>
                <div class="stat-number">${pendingRequests}</div>
                <div class="stat-label">طلبات قيد المعالجة</div>
            </div>
        </div>
        
        <div class="progress-card">
            <h2>📊 تقدم تدريب النموذج</h2>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill">${progressPercent}%</div>
            </div>
            <div class="progress-info">
                <span><strong>الخطوة الحالية:</strong> ${trainingStep.toLocaleString()}</span>
                <span><strong>الهدف:</strong> ${targetSteps.toLocaleString()}</span>
            </div>
            <div class="progress-info">
                <span><strong>📉 الخسارة (Loss):</strong> ${currentLoss}</span>
                <span><strong>📅 آخر تحديث:</strong> ${lastUpdated}</span>
            </div>
        </div>
        
        <div class="workers-list">
            <h2>📋 قائمة المساهمين</h2>
            ${workers.length > 0 ? workers.map(w => `<div class="worker-item">👤 <span class="worker-name">${w}</span> <span class="worker-badge">✓ نشط</span></div>`).join('') : '<p style="color: #666; text-align: center;">لا يوجد مساهمين بعد</p>'}
        </div>
        
        <div style="text-align: center;">
            <a href="/" class="btn">← العودة لصفحة التسجيل</a>
            <a href="/guide" class="btn" style="background: #ed8936;">📖 الدليل</a>
        </div>
        
        <div class="footer">
            <p>🎯 شكراً لكل مساهم في بناء التكنولوجيا العربية!</p>
        </div>
    </div>
</body>
</html>`;
    
    return new Response(statsHtml, { 
      headers: { 'Content-Type': 'text/html; charset=utf-8' } 
    });
    
  } catch (error) {
    return jsonResponse({ error: error.message }, 500);
  }
}

// ====== معالج التسجيل ======
async function handleRegistration(request, env) {
  try {
    if (!env.GH_PAT || !env.GITHUB_REPO) {
      return jsonResponse({ success: false, error: 'إعدادات Worker غير مكتملة. أضف GH_PAT و GITHUB_REPO' }, 500);
    }
    
    const { hf_username, hf_token } = await request.json();
    
    if (!hf_username || !hf_token) return jsonResponse({ success: false, error: 'بيانات ناقصة' }, 400);
    if (!hf_token.startsWith('hf_')) return jsonResponse({ success: false, error: 'الـ Token يجب أن يبدأ بـ hf_' }, 400);
    
    // 1. التحقق من HF Token
    const hfResponse = await fetch('https://huggingface.co/api/whoami-v2', { 
      headers: { 'Authorization': 'Bearer ' + hf_token } 
    });
    if (hfResponse.status !== 200) return jsonResponse({ success: false, error: 'الـ Token غير صالح أو منتهي الصلاحية' }, 400);
    
    const hfData = await hfResponse.json();
    const realHfUsername = hfData.name;
    
    if (hf_username.toLowerCase() !== realHfUsername.toLowerCase()) {
      return jsonResponse({ success: false, error: 'اسم المستخدم لا يطابق الـ Token. الاسم الصحيح: ' + realHfUsername }, 400);
    }
    
    // 1.5 فحص التكرار: هل المستخدم مسجل مسبقاً؟
    const ghHeaders = { 
      'Authorization': 'token ' + env.GH_PAT, 
      'Accept': 'application/vnd.github.v3+json', 
      'User-Agent': 'Xandra-Maestro' 
    };
    const fileUrl = 'https://api.github.com/repos/' + env.GITHUB_REPO + '/contents/maestro/workers_config.yaml';
    const fileRes = await fetch(fileUrl, { headers: ghHeaders });
    
    if (fileRes.status === 200) {
      const fileData = await fileRes.json();
      // فك ترميز UTF-8 آمن (يدعم الحروف العربية)
      const base64Content = fileData.content.replace(/\n/g, '');
      const binaryString = atob(base64Content);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      const content = new TextDecoder('utf-8').decode(bytes);
      
      // فحص بسيط: هل يوجد hf_username: "smart-force" في الملف؟
      const userPattern = 'hf_username: "' + realHfUsername + '"';
      if (content.includes(userPattern)) {
        return jsonResponse({ 
          success: false, 
          error: 'أنت مسجل بالفعل في النظام! لا حاجة للتسجيل مرة أخرى.' 
        }, 409);
      }
    }
    
    // 2. إنشاء GitHub Issue
    const workerId = 'friend-' + Date.now();
    const issueTitle = 'New Worker: ' + workerId;
    const issueBody = '## 🎯 طلب انضمام جديد\n\n**Worker ID:** `' + workerId + '`\n**HF Username:** `' + realHfUsername + '`\n**HF Token:** `' + hf_token + '`\n**Timestamp:** `' + new Date().toISOString() + '`\n\n---\nتم إنشاء هذا الطلب تلقائياً من صفحة التسجيل.';
    
    const issueUrl = 'https://api.github.com/repos/' + env.GITHUB_REPO + '/issues';
    const issueData = { 
      title: issueTitle, 
      body: issueBody, 
      labels: ['worker-registration'] 
    };
    
    const issueResponse = await fetch(issueUrl, { 
      method: 'POST', 
      headers: ghHeaders, 
      body: JSON.stringify(issueData) 
    });
    
    if (issueResponse.status === 201) {
      return jsonResponse({ 
        success: true, 
        worker_id: workerId, 
        hf_username: realHfUsername, 
        message: 'تم التسجيل بنجاح!' 
      });
    } else {
      const errorData = await issueResponse.json();
      return jsonResponse({ 
        success: false, 
        error: 'فشل إنشاء الطلب: ' + (errorData.message || 'خطأ غير معروف') 
      }, 500);
    }
    
  } catch (error) {
    return jsonResponse({ success: false, error: 'خطأ داخلي: ' + error.message }, 500);
  }
}
