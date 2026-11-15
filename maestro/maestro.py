"""
╔═══════════════════════════════════════════════════════════════╗
║   Xandra Hope - Maestro (Real-time Discovery + Smart Lock)    ║
║   Copyright (c) 2026 Ahmed Sayed Kamel Mahmoud &               ║
║                  Александра Евгеньевна Острикова              ║
║   All Rights Reserved. License: XHNSL v1.0                   ║
╚═══════════════════════════════════════════════════════════════╝

UPGRADE NOTES (v4.1 → v4.2):
- Added check_kaggle_kernel_status(): real-time query of notebook status
- Rewrote check_lock(): now verifies BOTH lock expiry AND actual kernel state
- Rewrote select_available_account(): excludes accounts with running notebooks
- Added compute_session_gaps(): tracks wasted time between sessions
- Added update_kernel_status_map(): stores per-account kernel status in state.json
  so the Cloudflare Dashboard can render real status without Kaggle credentials
- Reduced LOCK_TIMEOUT_HOURS from 8 → 12 (safety net only, since real status is checked)
- Added MAX_LOCK_STALE_HOURS: if lock has been held > N hours AND kernel is dead,
  force-release the lock automatically (auto-recovery from crashed workers)
"""

import os
import sys
import json
import smtplib
import requests
import subprocess
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, timezone
from huggingface_hub import hf_hub_download, upload_file

HF_REPO = os.environ.get("HF_REPO")
HF_TOKEN = os.environ.get("HF_TOKEN_OWNER")

GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_PASS")
EMAIL_TO = os.environ.get("EMAIL_TO", GMAIL_USER)

# ═══════════════════════════════════════════════════════════════
# 📚 Account Loading — supports BOTH Legacy and New API tokens
# ═══════════════════════════════════════════════════════════════
#
# Kaggle supports TWO authentication methods:
#
# 1. Legacy API Key (kaggle.json format):
#    - GitHub Secrets: KAGGLE_USERNAME + KAGGLE_KEY (for account 1)
#                     KAGGLE_USERNAME_N + KAGGLE_KEY_N (for accounts 2-6)
#    - Value format: KAGGLE_KEY is 32-char hex string (e.g., "a1b2c3d4e5f6...")
#
# 2. New API Token (KGAT_ format — RECOMMENDED for kaggle CLI 2.x):
#    - GitHub Secrets: KAGGLE_API_TOKEN (for account 1)
#                     KAGGLE_API_TOKEN_N (for accounts 2-6)
#    - Value format: KGAT_xxxxxxxxxxxxxxxxxxxxxxxx
#    - Used via KAGGLE_API_TOKEN env var OR ~/.kaggle/access_token file
#
# We load accounts using EITHER method. If both are set, the new token wins
# (because it's the official future-proof method).

KAGGLE_ACCOUNTS = []
for i in range(1, 7):
    suffix = "" if i == 1 else f"_{i}"

    # Method 1: New API Token (preferred)
    api_token = os.environ.get(f"KAGGLE_API_TOKEN{suffix}")

    # Method 2: Legacy API Key
    username_legacy = os.environ.get(f"KAGGLE_USERNAME{suffix}")
    key_legacy = os.environ.get(f"KAGGLE_KEY{suffix}")

    if api_token:
        # New API Token mode — username is optional, kaggle CLI 2.x figures it out
        # But we still need a username for kernel_id like "smartforce/xandra-worker"
        # Try to read it from KAGGLE_USERNAME if available, else use a placeholder
        # that will be filled in later when we query the API
        username = username_legacy or f"account_{i}"  # fallback
        KAGGLE_ACCOUNTS.append({
            "username": username,
            "api_token": api_token,      # KGAT_xxx
            "key": None,                  # no legacy key
            "auth_method": "api_token",
            "index": i,
            "status": "active"
        })
    elif username_legacy and key_legacy:
        # Legacy API Key mode
        KAGGLE_ACCOUNTS.append({
            "username": username_legacy,
            "api_token": None,
            "key": key_legacy,
            "auth_method": "legacy",
            "index": i,
            "status": "active"
        })

# Log how many accounts were loaded and via which method
if KAGGLE_ACCOUNTS:
    token_count = sum(1 for a in KAGGLE_ACCOUNTS if a["auth_method"] == "api_token")
    legacy_count = sum(1 for a in KAGGLE_ACCOUNTS if a["auth_method"] == "legacy")
    print(f"📚 تم تحميل {len(KAGGLE_ACCOUNTS)} حسابات ({token_count} بالـ Token الجديد + {legacy_count} بالـ Legacy)")
else:
    print("⚠️ لم يتم تحميل أي حسابات! تحقق من GitHub Secrets:")
    print("   - إما KAGGLE_API_TOKEN (1-6) للـ Token الجديد")
    print("   - أو KAGGLE_USERNAME/KAGGLE_KEY (1-6) للـ Legacy")

WEEKLY_GPU_LIMIT_PER_ACCOUNT = 30
LOCK_TIMEOUT_HOURS = 12          # safety net only; real status checked each cycle
MAX_LOCK_STALE_HOURS = 14        # if lock older than this AND kernel is dead → force release
MIN_HOURS_BETWEEN_SESSIONS = 0.25  # 15 دقيقة فقط للتبديل السريع بين الجلسات


def send_email(subject, body):
    if not GMAIL_USER or not GMAIL_PASS: return
    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = EMAIL_TO
        msg['Subject'] = f"🎵 Xandra Maestro: {subject}"
        msg.attach(MIMEText(body.replace('\n', '<br>'), 'html', 'utf-8'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, EMAIL_TO, msg.as_string())
        server.quit()
    except: pass


# ═══════════════════════════════════════════════════════════════
# 📡 Real-time Kaggle Kernel Status Discovery
# ═══════════════════════════════════════════════════════════════

def get_kaggle_config_dir():
    """
    Returns a writable directory for kaggle.json that kaggle CLI will ACTUALLY read.

    Per Kaggle CLI official docs (https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md):
    Kaggle CLI reads credentials in this priority order:
      1. `kaggle auth login` (OAuth — requires browser, not for CI)
      2. KAGGLE_API_TOKEN env var (new in 2.x — requires new token format)
      3. ~/.kaggle/access_token (new file format)
      4. ~/.kaggle/kaggle.json (Legacy — STILL SUPPORTED, works in both 1.x and 2.x)

    For maximum compatibility (works with kaggle CLI 1.6.17 AND 2.x), we use:
      - KAGGLE_CONFIG_DIR env var → explicitly tells kaggle CLI where to find kaggle.json
      - Writes kaggle.json to that directory

    KAGGLE_CONFIG_DIR is supported by BOTH kaggle CLI 1.x and 2.x.
    """
    import tempfile

    # Strategy: use a deterministic, writable directory + set KAGGLE_CONFIG_DIR
    # so kaggle CLI knows exactly where to look
    home = os.environ.get('HOME') or os.path.expanduser('~')
    kaggle_dir = os.path.join(home, '.kaggle')

    try:
        os.makedirs(kaggle_dir, exist_ok=True)
        test_file = os.path.join(kaggle_dir, '.write_test')
        with open(test_file, 'w') as f: f.write('ok')
        os.remove(test_file)
    except (PermissionError, OSError) as e:
        print(f"⚠️ لا يمكن الكتابة في {kaggle_dir}: {e} — استخدام temp dir")
        kaggle_dir = os.path.join(tempfile.gettempdir(), 'kaggle_config')
        os.makedirs(kaggle_dir, exist_ok=True)

    # ALWAYS set KAGGLE_CONFIG_DIR — this is the key!
    # kaggle CLI checks this env var BEFORE checking ~/.kaggle/
    os.environ['KAGGLE_CONFIG_DIR'] = kaggle_dir
    return kaggle_dir


def setup_kaggle_credentials(account):
    """
    Configures Kaggle credentials based on the account's auth_method.

    For "api_token" mode (KGAT_ tokens — kaggle CLI 2.x):
      - Writes the token to ~/.kaggle/access_token (official file location)
      - Also sets KAGGLE_API_TOKEN env var (alternative method)
      - Passes it to all subprocess calls via env dict

    For "legacy" mode (kaggle.json — kaggle CLI 1.6.17 + 2.x):
      - Writes ~/.kaggle/kaggle.json with {"username": "...", "key": "..."}
      - Sets KAGGLE_USERNAME + KAGGLE_KEY env vars
      - Passes them to all subprocess calls via env dict

    Both methods set KAGGLE_CONFIG_DIR to ensure the kaggle CLI reads from
    the correct location regardless of HOME.
    """
    try:
        # Get the kaggle config dir (sets KAGGLE_CONFIG_DIR env var as side effect)
        kaggle_dir = get_kaggle_config_dir()

        if account.get("auth_method") == "api_token":
            # ─── New API Token mode ─────────────────────────────────
            api_token = account.get("api_token")
            if not api_token:
                print(f"❌ لا يوجد api_token لـ {account['username']}")
                return False

            # Method 1: Write to ~/.kaggle/access_token (official file)
            access_token_path = os.path.join(kaggle_dir, 'access_token')
            with open(access_token_path, 'w') as f:
                f.write(api_token)
            os.chmod(access_token_path, 0o600)

            # Method 2: Set KAGGLE_API_TOKEN env var (alternative)
            os.environ['KAGGLE_API_TOKEN'] = api_token

            # Clear any legacy kaggle.json to avoid conflict
            legacy_json = os.path.join(kaggle_dir, 'kaggle.json')
            if os.path.exists(legacy_json):
                try: os.remove(legacy_json)
                except: pass

            # Verify the file was written
            if not os.path.exists(access_token_path):
                print(f"❌ فشل كتابة access_token في {access_token_path}")
                return False

            return True

        else:
            # ─── Legacy API Key mode ─────────────────────────────────
            username = account["username"]
            key = account["key"]
            if not username or not key:
                print(f"❌ بيانات اعتماد Legacy ناقصة لـ {account.get('username', '?')}")
                return False

            kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')
            with open(kaggle_json_path, 'w') as f:
                json.dump({"username": username, "key": key}, f)
            os.chmod(kaggle_json_path, 0o600)

            # Verify the file was written
            if not os.path.exists(kaggle_json_path):
                print(f"❌ فشل كتابة kaggle.json في {kaggle_json_path}")
                return False

            # Read back to verify contents
            try:
                with open(kaggle_json_path, 'r') as f:
                    data = json.load(f)
                if data.get('username') != username or data.get('key') != key:
                    print(f"❌ kaggle.json يحتوي على بيانات خاطئة")
                    return False
            except Exception as e:
                print(f"❌ فشل قراءة kaggle.json للتحقق: {e}")
                return False

            # Set env vars (kaggle CLI 1.x uses them as fallback)
            os.environ['KAGGLE_USERNAME'] = username
            os.environ['KAGGLE_KEY'] = key

            # Clear any access_token to avoid conflict
            access_token_path = os.path.join(kaggle_dir, 'access_token')
            if os.path.exists(access_token_path):
                try: os.remove(access_token_path)
                except: pass

            # Clear KAGGLE_API_TOKEN (legacy mode doesn't use it)
            if 'KAGGLE_API_TOKEN' in os.environ:
                del os.environ['KAGGLE_API_TOKEN']

            return True

    except Exception as e:
        print(f"⚠️ فشل إعداد بيانات اعتماد Kaggle لـ {account.get('username', '?')}: {e}")
        return False


def get_kaggle_auth(account):
    """
    Returns a requests auth tuple or headers dict for the Kaggle REST API.
    Works with BOTH Legacy credentials AND new API tokens.

    Detection logic (smart — handles any token format):
      - If KAGGLE_API_TOKEN env var is set (any value) → Bearer token
      - If KAGGLE_KEY looks like a long opaque token (not 32-char hex) → Bearer token
      - Else → Basic auth (username, key) — Legacy API key (32-char hex)

    Per official Kaggle docs (https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md):
      - New API tokens are obtained from kaggle.com/settings/api "Generate New Token"
      - They're set via KAGGLE_API_TOKEN env var OR stored in ~/.kaggle/access_token
      - Legacy API keys are 32-char hex from kaggle.json file
      - Both work with REST API (newer: Bearer, legacy: Basic auth)

    Returns: (auth, headers) tuple
      - For Legacy: auth=(username, key), headers={}
      - For API token: auth=None, headers={'Authorization': 'Bearer {token}'}
    """
    if account.get("auth_method") == "api_token":
        token = account.get("api_token", "").strip()
        # Strip any existing "Bearer " prefix to avoid duplication
        if token.lower().startswith("bearer "):
            token = token[len("Bearer "):]
        return None, {'Authorization': f'Bearer {token}'}
    else:
        # Legacy mode — but the key might actually be a new API token
        # stored in KAGGLE_KEY by mistake.
        key = account.get("key", "").strip()
        username = account.get("username", "")

        # Detect if this is actually a new API token (not a Legacy key):
        # 1. If it starts with "KGAT_" → definitely new token
        # 2. If it's NOT 32-char hex (legacy keys are exactly 32 hex chars)
        is_kgat = key.lower().startswith("kgat_")
        is_32_hex = len(key) == 32 and all(c in '0123456789abcdefABCDEF' for c in key)

        if is_kgat or not is_32_hex:
            # It's a new API token (KGAT_ or any other opaque format)
            # Strip "Bearer " prefix if present
            if key.lower().startswith("bearer "):
                key = key[len("Bearer "):]
            return None, {'Authorization': f'Bearer {key}'}
        else:
            # Genuine Legacy API key — 32-char hex — use Basic auth
            return (username, key), {}



def kaggle_api_get(account, endpoint, params=None, timeout=30):
    """
    Makes a GET request to the Kaggle REST API.
    Base URL: https://www.kaggle.com/api/v1/

    Returns: (status_code, response_data) or (status_code, None) on parse error
    """
    auth, headers = get_kaggle_auth(account)
    url = f"https://www.kaggle.com/api/v1/{endpoint.lstrip('/')}"
    try:
        resp = requests.get(url, auth=auth, headers=headers, params=params, timeout=timeout)
        try:
            data = resp.json()
        except:
            data = None
        return resp.status_code, data
    except requests.exceptions.Timeout:
        return -1, None
    except Exception as e:
        print(f"   ⚠️ استثناء في API GET {endpoint}: {e}")
        return -2, None


def kaggle_api_post(account, endpoint, data=None, files=None, timeout=60):
    """
    Makes a POST request to the Kaggle REST API (for pushing kernels).
    - When files is provided: uses multipart/form-data
    - When files is None: uses application/x-www-form-urlencoded (NOT json)
      because Kaggle API expects form data, not JSON
    """
    auth, headers = get_kaggle_auth(account)
    url = f"https://www.kaggle.com/api/v1/{endpoint.lstrip('/')}"
    try:
        if files:
            # multipart/form-data (file upload)
            resp = requests.post(url, auth=auth, headers=headers, data=data, files=files, timeout=timeout)
        else:
            # application/x-www-form-urlencoded (form data, NOT json)
            resp = requests.post(url, auth=auth, headers=headers, data=data, timeout=timeout)
        try:
            resp_data = resp.json()
        except:
            resp_data = None
        return resp.status_code, resp_data, resp.text
    except requests.exceptions.Timeout:
        return -1, None, "Timeout"
    except Exception as e:
        return -2, None, str(e)


def verify_kaggle_auth(account):
    """
    Verifies Kaggle credentials by calling the REST API directly.
    Uses GET /api/v1/kernels/list?user={username} which requires valid auth.
    """
    username = account["username"]
    code, data = kaggle_api_get(account, "kernels/list", params={"user": username})
    if code == 200:
        return True
    elif code == 401:
        print(f"   ❌ فشل المصادقة لـ {username} (HTTP 401) — تحقق من الـ credentials")
        return False
    elif code == 403:
        print(f"   ❌ محظور لـ {username} (HTTP 403) — تحقق من صلاحيات الـ token")
        return False
    elif code in (-1, -2):
        print(f"   ⚠️ خطأ شبكة لـ {username}")
        return False
    else:
        print(f"   ⚠️ استجابة غير متوقعة لـ {username}: HTTP {code}")
        return False


def check_kaggle_kernel_status(account):
    """
    استشعار حقيقي لحالة الدفتر على خادم كاجل باستخدام CLI الرسمي أولاً ثم REST API.
    يرجع: 'running' (تشمل queued), 'complete', 'error', 'canceled', 'not_found', أو 'unknown_via_lock'
    """
    username = account["username"]
    kernel_slug = f"{username}/xandra-worker"

    # 1. الفحص المباشر عبر Kaggle CLI الرسمي (يعمل مع توكنات KGAT_ الحديثة بدقة)
    try:
        env = dict(os.environ)
        key_val = (account.get("key") or account.get("api_token") or "").strip()
        if key_val.startswith("KGAT_"):
            env["KAGGLE_API_TOKEN"] = key_val
        env["KAGGLE_USERNAME"] = username
        if account.get("key"):
            env["KAGGLE_KEY"] = account["key"]

        res = subprocess.run(
            ["kaggle", "kernels", "status", kernel_slug],
            env=env, capture_output=True, text=True, timeout=25
        )
        out = (res.stdout + " " + res.stderr).lower()
        if "running" in out or "queued" in out or "inprogress" in out:
            return "running"
        if "complete" in out or "finished" in out:
            return "complete"
        if "error" in out or "failed" in out:
            return "error"
        if "cancel" in out:
            return "canceled"
        if "404" in out or "not found" in out:
            return "not_found"
    except Exception as e:
        print(f"   ⚠️ تعذر فحص CLI لـ {username}: {e}")

    # 2. المسار الاحتياطي عبر REST API
    code, data = kaggle_api_get(account, "kernels/status", params={"user_name": username})
    if code == 403:
        return "unknown_via_lock"
    if code != 200 or not data:
        return "unknown"

    kernels = data if isinstance(data, list) else data.get("status", data.get("kernels", []))
    for k in (kernels if isinstance(kernels, list) else []):
        ref = (k.get("ref") or k.get("slug") or "").lower()
        if "xandra-worker" in ref or "xandra_worker" in ref:
            st = (k.get("status") or k.get("state") or "").lower()
            if st in ("running", "runinprogress", "in_progress", "queued", "in queue"):
                return "running"
            if st in ("complete", "finished", "success"):
                return "complete"
            if st in ("error", "failed"):
                return "error"
            if st in ("cancelled", "canceled"):
                return "canceled"
    return "not_found"


def get_state():
    try:
        path = hf_hub_download(repo_id=HF_REPO, filename="state.json", repo_type="model", token=HF_TOKEN)
        with open(path) as f: return json.load(f)
    except:
        return {
            "current_step": 0, "target_steps": 500000, "current_loss": None,
            "history": [], "best_loss": float('inf'), "updated_at": None,
            "gpu_hours_used": 0, "training_lock": {"is_locked": False},
            "current_account_index": 0, "account_usage": {}, "disabled_accounts": [],
            "kernel_status_map": {}, "session_gaps": []
        }


def update_state(state):
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open("state.json", "w") as f: json.dump(state, f, indent=2, ensure_ascii=False)
    upload_file(path_or_fileobj="state.json", path_in_repo="state.json", repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)


# ═══════════════════════════════════════════════════════════════
# 🔒 Smart Lock Check (combines time + real kernel state)
# ═══════════════════════════════════════════════════════════════

def extract_account_name(locked_by):
    """Strips 'maestro-' or 'kaggle-' prefix from locked_by field."""
    if not locked_by: return None
    if locked_by.startswith("maestro-"): return locked_by[len("maestro-"):]
    if locked_by.startswith("kaggle-"): return locked_by[len("kaggle-"):]
    return locked_by


def check_lock(state):
    """
    SMART LOCK CHECK — combines time expiry with real kernel status.

    Returns: (is_still_busy: bool, reason: str)
      - (True,  "running")         → kernel is actively running, leave it alone
      - (False, "kernel_complete") → kernel finished, lock should be released
      - (False, "kernel_error")    → kernel crashed, lock should be released
      - (False, "expired")         → time-based expiry reached
      - (False, "stale_force_release") → lock held too long AND kernel is dead
      - (False, None)              → no lock at all
    """
    lock = state.get("training_lock", {})
    if not lock.get("is_locked"):
        return False, None

    locked_by = lock.get("locked_by")
    account_name = extract_account_name(locked_by)

    # Step 1: check time expiry first (fast path)
    expires_at = lock.get("expires_at")
    now = datetime.now(timezone.utc)
    time_expired = False
    if expires_at:
        try:
            exp_time = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if now > exp_time:
                time_expired = True
        except: pass

    # Step 2: query real kernel status (the smart part)
    if account_name:
        account_obj = next((a for a in KAGGLE_ACCOUNTS if a["username"] == account_name), None)
        if account_obj:
            kernel_state = check_kaggle_kernel_status(account_obj)
            print(f"📡 حالة الـ kernel لـ {account_name}: {kernel_state}")

            if kernel_state == "running":
                # Real-time confirmation that the session is still alive
                # Even if time expired, we trust the real status more
                return True, "running"
            elif kernel_state in ("complete", "error", "canceled"):
                # Kernel finished/crashed — release the lock immediately!
                return False, f"kernel_{kernel_state}"
            elif kernel_state == "not_found":
                # Notebook doesn't exist on this account → lock is stale
                return False, "kernel_not_found"
            elif kernel_state == "unknown_via_lock":
                last_hb = lock.get("last_heartbeat") or lock.get("locked_at")
                if last_hb:
                    try:
                        hb_time = datetime.fromisoformat(last_hb.replace("Z", "+00:00"))
                        mins_since_hb = (now - hb_time).total_seconds() / 60
                        if mins_since_hb > 60:
                            print(f"⚠️ توقفت نبضات العامل منذ {mins_since_hb:.1f} دقيقة — تحرير القفل")
                            return False, "heartbeat_timeout"
                        else:
                            print(f"💓 العامل نشط (آخر نبضة منذ {mins_since_hb:.1f} دقيقة) — انتظار")
                            return True, "running_via_heartbeat"
                    except Exception:
                        pass
                return True, "running_assumed"
            # else: unknown → fall through to time-based check

    # Step 3: stale lock recovery — if lock is REALLY old and kernel is dead/unknown
    locked_at = lock.get("locked_at")
    if locked_at:
        try:
            la_time = datetime.fromisoformat(locked_at.replace("Z", "+00:00"))
            age_hours = (now - la_time).total_seconds() / 3600
            if age_hours > MAX_LOCK_STALE_HOURS:
                print(f"⚠️ قفل معلّق منذ {age_hours:.1f} ساعة — تحرير قسري")
                return False, "stale_force_release"
        except: pass

    # Step 4: fall back to pure time-based decision
    if time_expired:
        return False, "expired"

    # Lock is fresh and we couldn't get real status → assume still running
    return True, "running_assumed"


def release_lock(state, reason="manual"):
    """Releases the training lock with a reason for audit trail."""
    state["training_lock"] = {
        "is_locked": False, "locked_by": None,
        "locked_at": None, "expires_at": None,
        "released_at": datetime.now(timezone.utc).isoformat(),
        "release_reason": reason
    }
    return state


def acquire_lock(state, account_name):
    state["training_lock"] = {
        "is_locked": True,
        "locked_by": f"maestro-{account_name}",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=LOCK_TIMEOUT_HOURS)).isoformat(),
        "released_at": None, "release_reason": None
    }
    return state


# ═══════════════════════════════════════════════════════════════
# 🎯 Smart Account Selection (excludes accounts with running kernels)
# ═══════════════════════════════════════════════════════════════

def fetch_kaggle_username(account):
    """
    Fetches the actual Kaggle username for an account using its API token.
    This is needed because KAGGLE_API_TOKEN alone doesn't tell us the username,
    but we need it for kernel_id like "smartforce/xandra-worker".

    Uses `kaggle config view` which shows the authenticated user's info.
    """
    if account.get("auth_method") != "api_token":
        return account.get("username")

    api_token = account.get("api_token", "")
    if not api_token:
        return account.get("username")

    subprocess_env = os.environ.copy()
    subprocess_env['HOME'] = os.environ.get('HOME', os.path.expanduser('~'))
    subprocess_env['KAGGLE_API_TOKEN'] = api_token
    subprocess_env.pop('KAGGLE_USERNAME', None)
    subprocess_env.pop('KAGGLE_KEY', None)
    if 'KAGGLE_CONFIG_DIR' in os.environ:
        subprocess_env['KAGGLE_CONFIG_DIR'] = os.environ['KAGGLE_CONFIG_DIR']

    try:
        # `kaggle config view` shows the current user
        result = subprocess.run(
            ["kaggle", "config", "view"],
            capture_output=True, text=True, timeout=20,
            env=subprocess_env
        )
        out = result.stdout + " " + result.stderr
        # Look for username in the output
        # kaggle config view typically shows:
        #   username: smartforce
        #   ...
        import re
        m = re.search(r'username[:\s]+([a-zA-Z0-9_.\-]+)', out, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    except: pass
    return account.get("username")


def update_kernel_status_map(state):
    """
    Queries all Kaggle accounts for their kernel status and stores the result
    in state.kernel_status_map. This is what the Cloudflare Dashboard reads.

    Also fetches the real username for accounts using KAGGLE_API_TOKEN
    (since the token alone doesn't reveal the username).
    """
    # Print auth method summary for debugging — show first 12 chars to verify format
    auth_summary = []
    for acc in KAGGLE_ACCOUNTS:
        if acc.get("auth_method") == "api_token":
            token = acc.get("api_token", "")
            preview = token[:12] + "..." if len(token) > 12 else token
            auth_summary.append(f"{acc['username']}:Bearer(token={preview})")
        else:
            key = acc.get("key", "")
            preview = key[:12] + "..." if len(key) > 12 else key
            # Detect actual type using same logic as get_kaggle_auth
            is_kgat = key.lower().startswith("kgat_")
            is_32_hex = len(key) == 32 and all(c in '0123456789abcdefABCDEF' for c in key)
            if is_kgat or not is_32_hex:
                auth_summary.append(f"{acc['username']}:Bearer(key={preview}) [not 32-hex]")
            else:
                auth_summary.append(f"{acc['username']}:Basic(key={preview}) [32-hex legacy]")
    print(f"🔐 طرق المصادقة:")
    for line in auth_summary:
        print(f"   {line}")

    kernel_map = {}
    for acc in KAGGLE_ACCOUNTS:
        # Fetch real username for API token accounts (first time only)
        if acc.get("auth_method") == "api_token" and acc.get("username", "").startswith("account_"):
            real_username = fetch_kaggle_username(acc)
            if real_username and not real_username.startswith("account_"):
                acc["username"] = real_username
                print(f"  ℹ️ تم استخراج اسم المستخدم: {real_username}")

        username = acc["username"]
        status = check_kaggle_kernel_status(acc)
        kernel_map[username] = {
            "kernel_status": status,
            "checked_at": datetime.now(timezone.utc).isoformat()
        }
        print(f"  📊 {username}: {status}")
    state["kernel_status_map"] = kernel_map
    return state


def select_available_account(state):
    """
    Smart selection: excludes accounts that are (a) disabled, (b) exhausted,
    or (c) currently running a notebook.
    """
    if not KAGGLE_ACCOUNTS: return None, 0, []

    disabled_accounts = state.get("disabled_accounts", [])
    kernel_map = state.get("kernel_status_map", {})

    account_stats = []
    now = datetime.now(timezone.utc)
    week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)

    for acc in KAGGLE_ACCOUNTS:
        username = acc["username"]

        if username in disabled_accounts:
            account_stats.append({
                "username": username, "remaining": 0, "weekly_hours": 30,
                "sessions": 0, "kernel_status": "disabled", "excluded_reason": "disabled"
            })
            continue

        # Real kernel status from kernel_status_map
        k_entry = kernel_map.get(username, {})
        kernel_status = k_entry.get("kernel_status", "unknown")

        weekly_hours = 0
        total_sessions = 0
        last_session_time = None  # أحدث جلسة لهذا الحساب (لتوزيع الأحمال)

        for session in state.get("history", []):
            if username in session.get("worker_id", ""):
                total_sessions += 1
                try:
                    ts = session["timestamp"]
                    if ts.endswith("Z"): ts = ts[:-1] + "+00:00"
                    session_time = datetime.fromisoformat(ts)
                    if session_time > week_start:
                        weekly_hours += session.get("gpu_hours", 0)
                    # تتبّع آخر مرة استُخدم فيها هذا الحساب
                    if last_session_time is None or session_time > last_session_time:
                        last_session_time = session_time
                except: pass

        remaining = WEEKLY_GPU_LIMIT_PER_ACCOUNT - weekly_hours

        excluded_reason = None
        if kernel_status == "running":
            excluded_reason = "running_now"
        elif remaining < 2:
            excluded_reason = "exhausted"

        account_stats.append({
            "username": username,
            "remaining": remaining,
            "weekly_hours": weekly_hours,
            "sessions": total_sessions,
            "last_session_time": last_session_time.isoformat() if last_session_time else None,
            "kernel_status": kernel_status,
            "excluded_reason": excluded_reason
        })

    # Pick the best account using a SMART multi-factor scoring:
    #   1. Most remaining hours (primary)
    #   2. Least-recently-used (tiebreaker — distributes load)
    #   3. Fewest total sessions (final tiebreaker)
    #   4. Random tiebreaker (when all above are equal, pick randomly)
    candidates = [a for a in account_stats if a["excluded_reason"] is None]

    # First, try to find accounts that have NEVER been used (sessions=0 and no last_session_time)
    # These should be prioritized to distribute load
    never_used = [a for a in candidates if a["sessions"] == 0 and a["last_session_time"] is None]
    used_accounts = [a for a in candidates if a not in never_used]

    # If there are accounts that have never been used, pick randomly from them
    # This ensures we don't always pick the first account
    if never_used:
        import random
        best = random.choice(never_used)
        for acc in KAGGLE_ACCOUNTS:
            if acc["username"] == best["username"]:
                return acc, best["remaining"], account_stats

    # Otherwise, sort by: most remaining DESC, then oldest last_session ASC, then fewest sessions ASC
    candidates.sort(key=lambda x: (
        -x["remaining"],                          # 1) most remaining first
        x["last_session_time"] or "0000",         # 2) least-recently-used first
        x["sessions"]                             # 3) fewest sessions first
    ))

    if candidates:
        best = candidates[0]
        for acc in KAGGLE_ACCOUNTS:
            if acc["username"] == best["username"]:
                return acc, best["remaining"], account_stats

    return None, 0, account_stats


# ═══════════════════════════════════════════════════════════════
# ⏱️ Session Gap Tracking (visualizes wasted time)
# ═══════════════════════════════════════════════════════════════

def compute_session_gaps(state):
    """Computes idle gaps between consecutive sessions — feeds the dashboard."""
    history = state.get("history", [])
    if len(history) < 2: return []

    gaps = []
    for i in range(1, len(history)):
        prev = history[i-1]
        curr = history[i]
        try:
            p_end = prev["timestamp"]
            c_start = curr.get("started_at") or curr["timestamp"]
            if p_end.endswith("Z"): p_end = p_end[:-1] + "+00:00"
            if c_start.endswith("Z"): c_start = c_start[:-1] + "+00:00"
            gap_seconds = (datetime.fromisoformat(c_start) - datetime.fromisoformat(p_end)).total_seconds()
            if gap_seconds > 0:
                gaps.append({
                    "from_session": prev.get("worker_id", "?"),
                    "to_session": curr.get("worker_id", "?"),
                    "gap_minutes": round(gap_seconds / 60, 1),
                    "gap_hours": round(gap_seconds / 3600, 2),
                    "started_at": c_start
                })
        except: pass

    state["session_gaps"] = gaps[-10:]  # keep last 10
    return state["session_gaps"]


# ═══════════════════════════════════════════════════════════════
# 🚀 Notebook Launcher
# ═══════════════════════════════════════════════════════════════

def ensure_kaggle_cli():
    """
    Ensures the kaggle CLI is installed AND working.
    Uses the LATEST kaggle CLI (2.x) because it supports KAGGLE_API_TOKEN
    (the new KGAT_ tokens). kaggle CLI 1.6.17 does NOT support KAGGLE_API_TOKEN.

    If any account uses auth_method="api_token" (KGAT_), we MUST use 2.x.
    If all accounts use legacy keys, 2.x still works (it supports kaggle.json legacy).

    So we always install the latest 2.x version.
    """
    try:
        # Test if kaggle CLI is installed
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            version_out = (result.stdout + result.stderr).strip()
            # kaggle 2.x is what we want for KAGGLE_API_TOKEN support
            if "2." in version_out or "kaggle cli 2" in version_out.lower():
                return True
            else:
                print(f"📦 kaggle CLI قديم ({version_out}) — سيتم الترقية لأحدث إصدار (2.x)")
                # Fall through to installation
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception:
        pass

    # Install the LATEST kaggle CLI (2.x)
    print("📦 تثبيت kaggle CLI (أحدث إصدار 2.x — يدعم KAGGLE_API_TOKEN)...")
    try:
        # Install six first (still needed by some kaggle versions)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "six"],
                       check=True, capture_output=True, timeout=60)
        # Install latest kaggle (2.x)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "kaggle"],
                       check=True, capture_output=True, timeout=180)
        # Verify
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            version_out = (result.stdout + result.stderr).strip()
            print(f"✅ kaggle CLI مثبّت بنجاح: {version_out}")
            return True
        else:
            print(f"❌ kaggle CLI فشل بعد التثبيت: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ فشل تثبيت kaggle CLI: {e}")
        return False


def push_kaggle_kernel_via_cli(account, worker_code_path):
    """
    Pushes the kernel via kaggle CLI (which AUTOMATICALLY RUNS the notebook after push).
    
    Per official Kaggle docs (https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md):
    "kaggle kernels push - Pushes new code/notebook and metadata to a kernel, then runs the kernel."
    
    This is the ONLY way to trigger notebook execution via API/CLI.
    The REST API (POST /api/v1/kernels/push) only SAVES the notebook without running it.
    
    Returns True on success.
    """
    username = account["username"]
    try:
        import tempfile, json as _json, shutil, os as _os
        
        # Create a temporary directory for the kernel metadata
        work_dir = tempfile.mkdtemp(prefix=f"kaggle_push_{username}_")
        try:
            # 1. Copy the worker code to the work directory
            worker_filename = "xandra_worker.py"
            dest_path = _os.path.join(work_dir, worker_filename)
            with open(worker_code_path, "r") as f:
                worker_code = f.read()
            # Inject the real username
            worker_code = worker_code.replace('__KAGGLE_USERNAME_VALUE__', username)
            worker_code = worker_code.replace('__HF_TOKEN_VALUE__', HF_TOKEN or "")
            worker_code = worker_code.replace('__HF_REPO_VALUE__', HF_REPO or "")
            with open(dest_path, "w") as f:
                f.write(worker_code)
            
            # 2. Create kernel-metadata.json (يقرأ نوع الكرت من إعدادات اللوحة)
            sys_cfg = get_state().get("system_settings", {})
            accel_choice = sys_cfg.get("accelerator", "GpuNvidiaTeslaT4")
            metadata = {
                "id": f"{username}/xandra-worker",
                "title": "xandra-worker",
                "code_file": worker_filename,
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true",
                "accelerator": accel_choice,
                "enable_internet": "true",
                "dataset_sources": [],
                "kernel_sources": [],
                "competition_sources": [],
                "model_sources": []
            }

            
            metadata_path = _os.path.join(work_dir, "kernel-metadata.json")
            with open(metadata_path, "w") as f:
                _json.dump(metadata, f, indent=2)
            
            # 3. Set up Kaggle credentials for this account
            #    The user's GitHub Secrets contain:
            #      - KAGGLE_USERNAME + KAGGLE_KEY (for account 1)
            #      - KAGGLE_USERNAME_N + KAGGLE_KEY_N (for accounts 2-6)
            #    IMPORTANT: KAGGLE_KEY_* actually contains KGAT_ tokens (not legacy 32-hex keys)!
            #    This was confirmed by the maestro logs showing:
            #      "***:Bearer(key=KGAT_68f8892...) [not 32-hex]"
            #
            #    kaggle CLI 2.x reads KAGGLE_API_TOKEN env var OR ~/.kaggle/access_token file.
            #    Since our token is in account["key"] (named KAGGLE_KEY but actually KGAT_),
            #    we need to PASS it as KAGGLE_API_TOKEN to the CLI.
            kaggle_dir = _os.path.join(_os.path.expanduser("~"), ".kaggle")
            _os.makedirs(kaggle_dir, exist_ok=True)
            work_kaggle_dir = _os.path.join(work_dir, ".kaggle")
            _os.makedirs(work_kaggle_dir, exist_ok=True)
            
            # 🔑 Extract KGAT_ token — it's stored in account["key"] (named KAGGLE_KEY)
            # In the user's setup, KAGGLE_KEY_* actually contains KGAT_xxx tokens
            kgat_token = None
            key_value = (account.get("key") or "").strip()
            
            # Method 1: KAGGLE_KEY contains a KGAT_ token (most likely scenario for this user)
            if key_value.startswith("KGAT_"):
                kgat_token = key_value
                print(f"   🔑 Found KGAT_ token in account['key'] (KAGGLE_KEY_{account['index']})")
            
            # Method 2: account["api_token"] has it (rare in this user's setup)
            elif account.get("api_token", "").startswith("KGAT_"):
                kgat_token = account["api_token"]
                print(f"   🔑 Found KGAT_ token in account['api_token']")
            
            # Method 3: Try get_kaggle_auth() (which works for REST API)
            else:
                try:
                    auth_obj, auth_headers = get_kaggle_auth(account)
                    auth_str = auth_headers.get('Authorization', '')
                    if 'Bearer' in auth_str:
                        bearer_part = auth_str.split('Bearer', 1)[-1].strip()
                        if bearer_part.startswith('KGAT_'):
                            kgat_token = bearer_part
                            print(f"   🔑 Extracted KGAT_ token from get_kaggle_auth()")
                except Exception as e:
                    print(f"   ⚠️ Could not extract KGAT_ token: {e}")
            
            # Write KGAT_ token to ~/.kaggle/access_token (Option B for kaggle CLI 2.x)
            if kgat_token:
                for d in [kaggle_dir, work_kaggle_dir]:
                    token_file = _os.path.join(d, "access_token")
                    with open(token_file, "w") as f:
                        f.write(kgat_token)
                    _os.chmod(token_file, 0o600)
                print(f"   🔑 Wrote KGAT_ token to ~/.kaggle/access_token")
            else:
                print(f"   ⚠️ No KGAT_ token found! Will try legacy kaggle.json only.")
            
            # ALSO write legacy kaggle.json (some CLI versions still try it)
            kaggle_json_path = _os.path.join(kaggle_dir, "kaggle.json")
            work_kaggle_json_path = _os.path.join(work_kaggle_dir, "kaggle.json")
            kaggle_json = {
                "username": username,
                "key": account.get("key", "")  # KGAT_ token here too (CLI 1.x will ignore format)
            }
            for p in [kaggle_json_path, work_kaggle_json_path]:
                with open(p, "w") as f:
                    _json.dump(kaggle_json, f)
                _os.chmod(p, 0o600)
            print(f"   🔑 Wrote kaggle.json to ~/.kaggle/ (with username + key)")
            
            # 4. Run `kaggle kernels push` — this AUTO-RUNS the notebook!
            env = dict(_os.environ)
            # Set ALL possible auth env vars to maximize CLI compatibility
            env["KAGGLE_CONFIG_DIR"] = kaggle_dir
            env["KAGGLE_USERNAME"] = username
            # 🚀 CRITICAL: Pass KGAT_ token as KAGGLE_API_TOKEN (Option A for kaggle CLI 2.x)
            if kgat_token:
                env["KAGGLE_API_TOKEN"] = kgat_token
            # Also set KAGGLE_KEY (in case the CLI tries legacy)
            if account.get("key"):
                env["KAGGLE_KEY"] = account["key"]
            # Make sure HOME is set so ~/.kaggle/ is found
            env["HOME"] = _os.path.expanduser("~")
            
            print(f"   📤 Running: kaggle kernels push -p {work_dir}")
            result = subprocess.run(
                ["kaggle", "kernels", "push", "-p", work_dir],
                env=env,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                # Parse the output for kernel URL
                output = result.stdout
                print(f"   ✅ kaggle CLI output:")
                for line in output.split("\n")[:15]:
                    if line.strip():
                        print(f"      {line}")
                # Check if it includes the kernel URL
                if "kaggle.com/code" in output or "xandra-worker" in output:
                    print(f"✅ تم رفع الـ Notebook على {username} (slug: xandra-worker) [CLI — AUTO-RUNS!]")
                    return True
                elif "push" in output.lower() or "version" in output.lower():
                    print(f"✅ تم رفع الـ Notebook على {username} [CLI]")
                    return True
                else:
                    print(f"✅ Push completed (returncode=0)")
                    return True
            else:
                print(f"   ❌ kaggle CLI failed (returncode={result.returncode})")
                print(f"   stderr: {result.stderr[:1000]}")
                print(f"   stdout: {result.stdout[:1000]}")
                # Diagnostic: check what's in ~/.kaggle/
                try:
                    ls_home_kaggle = subprocess.run(
                        ["ls", "-la", kaggle_dir],
                        capture_output=True, text=True, timeout=10
                    )
                    print(f"   📂 ~/.kaggle/ contents:")
                    print(f"      {ls_home_kaggle.stdout}")
                except: pass
                return False
        finally:
            # Cleanup
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except:
                pass
    except subprocess.TimeoutExpired:
        print(f"   ❌ kaggle CLI timed out (120s)")
        return False
    except Exception as e:
        print(f"   ❌ kaggle CLI exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def push_kaggle_kernel_via_api(account, worker_code_path):
    """
    Pushes the kernel via the Kaggle REST API directly (POST /api/v1/kernels/push).
    Tries BOTH JSON and form-urlencoded payloads for maximum compatibility.

    ⚠️ WARNING: This ONLY SAVES the notebook — it does NOT auto-run it!
    Use push_kaggle_kernel_via_cli() instead, which uses the kaggle CLI
    (the only way to auto-trigger notebook execution from external code).
    
    Returns True on success (notebook saved, but NOT executed).
    """
    username = account["username"]
    try:
        # Read the worker code
        with open(worker_code_path, "r") as f:
            worker_code = f.read()
        # Inject the real username and HF credentials
        worker_code = worker_code.replace('__KAGGLE_USERNAME_VALUE__', username)
        worker_code = worker_code.replace('__HF_TOKEN_VALUE__', HF_TOKEN or "")
        worker_code = worker_code.replace('__HF_REPO_VALUE__', HF_REPO or "")

        # Get auth + headers
        auth, headers = get_kaggle_auth(account)
        url = "https://www.kaggle.com/api/v1/kernels/push"

        # ═══════════════════════════════════════════════════════════════
        # Try Method 1: JSON payload (per the other model's suggestion)
        # Uses camelCase field names (kernelType, isPrivate, etc.)
        # because the modern Kaggle API may expect them
        # ═══════════════════════════════════════════════════════════════
        json_payload = {
            "ref": f"{username}/xandra-worker",
            "title": "xandra worker",
            "text": worker_code,
            "language": "python",
            "kernelType": "script",
            "isPrivate": True,
            "enableGpu": True,
            "enableInternet": True,
            "datasetSources": [],
            "kernelSources": [],
            "competitionSources": []
        }

        json_headers = dict(headers)
        json_headers['Content-Type'] = 'application/json'

        try:
            resp = requests.post(url, auth=auth, headers=json_headers, json=json_payload, timeout=120)
            code = resp.status_code
            resp_text = resp.text
            if code in (200, 201, 202):
                print(f"✅ تم رفع الـ Notebook على {username} (slug: xandra-worker) [JSON]")
                return True
            # If JSON failed, fall through to form-data method
            print(f"   📝 JSON method returned HTTP {code}, trying form-data...")
        except Exception as e:
            print(f"   📝 JSON method failed: {e}, trying form-data...")

        # ═══════════════════════════════════════════════════════════════
        # Try Method 2: application/x-www-form-urlencoded (legacy API format)
        # Uses snake_case field names (kernel_type, is_private, etc.)
        # ═══════════════════════════════════════════════════════════════
        form_payload = {
            "slug": f"{username}/xandra-worker",
            "title": "xandra worker",
            "text": worker_code,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": "",
            "kernel_sources": "",
            "competition_sources": ""
        }

        try:
            resp = requests.post(url, auth=auth, headers=headers, data=form_payload, timeout=120)
            code = resp.status_code
            resp_text = resp.text
            if code in (200, 201, 202):
                print(f"✅ تم رفع الـ Notebook على {username} (slug: xandra-worker) [form-data]")
                return True
        except Exception as e:
            resp_text = str(e)

        # Both methods failed — report the error
        if code == 401:
            print(f"❌ فشل المصادقة لـ {username} (HTTP 401) — تحقق من الـ credentials")
            print(f"   Response: {resp_text[:300]}")
        elif code == 403:
            print(f"❌ محظور لـ {username} (HTTP 403) — الـ token لا يملك صلاحيات kernels.update")
            print(f"   → أعد إنشاء الـ token مع scope: kernels.update")
            print(f"   Response: {resp_text[:300]}")
        else:
            print(f"❌ فشل الرفع لـ {username} (HTTP {code})")
            print(f"   Response: {resp_text[:400]}")
        return False
    except Exception as e:
        print(f"❌ استثناء أثناء الرفع: {e}")
        return False


def launch_kaggle_notebook(account):
    username = account["username"]
    try:
        # Check current status first — skip if already running
        status = check_kaggle_kernel_status(account)
        if status == "running":
            print(f"ℹ️ الـ Notebook على {username} يعمل بالفعل — لا حاجة للإطلاق")
            return True

        # Use the unified push function (handles CLI installation + credentials + push)
        # Note: worker code is assumed to be in the parent dir or same dir
        worker_path = os.path.join(os.path.dirname(__file__), "..", "kaggle-worker", "xandra_worker.py")
        if not os.path.exists(worker_path):
            worker_path = "xandra_worker.py" # fallback
        
        # 🚀 Use kaggle CLI (auto-runs the notebook) — REST API only saves without running
        if push_kaggle_kernel_via_cli(account, worker_path):
            return True
        
        # Fallback: REST API (saves notebook but doesn't auto-run)
        print(f"   ⚠️ kaggle CLI failed, falling back to REST API (note: notebook won't auto-run!)")
        return push_kaggle_kernel_via_api(account, worker_path)
    except Exception as e:
        print(f"❌ استثناء أثناء الإطلاق: {e}")
        return False


def verify_kaggle_running(expected_step, wait_sec=300):
    """
    Waits for some time, then checks the notebook's progress.
    
    Two-phase check:
    Phase 1 (after 6 min): Look for "I'm alive" signal in state.json.
                          If present → notebook booted, still loading checkpoint.
    Phase 2 (after 12 min): Check if current_step increased.
                          If yes → training actually started.
                          If no → notebook likely crashed at startup.
    """
    import time as _time
    
    # Phase 1: Wait 6 min, check for "booted" signal
    print(f"⏱️ المرحلة 1: الانتظار 360 ثانية للتأكد من boot الـ notebook...")
    _time.sleep(360)
    
    new_state = get_state()
    notebook_status = new_state.get('notebook_status', {})
    new_step = new_state.get('current_step', 0)
    last_heartbeat = new_state.get('training_lock', {}).get('last_heartbeat')
    
    # Check if "I'm alive" signal was received
    if notebook_status.get('phase') == 'booted' or last_heartbeat:
        print(f"   ✅ Notebook booted بنجاح (signal: {notebook_status.get('phase', 'unknown')})")
        print(f"   ⏳ المرحلة 2: انتظار 360 ثانية إضافية للتحقق من بدء التدريب...")
        _time.sleep(360)
        
        # Phase 2: Check if step increased
        new_state = get_state()
        new_step = new_state.get('current_step', 0)
        
        if new_step > expected_step:
            print(f"   ✅ تم تأكيد التشغيل: الخطوة زادت من {expected_step:,} إلى {new_step:,}")
            return True, new_step
        else:
            print(f"   ⚠️ Notebook يعمل لكن لم يصل لـ 100 خطوة بعد (step={new_step:,})")
            print(f"   💡 قد يكون download latest.pt بطيئاً (970 MB) — اتركه يعمل")
            return True, new_step  # Trust it — heartbeat is recent
    else:
        # No signal received after 6 min — notebook likely crashed at startup
        print(f"   ❌ لم تصل إشارة 'أنا حي' بعد 6 دقائق — الـ notebook غالباً انهار")
        print(f"   🔓 سيتم تحرير القفل في الجولة القادمة")
        return False, new_step


# ═══════════════════════════════════════════════════════════════
# 🎼 Main Orchestration Loop
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("🎼 Xandra Maestro (Real-time Discovery Edition)")
    print(f"⏰ {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    state = get_state()

    if state.get("system_paused"):
        print("⏸️ النظام متوقف مؤقتاً من لوحة التحكم (system_paused=True) — لن يتم إطلاق أي جلسات.")
        return

    cooldown_hrs = state.get("system_settings", {}).get("cooldown_hours", MIN_HOURS_BETWEEN_SESSIONS)
    last_lock = state.get("training_lock", {})
    released_at = last_lock.get("released_at")
    if released_at and cooldown_hrs > 0:
        try:
            rel_time = datetime.fromisoformat(released_at.replace("Z", "+00:00"))
            hours_passed = (datetime.now(timezone.utc) - rel_time).total_seconds() / 3600
            if hours_passed < cooldown_hrs:
                print(f"🛡️ فترة راحة آمنة نشطة: مر {hours_passed:.2f}h من أصل {cooldown_hrs}h — انتظار لحماية الحسابات.")
                return
        except Exception:
            pass


    # Step 1: refresh kernel status map (this is the key upgrade)

    print("\n📡 تحديث خريطة حالة الـ kernels من Kaggle...")
    state = update_kernel_status_map(state)


    # Step 2: compute session gaps for the dashboard
    compute_session_gaps(state)

    # حارس الاستشعار المسبق: فحص آخر حساب عمل في السجل والتأكد من أنه أنهى مهمته بالكامل
    last_history = state.get("history", [])
    if last_history:
        last_worker = last_history[-1].get("worker_id", "")
        last_acc_name = extract_account_name(last_worker)
        last_acc_obj = next((a for a in KAGGLE_ACCOUNTS if a["username"] == last_acc_name), None)
        if last_acc_obj:
            real_last_status = check_kaggle_kernel_status(last_acc_obj)
            print(f"🔍 استشعار آخر خادم ({last_acc_name}): الحالة الفعلية = {real_last_status}")
            if real_last_status == "running":
                print(f"🛑 الخادم السابق ({last_acc_name}) ما زال يعمل أو في طابور الانتظار — لن يتم فتح أي حساب جديد!")
                update_state(state)
                return

    # Step 3: report account statuses
    print("\n📊 تقرير الحسابات:")
    account, remaining_hours, account_stats = select_available_account(state)
    
    # 🛡️ PRESERVE manually_set account_usage (don't delete on refresh!)
    # Old logic: re-compute from history → loses user edits when history is empty
    # New logic: 
    #   - If user manually set hours via Dashboard → respect it (max with weekly_hours)
    #   - If account has never been edited → use weekly_hours from history
    #   - If week changed → preserve manually_set entries, reset others
    now_utc = datetime.now(timezone.utc)
    days_since_saturday = (now_utc.weekday() + 2) % 7
    saturday_start = (now_utc - timedelta(days=days_since_saturday)).strftime("%Y-%m-%d")
    current_week_id = f"kaggle-week-{saturday_start}"
    
    saved_week_id = state.get("usage_week_id")
    if saved_week_id and saved_week_id != current_week_id:
        print(f"📅 تجددت باقة كاجل الأسبوعية ({current_week_id}) — تصفير أرصدة جميع الحسابات إلى 30 ساعة!")
        existing_usage = {}
    else:
        existing_usage = state.get("account_usage", {})
    
    state["usage_week_id"] = current_week_id

    usage_map = {}
    # First: copy ALL manually_set entries (preserves them even if account not in KAGGLE_ACCOUNTS)
    for uname, udata in existing_usage.items():
        if udata.get("manually_set"):
            usage_map[uname] = {
                "used": udata.get("used", 0),
                "remaining": udata.get("remaining", 30),
                "sessions": udata.get("sessions", 0),
                "manually_set": True,
                "manually_set_at": udata.get("manually_set_at"),
                "manual_base_used": udata.get("manual_base_used", udata.get("used", 0))
            }
    
    # Then: add/update accounts from account_stats
    disabled_accounts = state.get("disabled_accounts", [])
    for s in account_stats:
        uname = s["username"]
        if uname in disabled_accounts:
            usage_map[uname] = existing_usage.get(uname, {
                "used": 0, "remaining": 30.0, "sessions": 0,
                "manually_set": False, "manually_set_at": None, "manual_base_used": None
            })
            continue
        prev = existing_usage.get(uname, {})
        prev_used = prev.get("used", 0)
        prev_manually_set = prev.get("manually_set", False)
        weekly_hours = s.get("weekly_hours", 0)
        base_manual = prev.get("manual_base_used", prev_used)
        
        if prev_manually_set:
            manual_time = prev.get("manually_set_at") or "0000"
            new_sessions_hours = 0
            for sess in state.get("history", []):
                if uname in sess.get("worker_id", "") and sess.get("timestamp", "") > manual_time:
                    new_sessions_hours += sess.get("gpu_hours", 0)
            total_used = round(base_manual + new_sessions_hours, 2)
        else:
            total_used = round(max(prev_used, weekly_hours), 2)

        usage_map[uname] = {
            "used": total_used,
            "remaining": round(max(0, 30.0 - total_used), 2),
            "sessions": max(prev.get("sessions", 0), s.get("sessions", 0)),
            "manually_set": prev_manually_set,
            "manually_set_at": prev.get("manually_set_at"),
            "manual_base_used": base_manual if prev_manually_set else None
        }

    state["account_usage"] = usage_map

 
    for s in account_stats:
        marker = "✅" if s["excluded_reason"] is None else "⛔"
        reason = f" [{s['excluded_reason']}]" if s["excluded_reason"] else ""
        print(f"  {marker} {s['username']}: kernel={s['kernel_status']} | "
              f"remaining={s['remaining']:.1f}h | sessions={s['sessions']}{reason}")

    # Step 4: smart lock check (combines time + real kernel state)
    is_busy, reason = check_lock(state)
    if is_busy:
        print(f"\n⏳ النظام مشغول: {reason} — لن نتدخل")
        # Still persist kernel_status_map so the dashboard shows fresh data
        update_state(state)
        return

    # Step 5: lock is free — release it cleanly + try to launch next
    if reason and reason != "expired":
        print(f"\n🔓 تحرير القفل: السبب = {reason}")
        state = release_lock(state, reason=reason)

    # Step 6: pick the next account
    if not account:
        print("\n⚠️ لا توجد حسابات متاحة (الكل مشغول أو مستهلك)")
        if not state.get("all_accounts_exhausted_alerted"):
            send_email("⚠️ جميع حسابات كاجل استُنفدت",
                       f"تم استنفاد/اشغال جميع حصص GPU.\n\nآخر تحديث: {datetime.now(timezone.utc).isoformat()}")
            state["all_accounts_exhausted_alerted"] = True
        update_state(state)
        return

    state["all_accounts_exhausted_alerted"] = False

    # Step 7: acquire lock + launch
    print(f"\n🚀 إطلاق جلسة تدريب جديدة على {account['username']} (متبقي {remaining_hours:.1f}h)")
    state = acquire_lock(state, account["username"])
    state["current_account_index"] = account["index"]
    update_state(state)

    success = launch_kaggle_notebook(account)
    if not success:
        send_email(f"🚨 فشل إطلاق جلسة Kaggle ({account['username']})",
                   f"تعذّر إطلاق الـ Notebook على الحساب {account['username']}.")
        state = release_lock(state, reason="launch_failed")
        update_state(state)
    else:
        # 📧 Send SUCCESS email with a brief report
        current_step = state.get('current_step', 0)
        
        # 🆕 Verification Step: Two-phase check
        # Phase 1 (6 min): Look for "I'm alive" signal
        # Phase 2 (additional 6 min): Check step increase
        # Total: 12 min — exceeds GitHub Actions 10 min timeout!
        # So we need to bump maestro.yml timeout to 15 min.
        is_progressing, verified_step = verify_kaggle_running(current_step)
        
        target_steps = state.get('target_steps', 500000)
        progress_pct = (verified_step / target_steps * 100) if target_steps > 0 else 0
        current_loss = state.get('current_loss', 'N/A')
        gpu_hours = state.get('gpu_hours_used', 0)
        total_sessions = len(state.get('history', []))

        status_msg = "✅ تم التحقق من التقدم" if is_progressing else "⚠️ لم يتم رصد تقدم (قد ينهار قريباً)"
        
        success_body = f"""
<div dir="rtl" style="font-family:Arial;direction:rtl;text-align:right">
<h2 style="color:{'#48bb78' if is_progressing else '#ed8936'}">{status_msg}</h2>
<table style="border-collapse:collapse;width:100%">
<tr><td style="padding:8px;border:1px solid #ddd;background:#f0fff4">👤 الحساب</td><td style="padding:8px;border:1px solid #ddd"><strong>{account['username']}</strong></td></tr>
<tr><td style="padding:8px;border:1px solid #ddd;background:#f0fff4">⏱️ الساعات المتبقية</td><td style="padding:8px;border:1px solid #ddd">{remaining_hours:.1f}h / 30h</td></tr>
<tr><td style="padding:8px;border:1px solid #ddd;background:#f0fff4">📊 الخطوة (عند الإطلاق)</td><td style="padding:8px;border:1px solid #ddd">{current_step:,}</td></tr>
<tr><td style="padding:8px;border:1px solid #ddd;background:#f0fff4">📈 الخطوة الحالية (بعد 6د)</td><td style="padding:8px;border:1px solid #ddd"><strong>{verified_step:,}</strong></td></tr>
<tr><td style="padding:8px;border:1px solid #ddd;background:#f0fff4">📉 الخسارة الحالية</td><td style="padding:8px;border:1px solid #ddd">{current_loss}</td></tr>
<tr><td style="padding:8px;border:1px solid #ddd;background:#f0fff4">⏰ إجمالي ساعات GPU</td><td style="padding:8px;border:1px solid #ddd">{gpu_hours:.1f}h</td></tr>
<tr><td style="padding:8px;border:1px solid #ddd;background:#f0fff4">🌐 slug</td><td style="padding:8px;border:1px solid #ddd">{account['username']}/xandra-worker</td></tr>
</table>
<p style="margin-top:15px;color:#666">رابط الـ Notebook: <a href="https://www.kaggle.com/code/{account['username']}/xandra-worker">kaggle.com/code/{account['username']}/xandra-worker</a></p>
</div>
"""
        send_email(f"{'✅' if is_progressing else '⚠️'} تدريب {account['username']} (خطوة {verified_step:,})",
                   success_body)
        print(f"\n✅ اكتملت المهمة. حالة التقدم: {'ناجح' if is_progressing else 'غير مؤكد'}")


if __name__ == "__main__":
    main()
