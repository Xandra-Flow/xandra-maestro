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
MIN_HOURS_BETWEEN_SESSIONS = 0   # 0 because we now rely on real-time kernel status


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


def verify_kaggle_auth(account):
    """
    Verifies that kaggle CLI can actually authenticate with the given credentials.
    Calls `kaggle competitions list` which requires auth — if it returns 401,
    we know the credentials are wrong BEFORE we try to push.

    Returns True if auth is working, False otherwise.
    """
    username = account["username"]
    subprocess_env = os.environ.copy()
    subprocess_env['HOME'] = os.environ.get('HOME', os.path.expanduser('~'))

    # Pass the right env vars based on auth method
    if account.get("auth_method") == "api_token":
        subprocess_env['KAGGLE_API_TOKEN'] = account.get("api_token", "")
        subprocess_env.pop('KAGGLE_USERNAME', None)
        subprocess_env.pop('KAGGLE_KEY', None)
    else:
        subprocess_env['KAGGLE_USERNAME'] = username
        subprocess_env['KAGGLE_KEY'] = account.get("key", "")
        subprocess_env.pop('KAGGLE_API_TOKEN', None)

    if 'KAGGLE_CONFIG_DIR' in os.environ:
        subprocess_env['KAGGLE_CONFIG_DIR'] = os.environ['KAGGLE_CONFIG_DIR']

    try:
        # Use `kaggle competitions list` — requires auth, returns list or 401
        result = subprocess.run(
            ["kaggle", "competitions", "list"],
            capture_output=True, text=True, timeout=20,
            env=subprocess_env
        )
        out = (result.stdout + " " + result.stderr).lower()
        if "401" in out or "unauthorized" in out or "authentication" in out:
            print(f"   ❌ فشل المصادقة لـ {username} — تحقق من الـ token/key في GitHub Secrets")
            return False
        if result.returncode == 0:
            return True
        return False
    except Exception as e:
        print(f"   ⚠️ تعذّر التحقق من auth لـ {username}: {e}")
        return False


def check_kaggle_kernel_status(account):
    """
    Queries Kaggle for the real status of xandra-worker notebook via the kaggle CLI.

    KERNEL SLUG DISCOVERY (verified from actual Kaggle URLs):
    Kaggle URL format: https://www.kaggle.com/code/{username}/{slug}
    The user's actual notebooks live at: kaggle.com/code/{username}/xandra-worker
    So the canonical slug is "xandra-worker" (with hyphen, NOT underscore).

    Returns one of:
        'running'   → notebook is actively training
        'complete'  → notebook finished successfully
        'error'     → notebook crashed or errored out
        'canceled'  → notebook was canceled manually
        'not_found' → no notebook exists yet (fresh account)
        'unknown'   → could not determine (network/auth error)
    """
    # Ensure CLI is available (installs if missing — also handles 'six' dep)
    if not ensure_kaggle_cli():
        return "unknown"

    if not setup_kaggle_credentials(account):
        return "unknown"

    username = account["username"]
    # The canonical slug verified from real Kaggle URLs
    possible_slugs = [
        f"{username}/xandra-worker",      # ✅ CANONICAL — verified from actual URLs
        f"{username}/xandraworker",       # alternate (no separator)
        f"{username}/xandra_worker"       # alternate (underscore)
    ]

    auth_failed_count = 0

    # Build the env dict to pass to subprocess — pass the right env vars
    # based on the account's auth method
    subprocess_env = os.environ.copy()
    subprocess_env['HOME'] = os.environ.get('HOME', os.path.expanduser('~'))

    if account.get("auth_method") == "api_token":
        # New API Token mode (KGAT_)
        subprocess_env['KAGGLE_API_TOKEN'] = account.get("api_token", "")
        subprocess_env.pop('KAGGLE_USERNAME', None)
        subprocess_env.pop('KAGGLE_KEY', None)
    else:
        # Legacy mode
        subprocess_env['KAGGLE_USERNAME'] = username
        subprocess_env['KAGGLE_KEY'] = account.get("key", "")
        subprocess_env.pop('KAGGLE_API_TOKEN', None)

    # KAGGLE_CONFIG_DIR — if set, kaggle CLI reads kaggle.json/access_token from there
    if 'KAGGLE_CONFIG_DIR' in os.environ:
        subprocess_env['KAGGLE_CONFIG_DIR'] = os.environ['KAGGLE_CONFIG_DIR']

    for slug in possible_slugs:
        try:
            result = subprocess.run(
                ["kaggle", "kernels", "status", slug],
                capture_output=True, text=True, timeout=30,
                env=subprocess_env
            )
            out = (result.stdout + " " + result.stderr).lower()

            # ⚠️ Detect auth failures FIRST
            if ("authentication required" in out or
                "you will need a kaggle account" in out or
                "kaggle auth login" in out or
                "401" in out or
                "unauthorized" in out):
                print(f"   ⚠️ فشل المصادقة لـ {username} — تحقق من KAGGLE_KEY")
                auth_failed_count += 1
                continue

            # If kernel not found, try the next slug
            if "no kernels found" in out or "not found" in out or "404" in out:
                continue

            # Found a real status
            if "running" in out or "in queue" in out:
                return "running"
            if "complete" in out:
                return "complete"
            if "error" in out or "failed" in out:
                return "error"
            if "cancel" in out:
                return "canceled"

            # CLI returned something but ambiguous — check return code
            if result.returncode == 0:
                if not result.stdout.strip():
                    return "unknown"
                print(f"   📝 raw output for {slug}: {result.stdout[:200]}")
                return "unknown"
            else:
                continue
        except subprocess.TimeoutExpired:
            print(f"⏱️ انتهت مهلة الاستعلام عن {slug}")
            return "unknown"
        except Exception as e:
            print(f"⚠️ خطأ في فحص حالة الـ kernel لـ {username}: {e}")
            return "unknown"

    # If auth failed for any slug, return "unknown" (NOT "not_found")
    if auth_failed_count > 0:
        print(f"   ⚠️ فشل المصادقة على {auth_failed_count} محاولة لـ {username} — إرجاع unknown")
        return "unknown"

    # All slugs returned "not found" — this is genuinely a fresh account (no kernel)
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
            # else: not_found / unknown → fall through to time-based check

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
    candidates = [a for a in account_stats if a["excluded_reason"] is None]
    # Sort by: most remaining DESC, then oldest last_session ASC, then fewest sessions ASC
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


def push_kaggle_kernel_via_api(account, worker_code_path):
    """
    Pushes the kernel via the kaggle CLI (the official method).

    KERNEL SLUG RULES (verified from actual Kaggle URLs):
    - Kaggle URL format: kaggle.com/code/{username}/{slug}
    - Slug is auto-generated from title by:
        * Converting spaces to hyphens
        * Converting underscores to hyphens
        * Lowercasing
        * Removing other special chars
    - User's existing notebooks live at: kaggle.com/code/{username}/xandra-worker
    - So title must be "xandra worker" (with space) or "xandra-worker" (with hyphen)
      → slug becomes "xandra-worker"
    - And id must be "{username}/xandra-worker" (matching the slug)

    Returns True on success.
    """
    if not ensure_kaggle_cli():
        return False

    if not setup_kaggle_credentials(account):
        return False

    username = account["username"]
    try:
        # Inject the real username into the worker code before pushing
        with open(worker_code_path, "r") as f:
            worker_code = f.read()
        worker_code = worker_code.replace('__KAGGLE_USERNAME__', username)
        with open(worker_code_path, "w") as f:
            f.write(worker_code)

        # Create kernel-metadata.json (required by kaggle kernels push)
        # Use title="xandra worker" → kaggle converts to slug "xandra-worker"
        # which matches the canonical URL kaggle.com/code/{username}/xandra-worker
        #
        # ⚠️ IMPORTANT: is_private/enable_gpu/enable_internet must be boolean true,
        # NOT string "true" — kaggle CLI 1.6.17 rejects string values
        # (per https://github.com/Kaggle/kaggle-cli/issues/821)
        metadata = {
            "id": f"{username}/xandra-worker",
            "title": "xandra worker",
            "code_file": "xandra_worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": []
        }
        with open("kernel-metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Push using the official CLI
        # Pass explicit env vars based on the account's auth method
        subprocess_env = os.environ.copy()
        subprocess_env['HOME'] = os.environ.get('HOME', os.path.expanduser('~'))

        if account.get("auth_method") == "api_token":
            subprocess_env['KAGGLE_API_TOKEN'] = account.get("api_token", "")
            subprocess_env.pop('KAGGLE_USERNAME', None)
            subprocess_env.pop('KAGGLE_KEY', None)
        else:
            subprocess_env['KAGGLE_USERNAME'] = username
            subprocess_env['KAGGLE_KEY'] = account.get("key", "")
            subprocess_env.pop('KAGGLE_API_TOKEN', None)

        if 'KAGGLE_CONFIG_DIR' in os.environ:
            subprocess_env['KAGGLE_CONFIG_DIR'] = os.environ['KAGGLE_CONFIG_DIR']

        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", "."],
            capture_output=True, text=True, timeout=120,
            env=subprocess_env
        )

        if result.returncode == 0:
            print(f"✅ تم رفع الـ Notebook على {username} (slug: xandra-worker)")
            return True
        else:
            stderr = result.stderr or ""
            stdout = result.stdout or ""
            print(f"❌ فشل الرفع عبر CLI (exit {result.returncode})")
            print(f"   stdout: {stdout[:400]}")
            print(f"   stderr: {stderr[:400]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏱️ انتهت مهلة رفع الـ Notebook على {username}")
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
        return push_kaggle_kernel_via_api(account, "xandra_worker.py")
    except Exception as e:
        print(f"❌ استثناء أثناء الإطلاق: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# 🎼 Main Orchestration Loop
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("🎼 Xandra Maestro (Real-time Discovery Edition)")
    print(f"⏰ {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    state = get_state()

    # Step 1: refresh kernel status map (this is the key upgrade)
    print("\n📡 تحديث خريطة حالة الـ kernels من Kaggle...")
    state = update_kernel_status_map(state)

    # Step 2: compute session gaps for the dashboard
    compute_session_gaps(state)

    # Step 3: report account statuses
    print("\n📊 تقرير الحسابات:")
    account, remaining_hours, account_stats = select_available_account(state)
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
        print(f"\n✅ تم بنجاح. المايسترو سينام حتى الجولة القادمة.")


if __name__ == "__main__":
    main()
