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

KAGGLE_ACCOUNTS = []
for i in range(1, 7):
    username = os.environ.get(f"KAGGLE_USERNAME_{i}" if i > 1 else "KAGGLE_USERNAME")
    key = os.environ.get(f"KAGGLE_KEY_{i}" if i > 1 else "KAGGLE_KEY")
    if username and key:
        KAGGLE_ACCOUNTS.append({"username": username, "key": key, "index": i, "status": "active"})

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
    """Returns a writable directory for kaggle.json (works on GitHub Actions, Kaggle, local)."""
    # GitHub Actions runners are user 'runner' (not root) — /root/.kaggle is forbidden
    # Use $HOME/.kaggle if HOME is set, else fall back to ~/.kaggle via os.path.expanduser
    # If even that fails, use a temp dir (kaggle CLI also reads KAGGLE_CONFIG_DIR env var)
    home = os.environ.get('HOME') or os.path.expanduser('~')
    kaggle_dir = os.path.join(home, '.kaggle')
    try:
        os.makedirs(kaggle_dir, exist_ok=True)
        # Test write access
        test_file = os.path.join(kaggle_dir, '.write_test')
        with open(test_file, 'w') as f: f.write('ok')
        os.remove(test_file)
        return kaggle_dir
    except (PermissionError, OSError):
        # Fall back to a temp dir
        import tempfile
        kaggle_dir = os.path.join(tempfile.gettempdir(), 'kaggle_config')
        os.makedirs(kaggle_dir, exist_ok=True)
        os.environ['KAGGLE_CONFIG_DIR'] = kaggle_dir
        return kaggle_dir


def setup_kaggle_credentials(account):
    """
    Configures Kaggle credentials for a specific account using THREE methods
    simultaneously to maximize compatibility:
      1. Sets KAGGLE_USERNAME and KAGGLE_KEY env vars (works in all CLIs)
      2. Writes ~/.kaggle/kaggle.json (the official way)
      3. Sets KAGGLE_CONFIG_DIR env var if HOME is not writable
    Does NOT install kaggle CLI (that's ensure_kaggle_cli's job, called separately).
    """
    try:
        username = account["username"]
        key = account["key"]

        # Method 1: env vars (most reliable — kaggle CLI checks these first)
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key

        # Method 2: write kaggle.json to a writable location
        kaggle_dir = get_kaggle_config_dir()
        kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')
        with open(kaggle_json_path, 'w') as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(kaggle_json_path, 0o600)

        return True
    except Exception as e:
        print(f"⚠️ فشل إعداد بيانات اعتماد Kaggle لـ {account['username']}: {e}")
        # Last resort: at least set env vars
        try:
            os.environ['KAGGLE_USERNAME'] = account["username"]
            os.environ['KAGGLE_KEY'] = account["key"]
            return True
        except: return False


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
    # The canonical slug verified from real Kaggle URLs:
    # kaggle.com/code/{username}/xandra-worker
    # We try the canonical one first, then alternates as fallback
    possible_slugs = [
        f"{username}/xandra-worker",      # ✅ CANONICAL — verified from actual URLs
        f"{username}/xandraworker",       # alternate (no separator)
        f"{username}/xandra_worker"       # alternate (underscore)
    ]

    for slug in possible_slugs:
        try:
            result = subprocess.run(
                ["kaggle", "kernels", "status", slug],
                capture_output=True, text=True, timeout=30
            )
            out = (result.stdout + " " + result.stderr).lower()

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
                # If stdout is empty, treat as unknown (don't assume error)
                if not result.stdout.strip():
                    return "unknown"
                # Otherwise log the raw output for debugging
                print(f"   📝 raw output for {slug}: {result.stdout[:200]}")
                return "unknown"
            else:
                # Non-zero exit + non-404 error → auth/network issue
                if "unauthorized" in out or "401" in out:
                    print(f"   ⚠️ 401 Unauthorized for {username}")
                    return "unknown"
                # For other errors, try next slug
                continue
        except subprocess.TimeoutExpired:
            print(f"⏱️ انتهت مهلة الاستعلام عن {slug}")
            return "unknown"
        except Exception as e:
            print(f"⚠️ خطأ في فحص حالة الـ kernel لـ {username}: {e}")
            return "unknown"

    # All slugs returned "not found" — this is a fresh account (no kernel yet)
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

def update_kernel_status_map(state):
    """
    Queries all Kaggle accounts for their kernel status and stores the result
    in state.kernel_status_map. This is what the Cloudflare Dashboard reads.
    """
    kernel_map = {}
    for acc in KAGGLE_ACCOUNTS:
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
        for session in state.get("history", []):
            if username in session.get("worker_id", ""):
                total_sessions += 1
                try:
                    ts = session["timestamp"]
                    if ts.endswith("Z"): ts = ts[:-1] + "+00:00"
                    session_time = datetime.fromisoformat(ts)
                    if session_time > week_start:
                        weekly_hours += session.get("gpu_hours", 0)
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
            "kernel_status": kernel_status,
            "excluded_reason": excluded_reason
        })

    # Pick the best: not excluded, has the most remaining hours
    candidates = [a for a in account_stats if a["excluded_reason"] is None]
    candidates.sort(key=lambda x: x["remaining"], reverse=True)

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
    Fixes the known issue where `pip install kaggle` doesn't install `six`
    on Python 3.11+. We install `six` explicitly first, then kaggle.
    """
    try:
        # Test if kaggle CLI actually works (not just exists)
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception:
        pass

    # Install six first (the missing dependency), then kaggle
    print("📦 تثبيت kaggle CLI + six...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "six"],
                       check=True, capture_output=True, timeout=60)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"],
                       check=True, capture_output=True, timeout=120)
        # Verify
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            print("✅ kaggle CLI مثبّت بنجاح")
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
        metadata = {
            "id": f"{username}/xandra-worker",
            "title": "xandra worker",
            "code_file": "xandra_worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": []
        }
        with open("kernel-metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Push using the official CLI
        # Env vars KAGGLE_USERNAME + KAGGLE_KEY are set by setup_kaggle_credentials
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", "."],
            capture_output=True, text=True, timeout=120
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
