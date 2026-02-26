import json
import os
import sys
import time
from datetime import datetime

CONFIRM_DIR = "output/log"
PENDING_FILE = os.path.join(CONFIRM_DIR, "translation_confirm.pending")
APPROVED_FILE = os.path.join(CONFIRM_DIR, "translation_confirm.approved")


def _ensure_dir():
    os.makedirs(CONFIRM_DIR, exist_ok=True)


def set_translation_confirmation_pending():
    _ensure_dir()
    if os.path.exists(APPROVED_FILE):
        os.remove(APPROVED_FILE)
    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        f.write(datetime.now().isoformat())


def approve_translation_confirmation(source: str):
    _ensure_dir()
    if not os.path.exists(PENDING_FILE):
        return
    with open(APPROVED_FILE, "w", encoding="utf-8") as f:
        json.dump({"source": source, "time": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)


def is_translation_confirmation_pending():
    return os.path.exists(PENDING_FILE)


def is_translation_confirmed():
    return os.path.exists(APPROVED_FILE)


def clear_translation_confirmation():
    for p in (PENDING_FILE, APPROVED_FILE):
        if os.path.exists(p):
            os.remove(p)


def consume_translation_confirmation():
    if not is_translation_confirmed():
        return False
    clear_translation_confirmation()
    return True


def wait_for_translation_confirmation_cli(prompt: str):
    set_translation_confirmation_pending()
    if sys.stdin.isatty():
        input(prompt)
        approve_translation_confirmation("terminal")
        clear_translation_confirmation()
        return

    print("PAUSE_BEFORE_TRANSLATE is waiting for external confirmation...")
    print(f"Create `{APPROVED_FILE}` to continue.")
    while not is_translation_confirmed():
        time.sleep(1)
    clear_translation_confirmation()
