"""
Keep the Streamlit Community Cloud app awake.

Streamlit Cloud puts an app to sleep after a period with no visitors. A plain
HTTP request does not count as a visit, so this opens the app in a headless
browser, clicks "Yes, get this app back up!" if the app is asleep, and waits
(up to about 6 minutes) for the app to finish starting.

If the app never loads, the run fails and the log says what the page showed
(the sleep screen, a startup message, an error, a sign-in page, ...) and a
screenshot is saved as keep_awake.png.

Usage: python scripts/keep_awake.py [URL]   (or set APP_URL)
"""

import os
import sys
import time

from playwright.sync_api import sync_playwright

URL = (sys.argv[1] if len(sys.argv) > 1 else
       os.environ.get("APP_URL",
                      "https://dpu-filing-assistant-6tsrcudhmux8mpakqcnlux.streamlit.app/"))
WAKE_TEXT = "get this app back up"
WAIT_SECONDS = int(os.environ.get("WAIT_SECONDS", 360))          # how long to wait for the app to start
CHECK_EVERY = 10


def app_loaded(page):
    """True once the Streamlit app itself has rendered (usually in an iframe)."""
    for frame in page.frames:
        try:
            if frame.locator("[data-testid='stApp'], [data-testid='stAppViewContainer']").count():
                return True
        except Exception:  # noqa: BLE001 - frame detached while the app boots
            pass
    return False


def visible_text(page, limit=600):
    parts = []
    for frame in page.frames:
        try:
            t = frame.locator("body").inner_text(timeout=3_000).strip()
        except Exception:  # noqa: BLE001
            continue
        if t:
            parts.append(" ".join(t.split()))
    return " | ".join(parts)[:limit] or "(blank page)"


def main():
    with sync_playwright() as p:
        # Look like a regular desktop Chrome: Streamlit Cloud may refuse to
        # serve its app code to an obviously headless browser.
        browser = p.chromium.launch(args=["--disable-blink-features=AutomationControlled"])
        context = browser.new_context(
            viewport={"width": 1280, "height": 900}, locale="en-US",
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"))
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        page = context.new_page()
        problems = []
        page.on("requestfailed", lambda r: problems.append(
            f"failed: {r.url[:120]} ({r.failure})"))
        page.on("response", lambda r: problems.append(f"HTTP {r.status}: {r.url[:120]}")
                if r.status >= 400 else None)
        page.on("console", lambda m: problems.append(f"console {m.type}: {m.text[:200]}")
                if m.type == "error" else None)
        print(f"Opening {URL}")
        resp = page.goto(URL, wait_until="domcontentloaded", timeout=90_000)
        print(f"HTTP status: {resp.status if resp else 'none'}")

        clicks = 0
        start = time.time()
        loaded = False
        while time.time() - start < WAIT_SECONDS:
            if app_loaded(page):
                loaded = True
                break
            wake = page.locator(f"button:has-text('{WAKE_TEXT}')")
            if wake.count() and clicks < 3:
                clicks += 1
                print(f"App is asleep — clicking the wake-up button (try {clicks})")
                try:
                    wake.first.click(timeout=10_000)
                except Exception as e:  # noqa: BLE001
                    print(f"  click failed: {e}")
            page.wait_for_timeout(CHECK_EVERY * 1_000)

        took = int(time.time() - start)
        if loaded:
            print(("App woke up and loaded" if clicks else "App was awake and loaded")
                  + f" ({took}s)")
            page.wait_for_timeout(15_000)          # stay briefly so it counts as a visit
        else:
            print(f"App did NOT load within {WAIT_SECONDS} seconds.")
            print(f"Final URL: {page.url}")
            print(f"Page title: {page.title()!r}")
            print(f"Page shows: {visible_text(page)}")
            for line in problems[:15]:
                print(f"  {line}")
        try:
            page.screenshot(path="keep_awake.png", full_page=True)
        except Exception:  # noqa: BLE001
            pass
        browser.close()
        return 0 if loaded else 1


if __name__ == "__main__":
    sys.exit(main())
