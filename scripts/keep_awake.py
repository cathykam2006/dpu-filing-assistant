"""
Keep the Streamlit Community Cloud app awake.

Streamlit Cloud puts an app to sleep after a period with no visitors. A plain
HTTP request does not count as a visit, so this opens the app in a headless
browser, clicks "Yes, get this app back up!" if the app is asleep, and waits
for the app to load.

Run by .github/workflows/keep-awake.yml on a schedule.
Usage: python scripts/keep_awake.py [URL]
"""

import os
import sys

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

URL = (sys.argv[1] if len(sys.argv) > 1 else
       os.environ.get("APP_URL",
                      "https://dpu-filing-assistant-6tsrcudhmux8mpakqcnlux.streamlit.app/"))
WAKE_TEXT = "get this app back up"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        print(f"Opening {URL}")
        page.goto(URL, wait_until="domcontentloaded", timeout=90_000)
        page.wait_for_timeout(8_000)

        wake = page.locator(f"button:has-text('{WAKE_TEXT}')")
        if wake.count() == 0:
            wake = page.get_by_text("Yes, get this app back up", exact=False)

        if wake.count() > 0:
            print("App was asleep — clicking the wake-up button")
            wake.first.click()
            page.wait_for_timeout(5_000)
        else:
            print("App was awake")

        # wait for the Streamlit app itself to render (it runs in an iframe)
        loaded = False
        for _ in range(12):                       # up to ~3 minutes
            for frame in page.frames:
                try:
                    if frame.locator("[data-testid='stApp']").count() > 0:
                        loaded = True
                        break
                except Exception:  # noqa: BLE001 - frame detached while booting
                    pass
            if loaded:
                break
            page.wait_for_timeout(15_000)

        print("App loaded" if loaded else "App did not finish loading within 3 minutes")
        browser.close()
        # Fail the run only if the app stayed asleep, so a red run in the
        # Actions tab means something needs attention.
        if not loaded and page_is_asleep_text(URL):
            sys.exit(1)


def page_is_asleep_text(url):
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(url, wait_until="domcontentloaded", timeout=90_000)
        pg.wait_for_timeout(5_000)
        asleep = WAKE_TEXT in pg.content().lower()
        b.close()
        return asleep


if __name__ == "__main__":
    try:
        main()
    except PWTimeout as e:
        print(f"Timed out: {e}")
        sys.exit(1)
