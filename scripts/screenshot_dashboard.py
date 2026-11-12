"""Screenshot of the running Streamlit dashboard. Usage: python scripts/screenshot_dashboard.py <out.png> [tab name]"""
import sys
from playwright.sync_api import sync_playwright

out, tab = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 1200})
    page.goto("http://localhost:8501")
    page.wait_for_timeout(9000)
    if tab:
        page.get_by_text(tab).first.click()
        page.wait_for_timeout(5000)
    page.screenshot(path=out, full_page=True)
    browser.close()
