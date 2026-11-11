"""Screenshot of the running Streamlit dashboard. Usage: python scripts/screenshot_dashboard.py <out.png>"""
import sys
from playwright.sync_api import sync_playwright

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 1100})
    page.goto("http://localhost:8501")
    page.wait_for_timeout(9000)
    page.screenshot(path=sys.argv[1], full_page=True)
    browser.close()
