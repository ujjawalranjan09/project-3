import asyncio
from playwright.async_api import async_playwright
import os
import subprocess
import time

async def run():
    # Start Streamlit in background
    streamlit_proc = subprocess.Popen(
        ["streamlit", "run", "railway-traffic-control-system/frontend/app.py", "--server.port", "8502", "--server.headless", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    time.sleep(15) # Wait for streamlit to start

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        try:
            await page.goto("http://localhost:8502")
            # Wait for KPIs to load
            await page.wait_for_timeout(5000)

            # Take a screenshot of the streamlit dashboard with data
            await page.screenshot(path="/home/jules/verification/streamlit_dashboard_with_data.png")

            # Click Analyze Conflict Risk
            await page.click("button:has-text('Analyze Conflict Risk')")
            await page.wait_for_timeout(3000)
            await page.screenshot(path="/home/jules/verification/streamlit_conflict_result.png")

            print("Streamlit full verification screenshots saved.")

        except Exception as e:
            print(f"Error during playwright: {e}")
        finally:
            await browser.close()
            streamlit_proc.terminate()

if __name__ == "__main__":
    asyncio.run(run())
