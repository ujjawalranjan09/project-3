import asyncio
from playwright.async_api import async_playwright
import os
import subprocess
import time

async def run():
    # Start Streamlit in background
    streamlit_proc = subprocess.Popen(
        ["streamlit", "run", "railway-traffic-control-system/frontend/app.py", "--server.port", "8501", "--server.headless", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    time.sleep(10) # Wait for streamlit to start

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        try:
            await page.goto("http://localhost:8501")
            await page.wait_for_timeout(5000)

            # Take a screenshot of the streamlit dashboard
            await page.screenshot(path="/home/jules/verification/streamlit_dashboard.png")
            print("Streamlit screenshot saved.")

        except Exception as e:
            print(f"Error during playwright: {e}")
        finally:
            await browser.close()
            streamlit_proc.terminate()

if __name__ == "__main__":
    asyncio.run(run())
