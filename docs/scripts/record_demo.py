"""
Record a short walkthrough of the Ridge playground (website/app/) as a video.

Usage:
    pip install playwright
    playwright install chromium
    python website/scripts/record_demo.py

Output:
    website/demo_recording.webm  (raw Playwright capture)
    website/demo_recording.mp4   (re-encoded via ffmpeg, if available)

Notes:
- Spins up `python -m http.server` on a free port, points Chromium at /app/,
  walks through the controls, then shuts the server down.
- Total runtime targets ~45s of footage.
"""

from __future__ import annotations

import contextlib
import http.server
import shutil
import socket
import socketserver
import subprocess
import threading
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

WEBSITE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = WEBSITE_DIR / "_recording"
WEBM_OUT = WEBSITE_DIR / "demo_recording.webm"
MP4_OUT = WEBSITE_DIR / "demo_recording.mp4"

VIDEO_SIZE = {"width": 1440, "height": 900}


def _pick_ffmpeg() -> str | None:
    """Return an ffmpeg binary that ships with libx264, or None."""
    candidates = ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", shutil.which("ffmpeg")]
    for cand in candidates:
        if not cand:
            continue
        try:
            out = subprocess.run(
                [cand, "-hide_banner", "-encoders"],
                capture_output=True, text=True, check=False,
            ).stdout
        except OSError:
            continue
        if "libx264 " in out:
            return cand
    return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def serve(directory: Path):
    port = _free_port()
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(
        *a, directory=str(directory), **kw
    )
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def wait_for_demo_ready(page: Page) -> None:
    page.wait_for_function(
        "() => document.querySelector('#demo-plot .plotly') !== null",
        timeout=60_000,
    )
    page.wait_for_function(
        "() => { const t = document.getElementById('m-mse'); return t && t.textContent && t.textContent.trim() !== '—'; }",
        timeout=120_000,
    )


def set_slider(page: Page, slider_id: str, value: float, *, fire_change: bool = True) -> None:
    """Set a range input's value and dispatch input/change so the demo refits."""
    page.evaluate(
        """({id, value, fireChange}) => {
            const el = document.getElementById(id);
            if (!el) return;
            el.value = String(value);
            el.dispatchEvent(new Event('input', {bubbles: true}));
            if (fireChange) el.dispatchEvent(new Event('change', {bubbles: true}));
        }""",
        {"id": slider_id, "value": value, "fireChange": fire_change},
    )


def sweep_slider(page: Page, slider_id: str, values, *, dwell_ms: int = 350) -> None:
    *intermediate, final = list(values)
    for v in intermediate:
        set_slider(page, slider_id, v, fire_change=False)
        page.wait_for_timeout(dwell_ms)
    set_slider(page, slider_id, final, fire_change=True)
    page.wait_for_timeout(dwell_ms)


def select_option(page: Page, select_id: str, value: str) -> None:
    page.select_option(f"#{select_id}", value)


def wait_for_fit(page: Page, timeout_ms: int = 15_000) -> None:
    """Wait for the metrics panel to leave the 'working' state."""
    deadline = time.time() + timeout_ms / 1000.0
    while time.time() < deadline:
        status = page.text_content("#m-status") or ""
        if "working" not in status.lower() and "queued" not in status.lower():
            return
        page.wait_for_timeout(150)


def run_walkthrough(page: Page) -> None:
    # 0:00 - hold on the landing state so the viewer can read the metrics
    page.wait_for_timeout(3000)

    # 0:03 - hop through horizons; autotune chips snap to each searched optimum
    sweep_slider(page, "horizon-slider", [96, 168, 240, 336, 480, 600, 720], dwell_ms=900)
    wait_for_fit(page)
    page.wait_for_timeout(1200)

    # 0:13 - back to a short horizon and let autotune re-snap
    sweep_slider(page, "horizon-slider", [720, 576, 432, 288, 192], dwell_ms=800)
    wait_for_fit(page)
    page.wait_for_timeout(1000)

    # 0:21 - turn autotune OFF
    page.click("#autotune-btn")
    page.wait_for_timeout(900)

    # 0:22 - drag the lookback slider; refits on each release
    sweep_slider(
        page,
        "lookback-slider",
        [96, 192, 320, 448, 640, 832, 1024, 1280, 1024, 768, 512, 320, 192],
        dwell_ms=700,
    )
    wait_for_fit(page)
    page.wait_for_timeout(1000)

    # 0:32 - drag the regularization slider across log-α
    sweep_slider(
        page,
        "alpha-slider",
        [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 4, 2, 0, -1],
        dwell_ms=600,
    )
    wait_for_fit(page)
    page.wait_for_timeout(1200)

    # 0:42 - autotune back ON
    page.click("#autotune-btn")
    wait_for_fit(page)
    page.wait_for_timeout(1500)

    # 0:44 - switch dataset (ETTh1 -> another bundled one)
    datasets = page.evaluate(
        "() => Array.from(document.querySelectorAll('#dataset-select option')).map(o => o.value)"
    )
    other = next((d for d in datasets if d.lower() != "etth1"), None)
    if other:
        select_option(page, "dataset-select", other)
        wait_for_fit(page, timeout_ms=30_000)
        page.wait_for_timeout(1800)

    # 0:50 - cycle through a couple of series in the new dataset
    series = page.evaluate(
        "() => Array.from(document.querySelectorAll('#series-select option')).map(o => o.value)"
    )
    for s in series[:3]:
        select_option(page, "series-select", s)
        wait_for_fit(page)
        page.wait_for_timeout(1500)

    # 0:55 - end on a final live lookback nudge
    page.click("#autotune-btn")
    page.wait_for_timeout(500)
    sweep_slider(page, "lookback-slider", [128, 256, 384, 256], dwell_ms=700)
    wait_for_fit(page)
    page.wait_for_timeout(2000)


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with serve(WEBSITE_DIR) as base_url:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(
                viewport=VIDEO_SIZE,
                record_video_dir=str(OUT_DIR),
                record_video_size=VIDEO_SIZE,
                device_scale_factor=1,
            )
            page = context.new_page()
            page.goto(f"{base_url}/app/")
            wait_for_demo_ready(page)
            run_walkthrough(page)
            video = page.video
            context.close()
            browser.close()
            raw_path = Path(video.path()) if video else None

    if raw_path and raw_path.exists():
        if WEBM_OUT.exists():
            WEBM_OUT.unlink()
        shutil.copy(raw_path, WEBM_OUT)
        print(f"Saved {WEBM_OUT.relative_to(WEBSITE_DIR.parent)}")

        ffmpeg_bin = _pick_ffmpeg()
        if ffmpeg_bin:
            cmd = [
                ffmpeg_bin, "-y", "-loglevel", "error",
                "-i", str(WEBM_OUT),
                "-c:v", "libx264", "-crf", "20", "-preset", "medium",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(MP4_OUT),
            ]
            try:
                subprocess.run(cmd, check=True)
                print(f"Saved {MP4_OUT.relative_to(WEBSITE_DIR.parent)}")
            except subprocess.CalledProcessError as e:
                print(f"ffmpeg re-encode failed ({e}); webm is still usable.")
        else:
            print("ffmpeg with libx264 not found; skipping mp4 re-encode.")
    else:
        raise SystemExit("Playwright did not produce a video file.")

    shutil.rmtree(OUT_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
