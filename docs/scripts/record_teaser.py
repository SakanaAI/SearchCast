"""
Render the project-page hero figure (the "lookback-span" Widget A) to a looping
animated GIF for the top of the repository README.

The hero is a JS/SVG widget on docs/index.html that sweeps the forecast horizon
H from 24 to 720 and shows, per dataset, the searched optimal lookback L*(H) as a
colored bar (turbo colormap). This script drives that widget deterministically:
it sets the widget's range slider to each H value (which the widget renders
synchronously), screenshots the SVG, and stitches the frames into a ping-pong GIF.

Usage:
    pip install playwright Pillow
    playwright install chromium
    python docs/scripts/record_teaser.py

Output:
    assets/teaser.gif   (repository root)

Notes:
- Spins up `python -m http.server` on a free port serving docs/, points Chromium
  at /index.html, captures the hero, then shuts the server down.
- Capture is deterministic (no reliance on the widget's autoplay timer): setting
  the slider value + dispatching an "input" event triggers the widget's own
  pause()+draw(), so each screenshot is an exact frame for that H.
"""

from __future__ import annotations

import contextlib
import http.server
import socket
import socketserver
import tempfile
import threading
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

DOCS_DIR = Path(__file__).resolve().parent.parent          # repo/docs
REPO_ROOT = DOCS_DIR.parent                                 # repo
GIF_OUT = REPO_ROOT / "assets" / "teaser.gif"

# Hero widget selectors (docs/scripts/widget_a.js).
HERO = ".la-widget.hero"
HERO_SVG = f"{HERO} svg.la-plot"
HERO_SLIDER = f"{HERO} input.la-slider"

# Horizon sweep — matches the widget (min=24, max=720, step=24).
H_MIN, H_MAX, H_STEP = 24, 720, 24

# Render the page wide enough that the 2x4 hero grid lays out at full width;
# device_scale_factor=2 keeps the SVG crisp.
VIEWPORT = {"width": 1040, "height": 480}
DEVICE_SCALE_FACTOR = 2

# GIF tuning.
TARGET_WIDTH = 900          # downscale the (retina) screenshots to this width
FRAME_MS = 180              # per-frame duration in the GIF


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


def _set_horizon(page: Page, h: int) -> None:
    """Set the hero slider to H and let the widget render that frame."""
    page.eval_on_selector(
        HERO_SLIDER,
        """(el, value) => {
            el.value = String(value);
            el.dispatchEvent(new Event('input', { bubbles: true }));
        }""",
        h,
    )
    page.wait_for_timeout(60)  # let the synchronous draw settle / fonts paint


def capture_frames(page: Page, frame_dir: Path) -> list[Path]:
    page.wait_for_selector(HERO_SVG, timeout=60_000)
    # Wait until the grid has actually rendered panels (text labels present).
    page.wait_for_function(
        "(sel) => document.querySelector(sel) && "
        "document.querySelector(sel).querySelectorAll('text').length > 0",
        arg=HERO_SVG,
        timeout=60_000,
    )

    svg = page.locator(HERO_SVG)
    paths: list[Path] = []
    for i, h in enumerate(range(H_MIN, H_MAX + 1, H_STEP)):
        _set_horizon(page, h)
        out = frame_dir / f"frame_{i:03d}.png"
        svg.screenshot(path=str(out))
        paths.append(out)
        print(f"  captured H={h:>3}  -> {out.name}")
    return paths


def encode_gif(frame_paths: list[Path], out_path: Path) -> None:
    frames = [Image.open(p).convert("RGB") for p in frame_paths]

    # Downscale to a README-friendly width (preserve aspect ratio).
    w0, h0 = frames[0].size
    if w0 > TARGET_WIDTH:
        new_h = round(h0 * TARGET_WIDTH / w0)
        frames = [f.resize((TARGET_WIDTH, new_h), Image.LANCZOS) for f in frames]

    # Ping-pong: forward then back (drop the shared endpoints) so the loop sweeps
    # H up and down like the live widget.
    sequence = frames + frames[-2:0:-1]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sequence[0].save(
        out_path,
        save_all=True,
        append_images=sequence[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="teaser_frames_") as tmp:
        frame_dir = Path(tmp)
        with serve(DOCS_DIR) as base_url, sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(
                viewport=VIEWPORT, device_scale_factor=DEVICE_SCALE_FACTOR
            )
            print(f"Loading {base_url}/index.html ...")
            page.goto(f"{base_url}/index.html", wait_until="networkidle")
            print("Capturing hero frames ...")
            frame_paths = capture_frames(page, frame_dir)
            browser.close()

        print(f"Encoding {len(frame_paths)} frames -> {GIF_OUT} ...")
        encode_gif(frame_paths, GIF_OUT)

    size_mb = GIF_OUT.stat().st_size / 1e6
    print(f"Done: {GIF_OUT} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
