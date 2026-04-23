"""Entry point that launches the Streamlit UI.

Usage:
    python run_app.py

This is a thin wrapper around ``streamlit run app/main.py`` so that users can
simply double-click or run a single Python command to start the demo.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    app_path = repo_root / "app" / "main.py"

    if not app_path.exists():
        print(f"[GlitchVision] Could not find Streamlit app at: {app_path}")
        return 1

    # Make ``src`` importable when Streamlit spawns its own process.
    os.environ["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    )

    # Defer import so a missing Streamlit gives a clean error message.
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print(
            "[GlitchVision] Streamlit is not installed. "
            "Run: pip install -r requirements.txt"
        )
        return 1

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
