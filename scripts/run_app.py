"""Streamlit app launcher script.

Spawns a streamlit subprocess to run the web dashboard.
"""

import subprocess
import sys
from pathlib import Path


def launch_app():
    script_dir = Path(__file__).resolve().parent
    app_path = script_dir.parent / "app" / "streamlit_app.py"

    if not app_path.exists():
        print(f"Error: Streamlit application entry point not found at: {app_path}")
        sys.exit(1)

    print(f"Launching Streamlit application from: {app_path}")
    try:
        # Run streamlit run command
        subprocess.run(["streamlit", "run", str(app_path)], check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Failed to launch application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    launch_app()
