from pathlib import Path
import runpy
import sys


PROJECT_ROOT = Path(__file__).resolve().parent


def run_dashboard():
    # Streamlit reruns the launcher in the same Python process. Clearing local
    # modules prevents stale UI code from being mixed with newly loaded CSS.
    local_module_names = [
        module_name
        for module_name in sys.modules
        if module_name == "dashboard"
        or module_name.startswith("dashboard.")
        or module_name == "src"
        or module_name.startswith("src.")
    ]
    for module_name in local_module_names:
        sys.modules.pop(module_name, None)

    runpy.run_path(
        str(PROJECT_ROOT / "dashboard" / "app.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    run_dashboard()
