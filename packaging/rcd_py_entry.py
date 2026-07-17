"""Frozen entry point for the Python Roblox character detector.

Bundles the classical multi-strategy image detector (roblox_detector) into a
single Windows .exe.

Behavior:
  * With command-line arguments -> runs the batch CLI:
        roblox-detector-py.exe shot.png --out annotated
        roblox-detector-py.exe screenshots\\ --out annotated --json results.json
        roblox-detector-py.exe shot.png --threshold 0.5 --onnx yolo.onnx
  * Double-clicked with no arguments -> launches the desktop GUI if Tk is
    available in the frozen runtime; otherwise prints CLI usage.

(Whether the GUI is present depends on the Python used to build the exe:
official Windows Python ships Tk, so the CI build is GUI-capable.)
"""
import sys


def _run_gui() -> bool:
    try:
        import tkinter  # noqa: F401  (probe: is Tk bundled?)
        from roblox_detector.gui import main as gui_main
    except Exception:
        return False
    gui_main()
    return True


def main() -> int:
    if len(sys.argv) == 1:
        if _run_gui():
            return 0
        sys.argv.append("--help")  # no GUI in this build; show CLI usage
    from roblox_detector.cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
