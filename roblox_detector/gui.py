"""Desktop GUI for the Roblox character detector.

Load an image, a screenshot, or a whole folder; run detection; and inspect
confidence-scored boxes with live-tunable sensitivity. Purely an analysis /
visualization tool - it never controls your mouse or interacts with a game.

Run:  python -m roblox_detector.gui
"""

from __future__ import annotations

import os
import threading

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:  # pragma: no cover - environment without Tk
    raise SystemExit(
        "tkinter is required for the GUI. On Debian/Ubuntu: "
        "sudo apt-get install python3-tk"
    ) from exc

from PIL import Image, ImageTk

from .detector import (
    Detection,
    DetectorConfig,
    RobloxCharacterDetector,
    _make_hog,
    annotate,
)

SUPPORTED = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif")


class DetectorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Roblox Character Detector")
        self.root.geometry("1180x760")
        self.root.minsize(900, 600)

        self.config = DetectorConfig()
        self.detector = RobloxCharacterDetector(self.config)

        self._image_bgr: np.ndarray | None = None
        self._detections: list[Detection] = []
        self._display_img: ImageTk.PhotoImage | None = None
        self._folder: list[str] = []
        self._folder_idx = 0
        self._busy = False

        self._build_style()
        self._build_layout()
        self._bind_keys()
        self._set_status("Open an image or folder to begin.")

    # -- styling ----------------------------------------------------------------

    def _build_style(self):
        self.root.configure(bg="#1b1d23")
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#1b1d23")
        style.configure("Side.TFrame", background="#23262e")
        style.configure("TLabel", background="#23262e", foreground="#e6e6e6")
        style.configure("Head.TLabel", background="#23262e", foreground="#8ecbff",
                        font=("Segoe UI", 11, "bold"))
        style.configure("Status.TLabel", background="#12141a", foreground="#9fb3c8")
        style.configure("TButton", background="#2f6fed", foreground="white",
                        font=("Segoe UI", 10, "bold"), borderwidth=0, padding=6)
        style.map("TButton", background=[("active", "#4b83f0")])
        style.configure("TScale", background="#23262e")
        style.configure("TCheckbutton", background="#23262e", foreground="#e6e6e6")

    def _build_layout(self):
        # Sidebar -------------------------------------------------------------
        side = ttk.Frame(self.root, style="Side.TFrame", width=280)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        ttk.Label(side, text="Roblox Detector", style="Head.TLabel").pack(
            anchor="w", padx=16, pady=(16, 2))
        ttk.Label(side, text="Character detection & analysis",
                  style="TLabel", font=("Segoe UI", 8)).pack(anchor="w", padx=16)

        box = ttk.Frame(side, style="Side.TFrame")
        box.pack(fill="x", padx=12, pady=12)
        ttk.Button(box, text="Open Image", command=self.open_image).pack(fill="x", pady=3)
        ttk.Button(box, text="Open Folder", command=self.open_folder).pack(fill="x", pady=3)
        ttk.Button(box, text="Load Screenshot File", command=self.open_image).pack(
            fill="x", pady=3)

        nav = ttk.Frame(side, style="Side.TFrame")
        nav.pack(fill="x", padx=12)
        ttk.Button(nav, text="◀ Prev", command=lambda: self._step_folder(-1)).pack(
            side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(nav, text="Next ▶", command=lambda: self._step_folder(1)).pack(
            side="left", expand=True, fill="x", padx=(3, 0))

        ttk.Separator(side, orient="horizontal").pack(fill="x", padx=12, pady=12)

        # Sensitivity slider --------------------------------------------------
        ttk.Label(side, text="Detection sensitivity", style="TLabel").pack(
            anchor="w", padx=16)
        self.sens_var = tk.DoubleVar(value=self.config.confidence_threshold)
        self.sens_lbl = ttk.Label(side, text="", style="TLabel")
        self.sens_lbl.pack(anchor="w", padx=16)
        sc = ttk.Scale(side, from_=0.05, to=0.9, variable=self.sens_var,
                       command=self._on_sensitivity)
        sc.pack(fill="x", padx=14)
        self._on_sensitivity(None)

        # Strategy toggles ----------------------------------------------------
        ttk.Label(side, text="Strategies", style="TLabel").pack(
            anchor="w", padx=16, pady=(14, 2))
        self.var_assembly = tk.BooleanVar(value=True)
        self.var_flat = tk.BooleanVar(value=True)
        self.var_hog = tk.BooleanVar(value=True)
        for text, var in [("Head + body assembly", self.var_assembly),
                          ("Flat-color regions", self.var_flat),
                          ("HOG humanoid silhouette", self.var_hog)]:
            ttk.Checkbutton(side, text=text, variable=var,
                            command=self._rerun).pack(anchor="w", padx=16)

        ttk.Separator(side, orient="horizontal").pack(fill="x", padx=12, pady=12)
        ttk.Button(side, text="Re-run Detection", command=self._rerun).pack(
            fill="x", padx=12, pady=3)
        ttk.Button(side, text="Save Annotated…", command=self.save_annotated).pack(
            fill="x", padx=12, pady=3)
        ttk.Button(side, text="Export JSON…", command=self.export_json).pack(
            fill="x", padx=12, pady=3)

        self.result_lbl = ttk.Label(side, text="", style="TLabel",
                                    font=("Segoe UI", 10, "bold"), wraplength=250)
        self.result_lbl.pack(anchor="w", padx=16, pady=14)

        # Canvas + status -----------------------------------------------------
        main = ttk.Frame(self.root, style="TFrame")
        main.pack(side="right", fill="both", expand=True)
        self.canvas = tk.Canvas(main, bg="#0e0f13", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._render())
        self.status = ttk.Label(main, text="", style="Status.TLabel", anchor="w")
        self.status.pack(fill="x", side="bottom", ipady=4)

    def _bind_keys(self):
        self.root.bind("<Left>", lambda e: self._step_folder(-1))
        self.root.bind("<Right>", lambda e: self._step_folder(1))
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-s>", lambda e: self.save_annotated())

    # -- actions ----------------------------------------------------------------

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.gif"),
                       ("All files", "*.*")])
        if path:
            self._folder = []
            self._load_path(path)

    def open_folder(self):
        folder = filedialog.askdirectory(title="Open folder of images")
        if not folder:
            return
        files = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(SUPPORTED))
        if not files:
            messagebox.showinfo("Empty", "No supported images in that folder.")
            return
        self._folder = files
        self._folder_idx = 0
        self._load_path(files[0])

    def _step_folder(self, delta: int):
        if not self._folder:
            return
        self._folder_idx = (self._folder_idx + delta) % len(self._folder)
        self._load_path(self._folder[self._folder_idx])

    def _load_path(self, path: str):
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Unsupported or corrupt image.")
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open image:\n{exc}")
            return
        self._image_bgr = img
        self._current_path = path
        name = os.path.basename(path)
        pos = f" [{self._folder_idx + 1}/{len(self._folder)}]" if self._folder else ""
        self._set_status(f"Loaded {name}{pos}  —  running detection…")
        self._run_detection_async()

    def _run_detection_async(self):
        if self._image_bgr is None or self._busy:
            return
        self._busy = True
        self._sync_config()
        img = self._image_bgr.copy()

        def work():
            try:
                dets = self.detector.detect(img)
            except Exception as exc:  # pragma: no cover
                self.root.after(0, lambda: self._detection_failed(exc))
                return
            self.root.after(0, lambda: self._detection_done(dets))

        threading.Thread(target=work, daemon=True).start()

    def _detection_done(self, dets: list[Detection]):
        self._busy = False
        self._detections = dets
        strong = sum(1 for d in dets if d.score >= 0.6)
        self.result_lbl.config(
            text=f"{len(dets)} character(s) detected\n"
                 f"{strong} high-confidence")
        name = os.path.basename(getattr(self, "_current_path", ""))
        self._set_status(f"{name}  —  {len(dets)} detection(s).")
        self._render()

    def _detection_failed(self, exc):  # pragma: no cover
        self._busy = False
        messagebox.showerror("Detection error", str(exc))

    def _rerun(self):
        if self._image_bgr is not None:
            self._run_detection_async()

    def _sync_config(self):
        self.config.confidence_threshold = float(self.sens_var.get())
        self.config.use_assembly = self.var_assembly.get()
        self.config.use_flat_regions = self.var_flat.get()
        self.config.use_hog = self.var_hog.get()
        # HOG toggling requires (re)creating the descriptor.
        if self.config.use_hog and self.detector._hog is None:
            self.detector._hog = _make_hog()
        if not self.config.use_hog:
            self.detector._hog = None

    def _on_sensitivity(self, _):
        self.sens_lbl.config(text=f"threshold = {self.sens_var.get():.2f}")
        # Live re-threshold without a full re-run when possible.
        if self._image_bgr is not None and not self._busy:
            self._rerun()

    def save_annotated(self):
        if self._image_bgr is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not path:
            return
        out = annotate(self._image_bgr, self._detections)
        ok, buf = cv2.imencode(os.path.splitext(path)[1], out)
        if ok:
            buf.tofile(path)
            self._set_status(f"Saved annotated image to {path}")

    def export_json(self):
        if self._image_bgr is None:
            return
        import json
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        payload = {
            "image": os.path.basename(getattr(self, "_current_path", "")),
            "count": len(self._detections),
            "detections": [d.as_dict() for d in self._detections],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        self._set_status(f"Exported {len(self._detections)} detection(s) to {path}")

    # -- rendering --------------------------------------------------------------

    def _render(self):
        if self._image_bgr is None:
            return
        annotated = annotate(self._image_bgr, self._detections)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img_h, img_w = rgb.shape[:2]
        cw = max(self.canvas.winfo_width(), 10)
        ch = max(self.canvas.winfo_height(), 10)
        scale = min(cw / img_w, ch / img_h)
        disp_w, disp_h = max(1, int(img_w * scale)), max(1, int(img_h * scale))
        pil = Image.fromarray(rgb).resize((disp_w, disp_h), Image.LANCZOS)
        self._display_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._display_img)

    def _set_status(self, text: str):
        self.status.config(text="  " + text)


def main():
    root = tk.Tk()
    DetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
