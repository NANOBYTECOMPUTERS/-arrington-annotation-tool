"""Arrington Annotation Tool (AAT) Viewer - detect annotation GUI with AI assist.

This is the primary desktop tool for reviewing, editing, and AI-suggesting YOLO detect labels.

It reuses proven editing/canvas logic while integrating the new modular
aat.inference + aat.generate stack for engine suggestions.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from PIL import Image, ImageTk

from aat.dataset import (
    guess_label_for_image,
    guess_label_root,
    scan_viewer_images,
)
from aat.generate import GenerateConfig, generate_detect_dataset
from aat.inference import get_engine
from aat.inference.protocol import DetectionEngine
from aat.io.shapes import (
    AnnotationShape,
    IMAGE_EXTENSIONS,
    read_annotation_shapes,
    write_annotation_shapes,
)

from yolo_annotator.classes import load_class_metadata  # type: ignore


PALETTE = ("#00d084", "#ff4d4f", "#4096ff", "#faad14", "#9254de", "#13c2c2")


def label_path_for_view(image_path: str | Path, image_root: str | Path, label_root: str | Path) -> Path:
    image = Path(image_path).expanduser().resolve()
    image_base = Path(image_root).expanduser().resolve()
    labels = Path(label_root).expanduser().resolve()
    relative = image.relative_to(image_base)
    direct = labels / relative.with_suffix(".txt")
    if direct.exists():
        return direct
    parts = list(relative.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return labels / Path(*parts).with_suffix(".txt")
    return direct


def canvas_rect_to_image_rect(
    start: tuple[float, float],
    end: tuple[float, float],
    transform: dict[str, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    scale = float(transform["scale"])
    if scale <= 0:
        raise ValueError("Canvas transform scale must be positive")
    canvas_x1, canvas_x2 = sorted((float(start[0]), float(end[0])))
    canvas_y1, canvas_y2 = sorted((float(start[1]), float(end[1])))
    x1 = int(max(0, min(image_width, (canvas_x1 - float(transform["x"])) / scale)))
    y1 = int(max(0, min(image_height, (canvas_y1 - float(transform["y"])) / scale)))
    x2 = int(max(0, min(image_width, ceil((canvas_x2 - float(transform["x"])) / scale))))
    y2 = int(max(0, min(image_height, ceil((canvas_y2 - float(transform["y"])) / scale))))
    return x1, y1, x2, y2


def remove_shapes_intersecting_rect(
    shapes: list[AnnotationShape],
    rect: tuple[int, int, int, int],
) -> tuple[list[AnnotationShape], int]:
    kept: list[AnnotationShape] = []
    removed = 0
    for shape in shapes:
        if _shape_center_inside_rect(shape, rect):
            removed += 1
        else:
            kept.append(shape)
    return kept, removed


def _shape_center_inside_rect(shape: AnnotationShape, rect: tuple[int, int, int, int]) -> bool:
    if not shape.points:
        return False
    x1, y1, x2, y2 = rect
    xs = [point[0] for point in shape.points]
    ys = [point[1] for point in shape.points]
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    return x1 <= center_x <= x2 and y1 <= center_y <= y2


class AATViewerApp:
    """Main Arrington Annotation Tool viewer window with engine integration."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Arrington Annotation Tool (AAT)")

        self.single_image_var = tk.StringVar()
        self.single_label_var = tk.StringVar()
        self.image_root_var = tk.StringVar()
        self.label_root_var = tk.StringVar()
        self.data_yaml_var = tk.StringVar()

        # NEW: Engine / AI Assist
        self.backend_var = tk.StringVar(value="worker")
        self.engine_path_var = tk.StringVar()
        self.worker_path_var = tk.StringVar()
        self.output_root_var = tk.StringVar()
        self.manual_classes_var = tk.StringVar()
        self.confidence_var = tk.StringVar(value="0.25")
        self.iou_var = tk.StringVar(value="0.45")
        self.batch_size_var = tk.StringVar(value="16")
        self.engine_status_var = tk.StringVar(value="No engine loaded")
        self.current_engine: DetectionEngine | None = None
        self.worker_thread: threading.Thread | None = None

        self.editor_mode_var = tk.StringVar(value="add")
        self.edit_class_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="Choose image and label folders or load a model for AI assist.")
        self.position_var = tk.StringVar(value="0 / 0")

        self.images: list = []
        self.index = -1
        self.class_names: list[str] = []
        self.single_image_mode = False
        self.current_photo: ImageTk.PhotoImage | None = None
        self.current_transform: dict[str, float] | None = None
        self.current_shapes: list[AnnotationShape] = []
        self.current_image_path: Any = None
        self.current_label_path: Any = None
        self.current_image_size: tuple[int, int] | None = None
        self.dirty = False
        self.selection_start: tuple[float, float] | None = None
        self.selection_rect_id: int | None = None

        self._build_layout()
        self._apply_smart_defaults()
        self.root.bind("<Left>", lambda _e: self.previous_image())
        self.root.bind("<Right>", lambda _e: self.next_image())

    def _apply_smart_defaults(self) -> None:
        """Auto-detect a usable C++ worker when the path field is still empty.

        Preference order (first existing wins):
        1. Bundled next to the frozen exe or in ./engine/ (what the build script produces)
        2. Classic 0BS dev location (for running from source during development)
        """
        if self.worker_path_var.get().strip():
            return  # user or previous logic already set one

        import sys

        candidates: list[Path] = []

        if getattr(sys, "frozen", False):
            # Running from PyInstaller onedir bundle
            base = Path(sys.executable).resolve().parent
            candidates.extend([
                base / "yolo_annotation_worker.exe",
                base / "engine" / "yolo_annotation_worker.exe",
            ])
        else:
            # Dev run (python tools/aat_viewer_app.py or -m)
            # Look relative to the AAT workspace first, then the well-known 0BS build output
            here = Path(__file__).resolve()
            # Walk up a few levels to find possible "engine" or root
            for up in [here] + list(here.parents)[:5]:
                candidates.append(up / "yolo_annotation_worker.exe")
                candidates.append(up / "engine" / "yolo_annotation_worker.exe")
                candidates.append(up / "build" / "Arrington Annotation Tool" / "engine" / "yolo_annotation_worker.exe")

        # Final dev convenience fallback (only used when running from source
        # on the original development machine that also contains the 0BS tree).
        # On a normal machine after `build.ps1`, the earlier relative-to-bundle
        # and relative-to-workspace checks will have already found engine/ inside AAT.
        candidates.append(Path(r"C:\Users\donar\OneDrive\Desktop\0BS\x64\CUDA\yolo_annotation_worker.exe"))

        for cand in candidates:
            try:
                if cand and cand.exists():
                    self.worker_path_var.set(str(cand))
                    return
            except Exception:
                pass

    # ---------- Layout ----------

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        # Paths
        paths = ttk.LabelFrame(self.root, text="Image and Labels")
        paths.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        paths.columnconfigure(1, weight=1)

        self._path_row(paths, 0, "Image File", self.single_image_var, self.choose_single_image)
        self._path_row(paths, 1, "Label File", self.single_label_var, self.choose_single_label)
        self._path_row(paths, 2, "Images Folder", self.image_root_var, self.choose_images)
        self._path_row(paths, 3, "Labels Folder", self.label_root_var, self.choose_labels)
        self._path_row(paths, 4, "data.yaml", self.data_yaml_var, self.choose_data_yaml)

        # Detect label editor
        editor = ttk.LabelFrame(self.root, text="Detect Label Editor")
        editor.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        for c in range(7):
            editor.columnconfigure(c, weight=1)

        ttk.Label(editor, text="Mode").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(editor, textvariable=self.editor_mode_var, values=("add", "remove"),
                     state="readonly", width=10).grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(editor, text="Class ID").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Entry(editor, textvariable=self.edit_class_var, width=8).grid(row=0, column=3, sticky="ew", padx=8, pady=6)
        ttk.Button(editor, text="Save", command=self.save_current).grid(row=0, column=4, sticky="ew", padx=8, pady=6)

        # NEW: AI Assist / Engine panel (the modular AAT differentiator)
        ai = ttk.LabelFrame(self.root, text="AI Assist - Detect Annotation")
        ai.grid(row=2, column=0, sticky="ew", padx=10, pady=(4, 6))
        ai.columnconfigure(3, weight=1)
        ai.columnconfigure(6, weight=1)

        ttk.Label(ai, text="Backend").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(
            ai,
            textvariable=self.backend_var,
            values=("worker",),
            state="readonly",
            width=12,
        ).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(ai, text="Model / Engine").grid(row=0, column=2, sticky="w", padx=8, pady=4)
        ttk.Entry(ai, textvariable=self.engine_path_var).grid(row=0, column=3, sticky="ew", padx=4)
        ttk.Button(ai, text="Browse", command=self.choose_engine).grid(row=0, column=4, padx=4)
        ttk.Button(ai, text="Load", command=self.load_engine).grid(row=0, column=5, padx=4)
        ttk.Button(ai, text="Suggest Current", command=self.suggest_current).grid(row=0, column=6, sticky="ew", padx=4)

        ttk.Label(ai, text="C++ Worker").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(ai, textvariable=self.worker_path_var).grid(row=1, column=1, columnspan=3, sticky="ew", padx=4)
        ttk.Button(ai, text="Browse", command=self.choose_worker).grid(row=1, column=4, padx=4)
        ttk.Label(ai, text="Classes").grid(row=1, column=5, sticky="w", padx=8)
        ttk.Entry(ai, textvariable=self.manual_classes_var).grid(row=1, column=6, sticky="ew", padx=4)

        ttk.Label(ai, text="Output").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(ai, textvariable=self.output_root_var).grid(row=2, column=1, columnspan=3, sticky="ew", padx=4)
        ttk.Button(ai, text="Browse", command=self.choose_output_root).grid(row=2, column=4, padx=4)
        ttk.Button(ai, text="Auto Annotate Folder", command=self.auto_annotate_folder).grid(
            row=2, column=5, columnspan=2, sticky="ew", padx=4
        )

        numeric = ttk.Frame(ai)
        numeric.grid(row=3, column=0, columnspan=7, sticky="ew", padx=8, pady=(2, 4))
        ttk.Label(numeric, text="Conf").pack(side="left")
        ttk.Entry(numeric, textvariable=self.confidence_var, width=7).pack(side="left", padx=(4, 12))
        ttk.Label(numeric, text="NMS").pack(side="left")
        ttk.Entry(numeric, textvariable=self.iou_var, width=7).pack(side="left", padx=(4, 12))
        ttk.Label(numeric, text="Batch").pack(side="left")
        ttk.Entry(numeric, textvariable=self.batch_size_var, width=7).pack(side="left", padx=(4, 12))
        ttk.Label(numeric, textvariable=self.engine_status_var, foreground="#0066cc").pack(side="left", padx=(12, 0))

        # Viewer area
        viewer = ttk.Frame(self.root)
        viewer.grid(row=3, column=0, sticky="nsew", padx=10, pady=4)
        viewer.columnconfigure(1, weight=1)
        viewer.rowconfigure(0, weight=1)

        self.image_list = tk.Listbox(viewer, width=34)
        self.image_list.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        self.image_list.bind("<<ListboxSelect>>", self._select_from_list)

        self.canvas = tk.Canvas(viewer, width=960, height=640, bg="#111111", highlightthickness=0)
        self.canvas.grid(row=0, column=1, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _e: self.render_current())
        self.canvas.bind("<ButtonPress-1>", self._selection_press)
        self.canvas.bind("<B1-Motion>", self._selection_motion)
        self.canvas.bind("<ButtonRelease-1>", self._selection_release)

        # Bottom actions
        actions = ttk.Frame(self.root)
        actions.grid(row=4, column=0, sticky="ew", padx=10, pady=(4, 10))
        actions.columnconfigure(0, weight=1)

        ttk.Label(actions, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="Load Image", command=self.load_single_image).grid(row=0, column=1, padx=4)
        ttk.Button(actions, text="Load Folder", command=self.load_images).grid(row=0, column=2, padx=4)
        ttk.Button(actions, text="Previous", command=self.previous_image).grid(row=0, column=3, padx=4)
        ttk.Label(actions, textvariable=self.position_var).grid(row=0, column=4, padx=8)
        ttk.Button(actions, text="Next", command=self.next_image).grid(row=0, column=5, padx=4)

    def _path_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, command: Any) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, sticky="ew", padx=8, pady=4)

    # ---------- Engine / AI integration (new modular feature) ----------

    def choose_engine(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select model (.pt, .engine, .onnx)",
            filetypes=[("Models", "*.pt *.engine *.onnx"), ("All files", "*.*")],
        )
        if selected:
            self.engine_path_var.set(selected)

    def choose_worker(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select C++ TensorRT worker executable",
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
        )
        if selected:
            self.worker_path_var.set(selected)

    def choose_output_root(self) -> None:
        selected = filedialog.askdirectory(title="Select output dataset folder", initialdir=self.output_root_var.get() or None)
        if selected:
            self.output_root_var.set(selected)

    def load_engine(self) -> None:
        path = self.engine_path_var.get().strip()
        if not path:
            messagebox.showwarning("AAT Viewer", "Choose a model file first.")
            return
        try:
            close = getattr(self.current_engine, "close", None)
            if close:
                close()
            backend = self.backend_var.get()
            kwargs = self._engine_kwargs()
            self.current_engine = get_engine(path, backend=backend, **kwargs)
            names = getattr(self.current_engine, "names", []) or []
            self.engine_status_var.set(f"Loaded: {Path(path).name}  ({len(names)} classes)")
            self.status_var.set(f"Engine ready - click Suggest Current or Auto Annotate Folder. {len(names)} classes")
        except Exception as exc:
            self.current_engine = None
            self.engine_status_var.set("Failed to load engine")
            messagebox.showerror("AAT Viewer - Engine Load Failed", str(exc))

    def suggest_current(self) -> None:
        if self.index < 0 or self.index >= len(self.images):
            messagebox.showinfo("AAT Viewer", "Load an image first.")
            return
        if self.current_engine is None:
            if not self.engine_path_var.get():
                self.choose_engine()
            if not self.engine_path_var.get():
                return
            self.load_engine()
            if self.current_engine is None:
                return

        image_path = self.images[self.index]
        try:
            preds = self.current_engine.predict(
                [image_path],
                conf=float(self.confidence_var.get()),
                iou=float(self.iou_var.get()),
                batch_size=max(1, int(self.batch_size_var.get())),
            )
            pred = preds[0] if preds else None
            if not pred or not pred.boxes:
                messagebox.showinfo("AAT Viewer", "Engine returned no detections on this image.")
                return

            added = 0
            for box in pred.boxes:
                # Convert model output pixels to an AnnotationShape rectangle.
                x1, y1 = box.x, box.y
                x2 = box.x + box.width
                y2 = box.y + box.height
                shape = AnnotationShape(
                    class_id=box.class_id,
                    kind="detect",
                    points=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                )
                self.current_shapes.append(shape)
                added += 1

            self.dirty = True
            self.render_current()
            self.status_var.set(f"Added {added} suggestion(s) from engine. Remember to Save.")
        except Exception as exc:
            messagebox.showerror("AAT Viewer - Suggest Failed", str(exc))

    def auto_annotate_folder(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showinfo("AAT Viewer", "Auto annotation is already running.")
            return
        if not self.image_root_var.get().strip():
            messagebox.showwarning("AAT Viewer", "Choose an Images Folder first.")
            return
        if not self.output_root_var.get().strip():
            messagebox.showwarning("AAT Viewer", "Choose an Output folder first.")
            return
        if not self.engine_path_var.get().strip():
            messagebox.showwarning("AAT Viewer", "Choose a model or TensorRT engine first.")
            return
        try:
            config = self._generate_config()
        except Exception as exc:
            messagebox.showerror("AAT Viewer", str(exc))
            return
        self.status_var.set("Auto annotation running...")
        self.worker_thread = threading.Thread(target=self._run_auto_annotate, args=(config,), daemon=True)
        self.worker_thread.start()

    def _run_auto_annotate(self, config: GenerateConfig) -> None:
        try:
            result = generate_detect_dataset(config, progress=self._annotation_progress)
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("AAT Viewer - Auto Annotate Failed", str(exc)))
            self.root.after(0, lambda: self.status_var.set("Auto annotation failed."))
            return
        self.root.after(
            0,
            lambda: self.status_var.set(
                f"Auto annotation complete: {result.processed} images, {result.total_detections} boxes, {result.failed} failed."
            ),
        )

    def _annotation_progress(self, event: dict[str, Any]) -> None:
        if "error" in event:
            self.root.after(0, lambda: self.status_var.set(f"Failed: {event.get('image')} - {event.get('error')}"))
            return
        self.root.after(
            0,
            lambda: self.status_var.set(
                f"Annotated {event.get('processed')} images - {event.get('dets')} detections on {Path(str(event.get('image'))).name}"
            ),
        )

    def _engine_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "confidence": float(self.confidence_var.get()),
            "iou": float(self.iou_var.get()),
            "batch_size": max(1, int(self.batch_size_var.get())),
        }
        if self.backend_var.get() == "worker":
            worker = self.worker_path_var.get().strip()
            if not worker:
                raise ValueError("Choose the C++ worker executable for worker backend.")
            kwargs["worker_command"] = [worker]
            names = self._active_class_names()
            if names:
                kwargs["names"] = names
        return kwargs

    def _generate_config(self) -> GenerateConfig:
        return GenerateConfig(
            images_root=Path(self.image_root_var.get()).expanduser(),
            model_path=Path(self.engine_path_var.get()).expanduser(),
            output_root=Path(self.output_root_var.get()).expanduser(),
            backend=self.backend_var.get(),
            worker_command=[self.worker_path_var.get().strip()] if self.backend_var.get() == "worker" else None,
            class_names=self._active_class_names() or None,
            confidence=float(self.confidence_var.get()),
            iou=float(self.iou_var.get()),
            batch_size=max(1, int(self.batch_size_var.get())),
            force=True,
        )

    def _active_class_names(self) -> list[str]:
        if self.class_names:
            return list(self.class_names)
        text = self.manual_classes_var.get().strip()
        if not text:
            return []
        if "\n" in text:
            return [line.strip() for line in text.splitlines() if line.strip()]
        return [item.strip() for item in text.split(",") if item.strip()]

    # ---------- Rest of the viewer (adapted from original, using aat.io where possible) ----------

    def choose_single_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if selected:
            p = selected
            self.single_image_var.set(p)
            if not self.single_label_var.get():
                guess = guess_label_for_image(Path(p))
                if guess:
                    self.single_label_var.set(str(guess))
            if not self.image_root_var.get():
                self.image_root_var.set(str(Path(p).parent))

    def choose_single_label(self) -> None:
        sel = filedialog.askopenfilename(title="Select label file", filetypes=[("YOLO labels", "*.txt"), ("All", "*.*")])
        if sel:
            self.single_label_var.set(sel)

    def choose_images(self) -> None:
        sel = filedialog.askdirectory(title="Select image folder")
        if sel:
            self.image_root_var.set(sel)
            guess = guess_label_root(Path(sel))
            if guess and not self.label_root_var.get():
                self.label_root_var.set(str(guess))

    def choose_labels(self) -> None:
        sel = filedialog.askdirectory(title="Select labels folder")
        if sel:
            self.label_root_var.set(sel)

    def choose_data_yaml(self) -> None:
        sel = filedialog.askopenfilename(title="Select data.yaml", filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")])
        if sel:
            self.data_yaml_var.set(sel)
            self._load_data_yaml_metadata()

    def load_single_image(self) -> None:
        if not self._confirm_discard():
            return
        try:
            p = Path(self.single_image_var.get()).expanduser().resolve()
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
                raise FileNotFoundError(str(p))
            if self.data_yaml_var.get():
                self._load_data_yaml_metadata()
        except Exception as e:
            messagebox.showerror("AAT Viewer", str(e))
            return

        self.single_image_mode = True
        self.images = [p]
        self.index = 0
        self.image_list.delete(0, "end")
        self.image_list.insert("end", p.name)
        self.image_list.selection_set(0)
        self.render_current(force_reload=True)

    def load_images(self) -> None:
        if not self._confirm_discard():
            return
        try:
            self.images = scan_viewer_images(self.image_root_var.get())
            if self.data_yaml_var.get():
                self._load_data_yaml_metadata()
        except Exception as e:
            messagebox.showerror("AAT Viewer", str(e))
            return

        self.single_image_mode = False
        self.image_list.delete(0, "end")
        root = Path(self.image_root_var.get()).expanduser().resolve()
        for img in self.images:
            try:
                self.image_list.insert("end", str(img.relative_to(root)))
            except Exception:
                self.image_list.insert("end", img.name)
        self.index = 0 if self.images else -1
        if self.index >= 0:
            self.image_list.selection_set(self.index)
        self.render_current(force_reload=True)

    def previous_image(self) -> None:
        if not self.images or not self._confirm_discard():
            return
        self.index = max(0, self.index - 1)
        self._sync_list()
        self.render_current(force_reload=True)

    def next_image(self) -> None:
        if not self.images or not self._confirm_discard():
            return
        self.index = min(len(self.images) - 1, self.index + 1)
        self._sync_list()
        self.render_current(force_reload=True)

    def render_current(self, *, force_reload: bool = False) -> None:
        self.canvas.delete("all")
        if self.index < 0 or self.index >= len(self.images):
            self.position_var.set("0 / 0")
            self.canvas.create_text(20, 24, anchor="nw", fill="#dddddd", text="Load images or use AI Suggest after loading an engine.")
            return

        image_path = self.images[self.index]
        with Image.open(image_path) as src:
            img = src.convert("RGB")
            w, h = img.size
            tf = self._fit_transform(w, h, self.canvas.winfo_width(), self.canvas.winfo_height())
            resized = img.resize((int(tf["width"]), int(tf["height"])))

        self.current_photo = ImageTk.PhotoImage(resized)
        self.current_transform = tf
        self.canvas.create_image(tf["x"], tf["y"], anchor="nw", image=self.current_photo)

        warnings: list[str] = []
        label_path = self._label_path_for_current(image_path)

        if force_reload or self.current_image_path != image_path or self.current_label_path != label_path:
            self.current_shapes = read_annotation_shapes(
                label_path, image_width=w, image_height=h, task=None, warnings=warnings
            )
            self.current_shapes = [self._as_detect_shape(shape) for shape in self.current_shapes]
            self.dirty = False

        self.current_image_path = image_path
        self.current_label_path = label_path
        self.current_image_size = (w, h)

        for shape in self.current_shapes:
            self._draw_shape(shape, tf)

        self.position_var.set(f"{self.index + 1} / {len(self.images)}")
        rel = image_path.name
        dirty = " | unsaved" if self.dirty else ""
        self.status_var.set(f"{rel} | labels: {len(self.current_shapes)}{dirty} | {label_path}")
        if warnings:
            self.canvas.create_text(12, 12, anchor="nw", fill="#ffcc00", text=f"Warnings: {len(warnings)}")

    def _draw_shape(self, shape: AnnotationShape, tf: dict[str, float]) -> None:
        color = PALETTE[shape.class_id % len(PALETTE)]
        flat: list[float] = []
        for x, y in shape.points:
            flat.extend([tf["x"] + x * tf["scale"], tf["y"] + y * tf["scale"]])
        self.canvas.create_polygon(flat, outline=color, fill="", width=2)
        if shape.points:
            x, y = shape.points[0]
            name = self.class_names[shape.class_id] if 0 <= shape.class_id < len(self.class_names) else str(shape.class_id)
            self.canvas.create_text(
                tf["x"] + x * tf["scale"],
                tf["y"] + y * tf["scale"] - 10,
                anchor="sw", fill=color,
                text=f"{name} ({shape.kind})",
            )

    def save_current(self) -> None:
        if self.current_label_path is None or self.current_image_size is None:
            messagebox.showwarning("AAT Viewer", "Load an image before saving.")
            return
        detect_shapes = [self._as_detect_shape(shape) for shape in self.current_shapes]
        write_annotation_shapes(
            self.current_label_path,
            detect_shapes,
            image_width=self.current_image_size[0],
            image_height=self.current_image_size[1],
            task=None,
        )
        self.current_shapes = detect_shapes
        self.dirty = False
        self.render_current()

    # Selection / editing (delegated to the same logic as before)
    def _selection_press(self, event: tk.Event) -> None:
        if self.current_transform is None or self.current_image_size is None:
            return
        self.selection_start = (float(event.x), float(event.y))
        if self.selection_rect_id is not None:
            self.canvas.delete(self.selection_rect_id)
        self.selection_rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="#ffffff", dash=(4, 3), width=2
        )

    def _selection_motion(self, event: tk.Event) -> None:
        if self.selection_start is None or self.selection_rect_id is None:
            return
        self.canvas.coords(self.selection_rect_id, self.selection_start[0], self.selection_start[1], event.x, event.y)

    def _selection_release(self, event: tk.Event) -> None:
        if self.selection_start is None or self.current_transform is None or self.current_image_size is None:
            return
        start = self.selection_start
        self.selection_start = None
        if self.selection_rect_id is not None:
            self.canvas.delete(self.selection_rect_id)
            self.selection_rect_id = None

        rect = canvas_rect_to_image_rect(start, (float(event.x), float(event.y)), self.current_transform,
                                         image_width=self.current_image_size[0], image_height=self.current_image_size[1])
        if rect[2] - rect[0] < 3 or rect[3] - rect[1] < 3:
            return

        if self.editor_mode_var.get() == "remove":
            self._remove_shapes_in_rect(rect)
        else:
            self._add_detect_from_rect(rect)

    def _add_detect_from_rect(self, rect: tuple[int, int, int, int]) -> None:
        try:
            cid = int(self.edit_class_var.get())
        except ValueError:
            messagebox.showerror("AAT Viewer", "Class ID must be numeric.")
            return

        x1, y1, x2, y2 = rect
        self.current_shapes.append(
            AnnotationShape(
                class_id=cid,
                kind="detect",
                points=[
                    (float(x1), float(y1)),
                    (float(x2), float(y1)),
                    (float(x2), float(y2)),
                    (float(x1), float(y2)),
                ],
            )
        )
        self.dirty = True
        self.render_current()

    def _remove_shapes_in_rect(self, rect: tuple[int, int, int, int]) -> None:
        kept, removed = remove_shapes_intersecting_rect(self.current_shapes, rect)
        if removed <= 0:
            self.status_var.set("No annotation center found inside the box.")
            return
        self.current_shapes = kept
        self.dirty = True
        self.render_current()

    # Helpers

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        return messagebox.askyesno("AAT Viewer", "Discard unsaved changes?")

    def _sync_list(self) -> None:
        self.image_list.selection_clear(0, "end")
        self.image_list.selection_set(self.index)
        self.image_list.see(self.index)

    def _select_from_list(self, _event: object) -> None:
        sel = self.image_list.curselection()
        if not sel:
            return
        if not self._confirm_discard():
            self._sync_list()
            return
        self.index = int(sel[0])
        self.render_current(force_reload=True)

    def _label_path_for_current(self, image_path: Any) -> Any:
        if self.single_image_mode and self.single_label_var.get():
            return self.single_label_var.get()
        if self.single_image_mode:
            g = guess_label_for_image(Path(image_path))
            return g or Path(image_path).with_suffix(".txt")
        return label_path_for_view(image_path, self.image_root_var.get(), self.label_root_var.get())

    def _load_data_yaml_metadata(self) -> None:
        try:
            meta = load_class_metadata(data_yaml=self.data_yaml_var.get())
            self.class_names = meta.names
        except Exception as e:
            messagebox.showwarning("AAT Viewer", f"Could not load class names:\n{e}")
            self.class_names = []

    def _as_detect_shape(self, shape: AnnotationShape) -> AnnotationShape:
        if shape.kind == "detect":
            return shape
        xs = [point[0] for point in shape.points]
        ys = [point[1] for point in shape.points]
        if not xs or not ys:
            return AnnotationShape(class_id=shape.class_id, kind="detect", points=[])
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return AnnotationShape(
            class_id=shape.class_id,
            kind="detect",
            points=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
        )

    # Thin wrappers around the public modular helpers (for minimal internal change)
    def _guess_label_root(self, image_root: Path) -> Path | None:
        return guess_label_root(image_root)

    def _guess_label_for_image(self, image_path: Path) -> Path | None:
        return guess_label_for_image(image_path)

    def _fit_transform(self, iw: int, ih: int, cw: int, ch: int) -> dict[str, float]:
        cw = max(int(cw), 1)
        ch = max(int(ch), 1)
        scale = min(cw / iw, ch / ih)
        dw = iw * scale
        dh = ih * scale
        return {"scale": scale, "x": (cw - dw) / 2.0, "y": (ch - dh) / 2.0, "width": dw, "height": dh}


def main() -> None:
    root = tk.Tk()
    root.geometry("1220x780")
    AATViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
