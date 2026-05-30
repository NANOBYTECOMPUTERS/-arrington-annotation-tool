"""Arrington Annotation Tool (AAT) Viewer — modular annotation GUI with AI assist.

This is the primary desktop tool. It reuses proven editing logic while integrating
the new modular aat.inference stack for "engine → suggestions".
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageTk

from aat.config import get_config
from aat.dataset import guess_label_for_image, guess_label_root, scan_dataset, scan_viewer_images
from aat.inference import get_engine
from aat.inference.protocol import DetectionEngine
from aat.io.labels import DetectionBox
from aat.io.shapes import (
    AnnotationShape,
    IMAGE_EXTENSIONS,
    parse_annotation_row,
    read_annotation_shapes,
    write_annotation_shapes,
)

# Bridge to mature editing primitives (will be further extracted over time)
from aat.annotation import (
    auto_trace_segment_points,
    canvas_rect_to_image_rect,
    remove_shapes_intersecting_rect,
)
# Still using legacy for a few advanced pieces (label_path_for_view + class loading).
# These will be further extracted in future passes.
from yolo_annotator.annotation_viewer import label_path_for_view  # type: ignore
from yolo_annotator.classes import load_class_metadata  # type: ignore


PALETTE = ("#00d084", "#ff4d4f", "#4096ff", "#faad14", "#9254de", "#13c2c2")


class AATViewerApp:
    """Main Arrington Annotation Tool viewer with engine integration."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Arrington Annotation Tool (AAT)")

        self.single_image_var = tk.StringVar()
        self.single_label_var = tk.StringVar()
        self.image_root_var = tk.StringVar()
        self.label_root_var = tk.StringVar()
        self.data_yaml_var = tk.StringVar()

        # AI Assist
        self.engine_path_var = tk.StringVar()
        self.engine_status_var = tk.StringVar(value="No engine loaded")
        self.current_engine: DetectionEngine | None = None

        self.editor_mode_var = tk.StringVar(value="add")
        self.edit_class_var = tk.StringVar(value="0")
        self.trace_threshold_var = tk.StringVar(value="18")
        self.status_var = tk.StringVar(value="Choose images or load a model for AI suggestions.")
        self.position_var = tk.StringVar(value="0 / 0")

        self.images: list = []
        self.index = -1
        self.class_names: list[str] = []
        self.annotation_task: str | None = None
        self.single_image_mode = False
        self.current_photo = None
        self.current_transform = None
        self.current_shapes: list[AnnotationShape] = []
        self.current_image_path = None
        self.current_label_path = None
        self.current_image_size = None
        self.dirty = False
        self.selection_start = None
        self.selection_rect_id = None

        self._build_layout()
        self.root.bind("<Left>", lambda _e: self.previous_image())
        self.root.bind("<Right>", lambda _e: self.next_image())

    def _build_layout(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        paths = ttk.LabelFrame(self.root, text="Image and Labels")
        paths.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        paths.columnconfigure(1, weight=1)
        self._path_row(paths, 0, "Image File", self.single_image_var, self.choose_single_image)
        self._path_row(paths, 1, "Label File", self.single_label_var, self.choose_single_label)
        self._path_row(paths, 2, "Images Folder", self.image_root_var, self.choose_images)
        self._path_row(paths, 3, "Labels Folder", self.label_root_var, self.choose_labels)
        self._path_row(paths, 4, "data.yaml", self.data_yaml_var, self.choose_data_yaml)

        editor = ttk.LabelFrame(self.root, text="Segment Editor")
        editor.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        for c in range(8):
            editor.columnconfigure(c, weight=1)
        ttk.Label(editor, text="Mode").grid(row=0, column=0, sticky="w", padx=8)
        ttk.Combobox(editor, textvariable=self.editor_mode_var, values=("add", "remove"), state="readonly", width=10).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Label(editor, text="Class ID").grid(row=0, column=2, sticky="w", padx=8)
        ttk.Entry(editor, textvariable=self.edit_class_var, width=8).grid(row=0, column=3, sticky="ew", padx=8)
        ttk.Label(editor, text="Trace Threshold").grid(row=0, column=4, sticky="w", padx=8)
        ttk.Entry(editor, textvariable=self.trace_threshold_var, width=8).grid(row=0, column=5, sticky="ew", padx=8)
        ttk.Button(editor, text="Save", command=self.save_current).grid(row=0, column=6, sticky="ew", padx=8)

        # AI Assist Panel
        ai = ttk.LabelFrame(self.root, text="AI Assist — Load Engine & Suggest")
        ai.grid(row=2, column=0, sticky="ew", padx=10, pady=(4, 6))
        ai.columnconfigure(1, weight=1)
        ttk.Label(ai, text="Model / Engine").grid(row=0, column=0, sticky="w", padx=8)
        ttk.Entry(ai, textvariable=self.engine_path_var).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(ai, text="Browse", command=self.choose_engine).grid(row=0, column=2, padx=4)
        ttk.Button(ai, text="Load Engine", command=self.load_engine).grid(row=0, column=3, padx=4)
        ttk.Button(ai, text="Suggest on Current", command=self.suggest_current).grid(row=0, column=4, padx=8)
        ttk.Label(ai, textvariable=self.engine_status_var, foreground="#0066cc").grid(row=1, column=0, columnspan=5, sticky="w", padx=8)

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

        actions = ttk.Frame(self.root)
        actions.grid(row=4, column=0, sticky="ew", padx=10, pady=(4, 10))
        actions.columnconfigure(0, weight=1)
        ttk.Label(actions, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="Load Image", command=self.load_single_image).grid(row=0, column=1, padx=4)
        ttk.Button(actions, text="Load Folder", command=self.load_images).grid(row=0, column=2, padx=4)
        ttk.Button(actions, text="Previous", command=self.previous_image).grid(row=0, column=3, padx=4)
        ttk.Label(actions, textvariable=self.position_var).grid(row=0, column=4, padx=8)
        ttk.Button(actions, text="Next", command=self.next_image).grid(row=0, column=5, padx=4)

    def _path_row(self, parent, row, label, variable, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, sticky="ew", padx=8)

    # === AI Engine Integration (the modular AAT superpower) ===
    def choose_engine(self):
        selected = filedialog.askopenfilename(title="Select model", filetypes=[("Models", "*.pt *.engine *.onnx"), ("All", "*.*")])
        if selected:
            self.engine_path_var.set(selected)

    def load_engine(self):
        path = self.engine_path_var.get().strip()
        if not path:
            messagebox.showwarning("AAT Viewer", "Choose a model file first.")
            return
        try:
            self.current_engine = get_engine(path)
            names = getattr(self.current_engine, "names", []) or []
            self.engine_status_var.set(f"Loaded: {Path(path).name} ({len(names)} classes)")
        except Exception as exc:
            self.current_engine = None
            self.engine_status_var.set("Failed to load engine")
            messagebox.showerror("AAT Viewer — Engine Load Failed", str(exc))

    def suggest_current(self):
        if self.index < 0 or self.index >= len(self.images):
            messagebox.showinfo("AAT Viewer", "Load an image first.")
            return
        if self.current_engine is None:
            if not self.engine_path_var.get():
                self.choose_engine()
            if self.engine_path_var.get():
                self.load_engine()
            if self.current_engine is None:
                return

        image_path = self.images[self.index]
        try:
            preds = self.current_engine.predict([image_path])
            pred = preds[0] if preds else None
            if not pred or not pred.boxes:
                messagebox.showinfo("AAT Viewer", "Engine returned no detections.")
                return

            added = 0
            for box in pred.boxes:
                x1, y1 = box.x, box.y
                x2 = box.x + box.width
                y2 = box.y + box.height
                shape = AnnotationShape(class_id=box.class_id, kind="detect",
                                        points=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                self.current_shapes.append(shape)
                added += 1
            self.dirty = True
            self.render_current()
            self.status_var.set(f"Added {added} suggestion(s) from engine. Save when ready.")
        except Exception as exc:
            messagebox.showerror("AAT Viewer — Suggest Failed", str(exc))

    # === Rest of viewer (adapted) ===
    def choose_single_image(self):
        selected = filedialog.askopenfilename(title="Select image", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")])
        if selected:
            p = selected
            self.single_image_var.set(p)
            if not self.single_label_var.get():
                guess = guess_label_for_image(Path(p))
                if guess:
                    self.single_label_var.set(str(guess))
            if not self.image_root_var.get():
                self.image_root_var.set(str(Path(p).parent))

    def choose_single_label(self):
        sel = filedialog.askopenfilename(title="Select label", filetypes=[("YOLO labels", "*.txt"), ("All", "*.*")])
        if sel:
            self.single_label_var.set(sel)

    def choose_images(self):
        sel = filedialog.askdirectory(title="Select image folder")
        if sel:
            self.image_root_var.set(sel)
            guess = guess_label_root(Path(sel))
            if guess and not self.label_root_var.get():
                self.label_root_var.set(str(guess))

    def choose_labels(self):
        sel = filedialog.askdirectory(title="Select labels folder")
        if sel:
            self.label_root_var.set(sel)

    def choose_data_yaml(self):
        sel = filedialog.askopenfilename(title="Select data.yaml", filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")])
        if sel:
            self.data_yaml_var.set(sel)
            self._load_data_yaml_metadata()

    def load_single_image(self):
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

    def load_images(self):
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

    def previous_image(self):
        if not self.images or not self._confirm_discard():
            return
        self.index = max(0, self.index - 1)
        self._sync_list()
        self.render_current(force_reload=True)

    def next_image(self):
        if not self.images or not self._confirm_discard():
            return
        self.index = min(len(self.images) - 1, self.index + 1)
        self._sync_list()
        self.render_current(force_reload=True)

    def render_current(self, *, force_reload=False):
        self.canvas.delete("all")
        if self.index < 0 or self.index >= len(self.images):
            self.position_var.set("0 / 0")
            self.canvas.create_text(20, 24, anchor="nw", fill="#dddddd", text="Load a folder or use AI Suggest after loading an engine.")
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

        warnings = []
        label_path = self._label_path_for_current(image_path)
        if force_reload or self.current_image_path != image_path or self.current_label_path != label_path:
            self.current_shapes = read_annotation_shapes(label_path, image_width=w, image_height=h, task=self.annotation_task, warnings=warnings)
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

    def _draw_shape(self, shape, tf):
        color = PALETTE[shape.class_id % len(PALETTE)]
        flat = []
        for x, y in shape.points:
            flat.extend([tf["x"] + x * tf["scale"], tf["y"] + y * tf["scale"]])
        self.canvas.create_polygon(flat, outline=color, fill="", width=2)
        if shape.points:
            x, y = shape.points[0]
            name = self.class_names[shape.class_id] if 0 <= shape.class_id < len(self.class_names) else str(shape.class_id)
            self.canvas.create_text(tf["x"] + x * tf["scale"], tf["y"] + y * tf["scale"] - 10, anchor="sw", fill=color, text=f"{name} ({shape.kind})")

    def save_current(self):
        if self.current_label_path is None or self.current_image_size is None:
            messagebox.showwarning("AAT Viewer", "Load an image before saving.")
            return
        task = self.annotation_task or ("segment" if any(s.kind == "segment" for s in self.current_shapes) else None)
        write_annotation_shapes(self.current_label_path, self.current_shapes,
                                image_width=self.current_image_size[0], image_height=self.current_image_size[1], task=task)
        self.annotation_task = task
        self.dirty = False
        self.render_current()

    # Selection handlers
    def _selection_press(self, event):
        if self.current_transform is None or self.current_image_size is None:
            return
        self.selection_start = (float(event.x), float(event.y))
        if self.selection_rect_id is not None:
            self.canvas.delete(self.selection_rect_id)
        self.selection_rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#ffffff", dash=(4, 3), width=2)

    def _selection_motion(self, event):
        if self.selection_start is None or self.selection_rect_id is None:
            return
        self.canvas.coords(self.selection_rect_id, self.selection_start[0], self.selection_start[1], event.x, event.y)

    def _selection_release(self, event):
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
            self._add_segment_from_rect(rect)

    def _add_segment_from_rect(self, rect):
        if self.current_image_path is None:
            return
        try:
            cid = int(self.edit_class_var.get())
            thresh = float(self.trace_threshold_var.get())
        except ValueError:
            messagebox.showerror("AAT Viewer", "Class ID and threshold must be numeric.")
            return
        with Image.open(self.current_image_path) as src:
            pts = auto_trace_segment_points(src.convert("RGB"), rect, threshold=thresh)
        if pts is None:
            messagebox.showwarning("AAT Viewer", "No traceable foreground found.")
            return
        self.annotation_task = "segment"
        self.current_shapes.append(AnnotationShape(class_id=cid, kind="segment", points=pts))
        self.dirty = True
        self.render_current()

    def _remove_shapes_in_rect(self, rect):
        kept, removed = remove_shapes_intersecting_rect(self.current_shapes, rect)
        if removed <= 0:
            self.status_var.set("No annotation center found inside the box.")
            return
        self.current_shapes = kept
        self.dirty = True
        self.render_current()

    def _confirm_discard(self):
        if not self.dirty:
            return True
        return messagebox.askyesno("AAT Viewer", "Discard unsaved changes?")

    def _sync_list(self):
        self.image_list.selection_clear(0, "end")
        self.image_list.selection_set(self.index)
        self.image_list.see(self.index)

    def _select_from_list(self, _event):
        sel = self.image_list.curselection()
        if not sel:
            return
        if not self._confirm_discard():
            self._sync_list()
            return
        self.index = int(sel[0])
        self.render_current(force_reload=True)

    def _label_path_for_current(self, image_path):
        if self.single_image_mode and self.single_label_var.get():
            return self.single_label_var.get()
        if self.single_image_mode:
            g = guess_label_for_image(Path(image_path))
            return g or Path(image_path).with_suffix(".txt")
        return label_path_for_view(image_path, self.image_root_var.get(), self.label_root_var.get())

    def _load_data_yaml_metadata(self):
        try:
            meta = load_class_metadata(data_yaml=self.data_yaml_var.get())
            self.class_names = meta.names
            from yolo_annotator.annotation_viewer import annotation_task_from_data_yaml
            self.annotation_task = annotation_task_from_data_yaml(self.data_yaml_var.get())
        except Exception as e:
            messagebox.showwarning("AAT Viewer", f"Could not load classes:\n{e}")
            self.class_names = []
            self.annotation_task = None

    def _fit_transform(self, iw, ih, cw, ch):
        cw = max(int(cw), 1)
        ch = max(int(ch), 1)
        scale = min(cw / iw, ch / ih)
        dw = iw * scale
        dh = ih * scale
        return {"scale": scale, "x": (cw - dw) / 2.0, "y": (ch - dh) / 2.0, "width": dw, "height": dh}


def main():
    root = tk.Tk()
    root.geometry("1220x780")
    AATViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()