"""Modern CustomTkinter GUI for Arrington Annotation Tool (AAT).

This provides a first-class graphical interface while keeping the CLI as an equally important way to use the tool.
"""

from __future__ import annotations

try:
    import customtkinter as ctk
except ImportError:
    raise ImportError(
        "CustomTkinter is required for the modern GUI.\n"
        "Install it with: pip install 'arrington-annotation-tool[gui]'"
    ) from None
from tkinter import messagebox
from pathlib import Path
import threading

from aat.generate import GenerateConfig, generate_detect_dataset


class AATGuiApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Arrington Annotation Tool (AAT)")
        self.geometry("1100x700")
        self.minsize(900, 600)

        # Set appearance
        ctk.set_appearance_mode("System")  # Options: "Light", "Dark", "System"
        ctk.set_default_color_theme("blue")

        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # === Sidebar ===
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="AAT", 
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Arrington Annotation Tool", 
            font=ctk.CTkFont(size=12)
        )
        self.sidebar_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        # Navigation buttons
        self.home_button = ctk.CTkButton(
            self.sidebar_frame, text="Home", command=self.show_home
        )
        self.home_button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.generate_button = ctk.CTkButton(
            self.sidebar_frame, text="Generate Dataset", command=self.show_generate
        )
        self.generate_button.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        self.viewer_button = ctk.CTkButton(
            self.sidebar_frame, text="Open Viewer", command=self.launch_viewer
        )
        self.viewer_button.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

        # Appearance mode
        self.appearance_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance:")
        self.appearance_label.grid(row=7, column=0, padx=20, pady=(10, 0))

        self.appearance_option = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["Light", "Dark", "System"],
            command=self.change_appearance_mode
        )
        self.appearance_option.grid(row=8, column=0, padx=20, pady=10, sticky="ew")
        self.appearance_option.set("System")

        # === Main content area ===
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Start with Home view
        self.current_view = None
        self.show_home()

    def change_appearance_mode(self, new_mode: str):
        ctk.set_appearance_mode(new_mode)

    def clear_main_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    # ====================== VIEWS ======================

    def show_home(self):
        self.clear_main_frame()

        title = ctk.CTkLabel(
            self.main_frame, 
            text="Welcome to Arrington Annotation Tool", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=30)

        desc = ctk.CTkLabel(
            self.main_frame,
            text="A modular, powerful tool for AI-assisted YOLO annotation.\n\n"
                 "Use the sidebar to generate datasets from models or open the advanced viewer.",
            font=ctk.CTkFont(size=14)
        )
        desc.pack(pady=10)

        # Quick action buttons
        btn_frame = ctk.CTkFrame(self.main_frame)
        btn_frame.pack(pady=40)

        ctk.CTkButton(
            btn_frame, text="Generate Dataset", width=200, height=40,
            command=self.show_generate
        ).pack(side="left", padx=15)

        ctk.CTkButton(
            btn_frame, text="Open Viewer", width=200, height=40,
            command=self.launch_viewer
        ).pack(side="left", padx=15)

    def show_generate(self):
        self.clear_main_frame()

        title = ctk.CTkLabel(
            self.main_frame, 
            text="Generate YOLO Dataset from Model", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(pady=20)

        # Form frame
        form = ctk.CTkFrame(self.main_frame)
        form.pack(padx=40, pady=10, fill="x")

        # Images folder
        ctk.CTkLabel(form, text="Images Folder:").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        self.images_var = ctk.StringVar()
        ctk.CTkEntry(form, textvariable=self.images_var, width=400).grid(row=0, column=1, padx=5)
        ctk.CTkButton(form, text="Browse", width=80, command=self.browse_images).grid(row=0, column=2, padx=5)

        # Model path
        ctk.CTkLabel(form, text="Model / Engine:").grid(row=1, column=0, sticky="w", padx=10, pady=8)
        self.model_var = ctk.StringVar()
        ctk.CTkEntry(form, textvariable=self.model_var, width=400).grid(row=1, column=1, padx=5)
        ctk.CTkButton(form, text="Browse", width=80, command=self.browse_model).grid(row=1, column=2, padx=5)

        # Output folder
        ctk.CTkLabel(form, text="Output Folder:").grid(row=2, column=0, sticky="w", padx=10, pady=8)
        self.output_var = ctk.StringVar()
        ctk.CTkEntry(form, textvariable=self.output_var, width=400).grid(row=2, column=1, padx=5)
        ctk.CTkButton(form, text="Browse", width=80, command=self.browse_output).grid(row=2, column=2, padx=5)

        # Sliders
        ctk.CTkLabel(form, text="Confidence:").grid(row=3, column=0, sticky="w", padx=10, pady=8)
        self.conf_slider = ctk.CTkSlider(form, from_=0.05, to=0.95, number_of_steps=90)
        self.conf_slider.set(0.25)
        self.conf_slider.grid(row=3, column=1, sticky="ew", padx=5)
        self.conf_label = ctk.CTkLabel(form, text="0.25")
        self.conf_label.grid(row=3, column=2)
        self.conf_slider.configure(command=lambda v: self.conf_label.configure(text=f"{v:.2f}"))

        ctk.CTkLabel(form, text="IoU:").grid(row=4, column=0, sticky="w", padx=10, pady=8)
        self.iou_slider = ctk.CTkSlider(form, from_=0.1, to=0.9, number_of_steps=80)
        self.iou_slider.set(0.45)
        self.iou_slider.grid(row=4, column=1, sticky="ew", padx=5)
        self.iou_label = ctk.CTkLabel(form, text="0.45")
        self.iou_label.grid(row=4, column=2)
        self.iou_slider.configure(command=lambda v: self.iou_label.configure(text=f"{v:.2f}"))

        # Run button
        self.run_button = ctk.CTkButton(
            self.main_frame, 
            text="Generate Dataset", 
            height=40, 
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self.run_generation
        )
        self.run_button.pack(pady=25)

        # Log area
        self.log_text = ctk.CTkTextbox(self.main_frame, height=200, width=700)
        self.log_text.pack(padx=20, pady=10, fill="both", expand=True)

    def browse_images(self):
        folder = ctk.filedialog.askdirectory(title="Select Images Folder")
        if folder:
            self.images_var.set(folder)

    def browse_model(self):
        filetypes = [("Model files", "*.pt *.engine *.onnx"), ("All files", "*.*")]
        path = ctk.filedialog.askopenfilename(title="Select Model", filetypes=filetypes)
        if path:
            self.model_var.set(path)

    def browse_output(self):
        folder = ctk.filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_var.set(folder)

    def run_generation(self):
        images = self.images_var.get()
        model = self.model_var.get()
        output = self.output_var.get()

        if not all([images, model, output]):
            messagebox.showerror("Missing Information", "Please fill in Images, Model, and Output folders.")
            return

        self.run_button.configure(state="disabled", text="Generating...")
        self.log_text.delete("0.0", "end")
        self.log_text.insert("end", "Starting generation...\n")

        def progress_callback(info):
            self.log_text.insert("end", f"{info}\n")
            self.log_text.see("end")

        def worker():
            try:
                cfg = GenerateConfig(
                    images_root=Path(images),
                    model_path=Path(model),
                    output_root=Path(output),
                    confidence=self.conf_slider.get(),
                    iou=self.iou_slider.get(),
                    force=True
                )
                result = generate_detect_dataset(cfg, progress=progress_callback)
                self.log_text.insert("end", f"\nDone! Processed {result.processed} images.\n")
                messagebox.showinfo("Success", f"Dataset generated successfully!\n\n{result.processed} images processed.")
            except Exception as e:
                self.log_text.insert("end", f"\nError: {e}\n")
                messagebox.showerror("Error", str(e))
            finally:
                self.run_button.configure(state="normal", text="Generate Dataset")

        threading.Thread(target=worker, daemon=True).start()

    def launch_viewer(self):
        """Launch the powerful existing Tkinter viewer."""
        try:
            from aat.viewer import main as viewer_main
            threading.Thread(target=viewer_main, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch viewer:\n{e}")


def main():
    app = AATGuiApp()
    app.mainloop()


if __name__ == "__main__":
    main()