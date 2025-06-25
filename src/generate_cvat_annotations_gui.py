import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os

class CVATAnnotationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CVAT Annotation Generator")
        
        # Model Path
        tk.Label(root, text="Model Path:").grid(row=0, column=0, padx=5, pady=5)
        self.model_path = tk.Entry(root, width=40)
        self.model_path.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=5, pady=5)
        
        # Config Path
        tk.Label(root, text="Config Path:").grid(row=1, column=0, padx=5, pady=5)
        self.config_path = tk.Entry(root, width=40)
        self.config_path.grid(row=1, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=self.browse_config).grid(row=1, column=2, padx=5, pady=5)
        
        # Output Directory
        tk.Label(root, text="Output Directory:").grid(row=2, column=0, padx=5, pady=5)
        self.output_dir = tk.Entry(root, width=40)
        self.output_dir.grid(row=2, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=self.browse_output).grid(row=2, column=2, padx=5, pady=5)
        
        # Video Directory
        tk.Label(root, text="Video Directory:").grid(row=3, column=0, padx=5, pady=5)
        self.video_dir = tk.Entry(root, width=40)
        self.video_dir.grid(row=3, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=self.browse_video).grid(row=3, column=2, padx=5, pady=5)
        
        # Initial Frames
        tk.Label(root, text="Initial Frames:").grid(row=4, column=0, padx=5, pady=5)
        self.initial_frames = tk.Entry(root, width=10)
        self.initial_frames.insert(0, "8")
        self.initial_frames.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        
        # Final Frames
        tk.Label(root, text="Final Frames:").grid(row=5, column=0, padx=5, pady=5)
        self.final_frames = tk.Entry(root, width=10)
        self.final_frames.insert(0, "8")
        self.final_frames.grid(row=5, column=1, padx=5, pady=5, sticky="w")
        
        # Run Button
        tk.Button(root, text="Generate Annotations", command=self.run_generator, 
                 bg="green", fg="white").grid(row=6, column=1, pady=10)
        
        # Status Label
        self.status = tk.Label(root, text="Ready", fg="blue")
        self.status.grid(row=7, column=1, pady=5)
        
    def browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pt")])
        if path:
            self.model_path.delete(0, tk.END)
            self.model_path.insert(0, path)
            
    def browse_config(self):
        path = filedialog.askopenfilename(filetypes=[("Config Files", "*.json")])
        if path:
            self.config_path.delete(0, tk.END)
            self.config_path.insert(0, path)
            
    def browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.delete(0, tk.END)
            self.output_dir.insert(0, path)
            
    def browse_video(self):
        path = filedialog.askdirectory()
        if path:
            self.video_dir.delete(0, tk.END)
            self.video_dir.insert(0, path)
            
    def validate_inputs(self):
        if not os.path.exists(self.model_path.get()):
            messagebox.showerror("Error", "Invalid model path")
            return False
            
        if not os.path.exists(self.config_path.get()):
            messagebox.showerror("Error", "Invalid config path")
            return False
            
        if not os.path.exists(self.output_dir.get()):
            messagebox.showerror("Error", "Invalid output directory")
            return False
            
        if not os.path.exists(self.video_dir.get()):
            messagebox.showerror("Error", "Invalid video directory")
            return False
            
        try:
            int(self.initial_frames.get())
            int(self.final_frames.get())
        except ValueError:
            messagebox.showerror("Error", "Frames must be integers")
            return False
            
        return True
        
    def run_generator(self):
        if not self.validate_inputs():
            return
            
        cmd = [
            "python", "src/generate_cvat_annotations.py",
            "--model_path", self.model_path.get(),
            "--config_path", self.config_path.get(),
            "--output_dir", self.output_dir.get(),
            "--video_dir", self.video_dir.get(),
            "--initial_frames", self.initial_frames.get(),
            "--final_frames", self.final_frames.get()
        ]
        
        self.status.config(text="Running...", fg="orange")
        self.root.update()
        
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                self.status.config(text="Completed Successfully", fg="green")
                output_path = os.path.abspath(self.output_dir.get())
                messagebox.showinfo("Success", 
                    f"Annotations generated successfully!\n\n"
                    f"Saved in: {output_path}")
            else:
                self.status.config(text="Failed", fg="red")
                messagebox.showerror("Error", f"Process failed:\n{stderr.decode()}")
        except Exception as e:
            self.status.config(text="Failed", fg="red")
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = CVATAnnotationGUI(root)
    root.mainloop()
