

import tkinter as tk
from tkinter import ttk

class ProgressGUI:
    def __init__(self, total_files):
        self.window = tk.Tk()
        self.window.title("Processing Volumes")
        self.window.configure(bg="#2e2e2e")  # Dark background
        self.window.geometry("500x150")
        self.cancelled = False

        style = ttk.Style(self.window)
        style.theme_use("default")
        style.configure("TProgressbar",
                        troughcolor="#3a3a3a",
                        background="#00bcd4",
                        thickness=20)

        self.label = tk.Label(self.window, text="Starting...", fg="white", bg="#2e2e2e", font=("Arial", 12))
        self.label.pack(pady=10)

        self.pb = ttk.Progressbar(self.window, orient="horizontal", mode="determinate",
                                  maximum=total_files, length=400)
        self.pb.pack(pady=5)

        self.cancel_btn = tk.Button(self.window, text="Cancel", bg="#d32f2f", fg="white",
                                    command=self.cancel, font=("Arial", 10))
        self.cancel_btn.pack(pady=10)

    def update(self, index, filename):
        if not self.cancelled:
            self.pb["value"] = index
            self.label.config(text=f"Processing: {filename}")
            self.window.update_idletasks()

    def cancel(self):
        self.cancelled = True
        self.label.config(text="Cancelled by user.")

    def is_cancelled(self):
        return self.cancelled

    def close(self):
        self.window.destroy()


# Usage Example:
# gui = ProgressGUI(len(files))
# for idx, file in enumerate(files):
#     if gui.is_cancelled():
#         break
#     gui.update(idx + 1, file)
# gui.close()
















# def show_progress_gui(total_files):
#     window = tk.Tk()
#     window.title("3D Volume Processing Progress")
#     pb = ttk.Progressbar(window, orient="horizontal", mode="determinate", maximum=total_files, length=400)
#     pb.pack(padx=30, pady=15)
#     label = tk.Label(window, text="Starting...")
#     label.pack(pady=5)

#     def update(index, fname):
#         pb["value"] = index
#         label.config(text=f"Processing: {fname}")
#         window.update_idletasks()

#     return window, update
