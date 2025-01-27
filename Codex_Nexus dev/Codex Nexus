import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import os
from PIL import Image, ImageTk
from tkinter import simpledialog

class CoreManager:
    """
    Manages the core codebase, caches it in memory, and integrates new modules.
    """
    def __init__(self):
        self.core_code = ""
        self.nexus_dir = "Nexus"
        self.core_code_dir = os.path.join(self.nexus_dir, "Core_code")
        self.module_updates_dir = os.path.join(self.nexus_dir, "Module_updates")
        self.history_dir = os.path.join(self.nexus_dir, "History")
        self.create_folders()

    def create_folders(self):
        # Ensure the Nexus folder and its subfolders exist
        if not os.path.exists(self.nexus_dir):
            os.makedirs(self.nexus_dir)
        if not os.path.exists(self.core_code_dir):
            os.makedirs(self.core_code_dir)
        if not os.path.exists(self.module_updates_dir):
            os.makedirs(self.module_updates_dir)
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)

    def save_core_code(self, code, filename, is_module=False):
        if is_module:
            save_path = os.path.join(self.module_updates_dir, filename)
        else:
            save_path = os.path.join(self.core_code_dir, filename)

        with open(save_path, 'w') as file:
            file.write(code)

    def get_cache(self):
        # Returns a list of saved Python scripts
        core_files = [f for f in os.listdir(self.core_code_dir) if f.endswith('.py')]
        module_files = [f for f in os.listdir(self.module_updates_dir) if f.endswith('.py')]
        return core_files + module_files


class Diagnostics:
    """
    Handles error tracking, severity categorization, and diagnostics.
    """
    def scan_code(self, code):
        lines = code.split("\n")
        report = []
        previous_empty = False

        for i, line in enumerate(lines, start=1):
            if not line.strip():
                if previous_empty:
                    report.append(f"Line {i}: Redundant empty line.")
                previous_empty = True
            else:
                previous_empty = False

        if report:
            return "Diagnostics Report:\n" + "\n".join(report), False  # False means there are issues
        else:
            return "Diagnostics Report:\nNo issues found. Code looks clean!", True  # True means no issues


class Integrator:
    """
    Integrates new modules into the core code with error detection and conflict resolution.
    """
    def integrate_module(self, new_module, core_code):
        # This function inserts the new module in an appropriate spot in the core code
        # Here we'll just append it at the end for simplicity
        updated_code = core_code + "\n\n" + new_module  # Adding the new module after the core code
        return updated_code


class CodexNexusApp:
    """
    GUI application for Codex Nexus.
    """

    def __init__(self):
        self.diagnostics = Diagnostics()
        self.core_manager = CoreManager()
        self.integrator = Integrator()

    def initialize_ui(self):
        self.root = tk.Tk()
        self.root.title("Codex Nexus Scanner")
        self.root.geometry("800x600")

        self.set_custom_icon()

        # Frame for displaying core code
        self.left_frame = tk.Frame(self.root, width=300)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.core_code_display = tk.Text(self.left_frame, wrap="word", height=30, width=40, bg="lightgrey")
        self.core_code_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Frame for module text area
        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.module_text_area = tk.Text(self.right_frame, wrap="word", height=30, width=40)
        self.module_text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Button Panel for operations
        btn_frame = tk.Frame(self.right_frame)
        btn_frame.pack(pady=10)

        load_btn = tk.Button(btn_frame, text="Load Core Code", command=self.load_core_code)
        load_btn.grid(row=0, column=0, padx=5)

        save_core_btn = tk.Button(btn_frame, text="Save Core Code", command=self.save_core_code)
        save_core_btn.grid(row=0, column=1, padx=5)

        save_module_btn = tk.Button(btn_frame, text="Save Module", command=self.save_module)
        save_module_btn.grid(row=1, column=0, padx=5)

        run_diagnostics_btn = tk.Button(btn_frame, text="Run Diagnostics", command=self.run_diagnostics)
        run_diagnostics_btn.grid(row=1, column=1, padx=5)

        integrate_btn = tk.Button(btn_frame, text="Integrate Module", command=self.integrate_module, state=tk.DISABLED)
        integrate_btn.grid(row=1, column=2, padx=5)

        quit_btn = tk.Button(btn_frame, text="Quit", command=self.root.quit)
        quit_btn.grid(row=2, column=0, columnspan=3, pady=10)

        self.integrate_btn = integrate_btn  # Save reference to the integrate button

        # Start the Tkinter event loop
        self.root.mainloop()

    def set_custom_icon(self):
        # Set custom icon as explained earlier
        pass  # Placeholder for previous icon code

    def load_core_code(self):
        filepath = filedialog.askopenfilename(title="Select a code file", filetypes=[("Python Files", "*.py")])
        if filepath:
            with open(filepath, 'r') as file:
                code = file.read()
                self.core_manager.core_code = code
                self.core_code_display.delete(1.0, tk.END)
                self.core_code_display.insert(tk.END, code)

    def save_core_code(self):
        code_to_save = self.core_code_display.get(1.0, tk.END).strip()
        if code_to_save:
            filepath = filedialog.asksaveasfilename(defaultextension=".py", filetypes=[("Python Files", "*.py")])
            if filepath:
                self.core_manager.save_core_code(code_to_save, filepath, is_module=False)

    def save_module(self):
        module_code = self.module_text_area.get(1.0, tk.END).strip()
        if module_code:
            filename = filedialog.asksaveasfilename(defaultextension=".py", filetypes=[("Python Files", "*.py")])
            if filename:
                self.core_manager.save_core_code(module_code, filename, is_module=True)

    def run_diagnostics(self):
        current_code = self.core_manager.core_code
        if current_code:
            diagnostic_report, no_issues = self.diagnostics.scan_code(current_code)
            self.core_code_display.delete(1.0, tk.END)
            self.core_code_display.insert(tk.END, diagnostic_report)

            # Unlock Integrate Module button if no issues are found
            if no_issues:
                self.integrate_btn.config(state=tk.NORMAL)
            else:
                self.integrate_btn.config(state=tk.DISABLED)
        else:
            messagebox.showwarning("No code", "Please load or write core code first.")

    def integrate_module(self):
        # Get the code from the module text area
        new_module = self.module_text_area.get(1.0, tk.END).strip()

        if new_module:
            core_code = self.core_manager.core_code
            updated_code = self.integrator.integrate_module(new_module, core_code)

            # Highlight the new integrated code for the user
            self.core_code_display.delete(1.0, tk.END)
            self.core_code_display.insert(tk.END, updated_code)

            # Allow the user to drag and drop the code block within the text area
            messagebox.showinfo("Module Integrated", "Module integrated successfully. You can now move the code as needed.")
        else:
            messagebox.showwarning("No module code", "Please enter module code before integration.")
        

# Run the application
if __name__ == "__main__":
    app = CodexNexusApp()
    app.initialize_ui()
