import sys
import os
from pathlib import Path

class AddPath:
    """
    Adds 'src/modules' to sys.path, automatically detects your project root from notebooks.
    """

    def __init__(self):
        self.BASE_DIR = self.get_project_root()
        self.module_path = self.find_module_path()
        self.add_to_sys_path()

    def get_project_root(self):
        """
        Returns the assumed project root (parent of notebooks/).
        """
        cwd = Path.cwd()
        if cwd.name.lower() == 'notebooks':
            return cwd.parent
        return cwd  # fallback if not running inside notebooks

    def find_module_path(self):
        """
        Constructs 'src/modules' path relative to BASE_DIR
        """
        module_path = self.BASE_DIR / "src" / "modules"
        if module_path.exists():
            print(f"✅ Using module path: {module_path}")
            return str(module_path)
        else:
            print("❌ 'src/modules' directory not found.")
            return None

    def add_to_sys_path(self):
        """
        Adds module path to sys.path if not already present.
        """
        if self.module_path and self.module_path not in sys.path:
            sys.path.append(self.module_path)
            print(f"🔄 Module path added to sys.path: {self.module_path}")
        elif self.module_path:
            print(f"✅ Module path already in sys.path: {self.module_path}")
        else:
            print("⚠ Skipped sys.path update because module path was not found.")

if __name__=="__main__":
    from path_manager import AddPath
    AddPath()


# import sys
# import os
# from pathlib import Path

# class AddPath:
#     def __init__(self):
#         self.base_dir = Path(__file__).resolve().parent  # This is src/
#         self.module_path = self.base_dir / "modules"
#         self.add_module_path()

#     def add_module_path(self):
#         if not self.module_path.exists():
#             print(f"❌ Module path does not exist: {self.module_path}")
#             return

#         if str(self.module_path) not in sys.path:
#             sys.path.append(str(self.module_path))
#             print(f"✅ Module path added to sys.path: {self.module_path}")
#         else:
#             print(f"⚠ Module path already in sys.path: {self.module_path}")

# # from path_manager import AddPath
# # AddPath()